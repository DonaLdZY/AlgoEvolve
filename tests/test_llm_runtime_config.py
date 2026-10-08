from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm import gemini as gemini_backend
from llm import openai as openai_backend


def test_request_context_allots_output_after_complete_input(monkeypatch):
    received = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: received.append(kwargs) or "ok")))
    stage = SimpleNamespace(context_window_tokens=278528)
    params = {"messages": [{"role": "user", "content": "original constraint"}], "max_tokens": 278528}
    monkeypatch.setattr(openai_backend, "estimate_text_tokens", lambda _: 60000)
    assert openai_backend._create_with_retry(client, params, label="test", stage=stage) == "ok"
    assert received[0]["max_tokens"] == 278528 - 60000 - int(278528 * .18)
    assert received[0]["messages"] == params["messages"]
    monkeypatch.setattr(openai_backend, "estimate_text_tokens", lambda _: 278000)
    with pytest.raises(ValueError, match="context budget"):
        openai_backend._create_with_retry(client, params, label="test", stage=stage)
    assert len(received) == 1


def test_normal_requests_enforce_32768_output_floor_for_both_backends() -> None:
    stage = SimpleNamespace(max_tokens=4096, minimum_output_tokens=32768)

    assert openai_backend._resolve_max_tokens(3072, stage) == 32768
    assert gemini_backend._resolve_max_tokens(3072, stage) == 32768
    assert openai_backend._resolve_max_tokens(65536, stage) == 65536
    assert gemini_backend._resolve_max_tokens(65536, stage) == 65536
    assert openai_backend._resolve_max_tokens(None, SimpleNamespace(max_tokens=None)) == 32768
    assert gemini_backend._resolve_max_tokens(None, SimpleNamespace(max_tokens=None)) == 32768


def test_openai_retry_policy_uses_stage_config(monkeypatch) -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    class Completions:
        def create(self, **_kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                raise TimeoutError("temporary timeout")
            return "ok"

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    stage = SimpleNamespace(
        network_retry_max_attempts=3,
        network_retry_base_sleep_seconds=0.25,
        network_retry_max_sleep_seconds=0.4,
    )
    monkeypatch.setattr(openai_backend.time, "sleep", sleeps.append)

    result = openai_backend._create_with_retry(
        client,
        {"model": "demo"},
        label="test",
        stage=stage,
    )

    assert result == "ok"
    assert calls["count"] == 3
    assert sleeps == [0.25, 0.4]


def test_continuation_overlap_window_is_configurable() -> None:
    overlap = "0123456789abcdefghij"
    assert openai_backend._append_with_overlap(f"abc{overlap}", f"{overlap}def", 20) == f"abc{overlap}def"
    assert openai_backend._append_with_overlap(f"abc{overlap}", f"{overlap}def", 15) == f"abc{overlap}{overlap}def"


def test_deepseek_beta_uses_chat_prefix_completion_and_continuation_clears_marker() -> None:
    messages = openai_backend._prompt_to_messages(
        {
            "system": "You are a coding agent.",
            "user": "Implement the task.",
            "assistant": "Let me solve this systematically.\n",
        },
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/beta",
    )

    assert messages[-1] == {
        "role": "assistant",
        "content": "Let me solve this systematically.\n",
        "prefix": True,
    }
    continuation = openai_backend._build_continuation_messages(
        messages,
        "partial output",
        max_tokens=100,
        round_index=1,
    )
    assert "prefix" not in continuation[-2]
    assert continuation[-2]["content"].endswith("partial output")
    assert continuation[-1]["role"] == "user"


def test_deepseek_non_beta_does_not_send_prefix_extension() -> None:
    messages = openai_backend._prompt_to_messages(
        {"user": "Implement the task.", "assistant": "Start here:"},
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
    )

    assert "prefix" not in messages[-1]


def test_deepseek_beta_structured_output_does_not_send_prefix_extension() -> None:
    messages = openai_backend._prompt_to_messages(
        {
            "system": "Return valid JSON.",
            "user": "Return the structured result.",
            "assistant": "{",
        },
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/beta",
        structured_output=True,
    )

    assert "prefix" not in messages[-1]


def test_deepseek_thinking_options_use_top_level_effort_and_omit_ignored_sampling() -> None:
    stage = SimpleNamespace(reasoning_effort="xhigh")
    params = {
        "temperature": 0.2,
        "top_p": 0.8,
        "presence_penalty": 0.3,
        "frequency_penalty": 0.4,
    }

    openai_backend._apply_deepseek_request_options(
        params,
        stage,
        "deepseek-v4-pro",
        None,
    )

    assert params == {"reasoning_effort": "max"}

    disabled = {"temperature": 0.2}
    openai_backend._apply_deepseek_request_options(
        disabled,
        stage,
        "deepseek-v4-pro",
        False,
    )
    assert disabled == {
        "temperature": 0.2,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def test_deepseek_resource_limited_completion_retries(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    class Completions:
        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="insufficient_system_resource" if calls == 1 else "stop"
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    stage = SimpleNamespace(
        network_retry_max_attempts=2,
        network_retry_base_sleep_seconds=0.1,
        network_retry_max_sleep_seconds=0.1,
    )
    monkeypatch.setattr(openai_backend.time, "sleep", sleeps.append)

    result = openai_backend._create_with_retry(
        client,
        {"model": "deepseek-v4-pro"},
        label="resource-test",
        stage=stage,
    )

    assert result.choices[0].finish_reason == "stop"
    assert calls == 2
    assert sleeps == [0.1]


def test_gemini_client_uses_stage_endpoint_and_timeout(monkeypatch) -> None:
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(gemini_backend.genai, "Client", fake_client)
    stage = SimpleNamespace(
        api_key="test-key",
        base_url="https://example.invalid",
        request_timeout_seconds=12.5,
    )

    gemini_backend._setup_gemini_client(stage)

    assert captured["api_key"] == "test-key"
    assert captured["http_options"]["base_url"] == "https://example.invalid"
    assert captured["http_options"]["timeout"] == 12500


@pytest.mark.parametrize("marker", ["> \u274c \u7f51\u7edc\u51fa\u73b0\u5f02\u5e38", "> Network error occurred\r\n"])
def test_gateway_false_stop_preserves_partial_whitespace(marker):
    partial = "```python\r\ndef predict():\r\n"
    assert openai_backend._normalize_stream_completion(partial + marker, "stop") == (partial, "provider_interrupted")


@pytest.mark.parametrize("text", [
    "The provider may report Network error occurred.",
    "Example:\n> Network error occurred\nThen retry.",
    '```python\nmessage = """\n> Network error occurred\n"""\n```',
])
def test_gateway_marker_does_not_strip_embedded_prose(text):
    assert openai_backend._normalize_stream_completion(text, "stop") == (text, "stop")


def _mock_interrupted_generation(monkeypatch, tmp_path, responses, rounds=2):
    stage = SimpleNamespace(model="gpt-5.6-luna", api_key="test-only", base_url="https://example.invalid/v1",
                            max_tokens=32768, minimum_output_tokens=32768, reasoning_effort="xhigh",
                            continuation_max_rounds=rounds, generation_max_retries=1)
    cfg = SimpleNamespace(agent=SimpleNamespace(code=stage), log_dir=tmp_path)
    requests, rows = [], []
    iterator = iter(responses)

    def collect(client, params, **kwargs):
        requests.append(params)
        return next(iterator), "stop", {"prompt_tokens": 0, "completion_tokens": 10}, 1.0

    monkeypatch.setattr(openai_backend, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(openai_backend, "_collect_stream_response", collect)
    monkeypatch.setattr(openai_backend, "log_llm_usage", lambda **kwargs: rows.append(kwargs))
    return cfg, requests, rows


def test_query_keeps_requested_model_separate_from_provider_alias(monkeypatch, tmp_path):
    cfg, _, rows = _mock_interrupted_generation(monkeypatch, tmp_path, [])
    response = SimpleNamespace(model="gpt-5.6-terra", usage=SimpleNamespace(prompt_tokens=200, completion_tokens=4),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ok", tool_calls=[]))])
    captured = []

    def complete(client, params, **kwargs):
        captured.append(params)
        return response

    monkeypatch.setattr(openai_backend, "_create_with_retry", complete)
    output = openai_backend.query("Review constraints", "Candidate", cfg=cfg, model="gpt-5.6-luna")
    assert output[0] == "ok"
    assert captured[0]["model"] == rows[0]["model"] == "gpt-5.6-luna"
    assert rows[0]["response"].model == "gpt-5.6-terra"


def test_false_stop_continues_exact_original_task_and_partial(monkeypatch, tmp_path):
    partial = "```python\ndef predict():\n"
    cfg, requests, rows = _mock_interrupted_generation(
        monkeypatch, tmp_path, [partial + "> Network error occurred", "    return 42\n```"])
    prompt = [{"role": "system", "content": "Preserve every constraint."},
              {"role": "user", "content": "Use only past observations; keep all output keys."}]
    result = openai_backend.generate(prompt, cfg)
    assert result == partial + "    return 42\n```"
    assert requests[1]["messages"][:2] == requests[0]["messages"] == prompt
    assert requests[1]["messages"][-2] == {"role": "assistant", "content": partial}
    assert "provider connection" in requests[1]["messages"][-1]["content"]
    assert all(request["max_tokens"] == 32768 and request["reasoning_effort"] == "xhigh" for request in requests)
    assert rows[0]["parsed_ok"] is False
    assert rows[0]["finish_reason"] == "provider_interrupted"
    assert rows[0]["extra"]["provider_finish_reason"] == "stop"
    assert rows[1]["parsed_ok"] is True


def test_provider_interruption_exhaustion_is_bounded_and_not_success(monkeypatch, tmp_path):
    cfg, requests, rows = _mock_interrupted_generation(
        monkeypatch, tmp_path, ["partial\n> Network error occurred"] * 3)
    with pytest.raises(RuntimeError, match="bounded continuation"):
        openai_backend.generate("Implement the complete task", cfg)
    assert len(requests) == 3
    assert all(row["parsed_ok"] is False for row in rows)


def test_structured_interruption_is_never_returned_as_json(monkeypatch, tmp_path):
    cfg, requests, rows = _mock_interrupted_generation(monkeypatch, tmp_path, ['{"score":\n> Network error occurred'])
    with pytest.raises(RuntimeError, match="interrupted structured output"):
        openai_backend.generate("Return the score", cfg, json_schema={"type": "object"})
    assert len(requests) == 1
    assert rows[0]["parsed_ok"] is False


def test_continuation_thinking_does_not_erase_preceding_code(monkeypatch, tmp_path):
    partial = '```python\ndef predict():\n    marker = "</think>"\n'
    cfg, requests, _ = _mock_interrupted_generation(monkeypatch, tmp_path,
        [partial + "> Network error occurred", "<think>Continue the function.</think>    return marker\n```"])
    assert openai_backend.generate("Implement predict", cfg) == partial + "    return marker\n```"
    assert requests[1]["messages"][-2]["content"] == partial
