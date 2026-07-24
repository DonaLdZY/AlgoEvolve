from __future__ import annotations

from types import SimpleNamespace

from llm import gemini as gemini_backend
from llm import openai as openai_backend


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
