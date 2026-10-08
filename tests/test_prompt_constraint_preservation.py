from types import SimpleNamespace

from agents.prompt_cache import looks_same_context, stable_data_context
from llm.openai import (
    CACHE_FRIENDLY_SYSTEM, TASK_CONTEXT_END_MARKER, _apply_deepseek_request_options,
    _build_messages, _create_with_retry,
)


def test_differing_tail_constraints_are_not_duplicates():
    prefix = "Shared task background. " * 300
    task = prefix + "Maximum vehicle weight: 500 kg."
    preview = prefix + "Maximum vehicle weight: 800 kg."
    assert not looks_same_context(task, preview)
    combined = stable_data_context(task, preview)
    assert "500 kg" in combined and "800 kg" in combined
    assert looks_same_context(task, task)


def test_preprocessing_transformers_do_not_imply_neural_models():
    from agents.prompt_policy import infer_method_family

    assert infer_method_family("", code="from sklearn.compose import ColumnTransformer\nfrom sklearn.svm import SVC\n") == "linear_or_kernel"
    assert infer_method_family("", code="from sklearn.preprocessing import PowerTransformer\nfrom sklearn.ensemble import HistGradientBoostingClassifier\n") == "tree_boosting"
    assert infer_method_family("", code="from torch.nn import Transformer\nmodel = Transformer()\n") == "deep_learning"


def test_gpt_reasoning_effort_is_not_deepseek_remapped():
    params = {"temperature": 1.0}
    _apply_deepseek_request_options(params, SimpleNamespace(reasoning_effort="xhigh"), "gpt-5.6-luna", None)
    assert params == {"reasoning_effort": "xhigh"}


def test_cache_key_reuses_only_the_stable_prefix(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_PROMPT_CACHE_KEY_MODE", "enabled")
    stage = SimpleNamespace(reasoning_effort="xhigh", base_url="https://compatible.example/v1")
    keys = []
    for suffix in ("draft node", "review another node"):
        params = {"messages": [{"role": "system", "content": CACHE_FRIENDLY_SYSTEM},
                               {"role": "user", "content": "# Task description\nAuthoritative constraints"},
                               {"role": "user", "content": suffix}]}
        _apply_deepseek_request_options(params, stage, "gpt-5.6-luna", None)
        keys.append(params["prompt_cache_key"])
    assert keys[0] == keys[1]


def test_cache_key_uses_actual_combined_message_layout_without_dropping_constraints(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_PROMPT_CACHE_KEY_MODE", "enabled")
    stage = SimpleNamespace(reasoning_effort="xhigh", base_url="https://compatible.example/v1")
    for marker in ("", "\n" + TASK_CONTEXT_END_MARKER):
        keys = []
        for instructions, action, limit in (("Generate code", "draft A", 500),
                                             ("Review code", "review B", 500),
                                             ("Review code", "review B", 800)):
            task = f"# Task description\n## Rules\nVehicle capacity: {limit} kg." + marker
            messages = _build_messages(instructions, task + "\n# Implementation\n" + action)
            assert len(messages) == 2
            original = [dict(message) for message in messages]
            params = {"messages": messages}
            _apply_deepseek_request_options(params, stage, "gpt-5.6-luna", None)
            assert params["messages"] == original
            assert task in messages[1]["content"]
            assert instructions in messages[1]["content"] and action in messages[1]["content"]
            keys.append(params["prompt_cache_key"])
        assert keys[0] == keys[1]
        assert keys[1] != keys[2]


def test_unsupported_cache_key_does_not_drop_requested_reasoning():
    calls = []
    response = SimpleNamespace(choices=[])

    def create(**kwargs):
        calls.append(kwargs)
        if "prompt_cache_key" in kwargs:
            raise TypeError("unexpected keyword argument 'prompt_cache_key'")
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = _create_with_retry(client, {"model": "gpt-5.6-luna", "reasoning_effort": "xhigh", "prompt_cache_key": "shared"},
                                label="test", stage=SimpleNamespace(network_retry_max_attempts=1))
    assert result is response
    assert len(calls) == 2
    assert calls[1]["reasoning_effort"] == "xhigh"
