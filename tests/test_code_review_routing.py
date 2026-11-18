from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents import code_review_agent
from engine.search_node import SearchNode


def _agent(*, escalate: bool = True):
    retries = SimpleNamespace(
        code_review_max_attempts=2,
        code_review_delay_seconds=0.0,
        code_review_model_role="feedback",
        code_review_escalate_to_code=escalate,
    )
    return SimpleNamespace(
        task_desc="Demo task",
        autorealize_context="",
        cfg=SimpleNamespace(pretrain_model_dir=""),
        acfg=SimpleNamespace(
            use_diff_mode=True,
            generate_submission=False,
            retries=retries,
            feedback=SimpleNamespace(model="fast-review", temp=0.0),
            code=SimpleNamespace(model="main-code", temp=0.0),
        ),
    )


def _node() -> SearchNode:
    return SearchNode(code="print(1)", plan="demo", parent=None, stage="draft")


def test_approved_review_null_patch_passes_real_provider_validation(monkeypatch):
    import json
    import jsonschema
    from llm.openai import _validated_json_content, _build_json_mode_messages

    response = {'needs_revision': False, 'reasoning': 'No critical issue.', 'revised_code': None}
    messages = _build_json_mode_messages('Review the code.', 'print(1)', code_review_agent.CODE_REVIEW_SPEC)
    assert '"revised_code":null' in '\n'.join(m['content'] for m in messages)
    calls = []
    def actual_validation(**kwargs):
        calls.append(kwargs)
        return _validated_json_content(json.dumps(response), code_review_agent.CODE_REVIEW_SPEC, 'stop')
    monkeypatch.setattr(code_review_agent, 'query', actual_validation)
    assert code_review_agent.run(_agent(), _node()) == 'print(1)'
    assert len(calls) == 1
    with pytest.raises(jsonschema.ValidationError):
        _validated_json_content(json.dumps({**response, 'revised_code': 123}), code_review_agent.CODE_REVIEW_SPEC, 'stop')


def test_code_review_uses_feedback_role_first(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        return {"needs_revision": False, "reasoning": "No concrete P0/P1 issue."}

    monkeypatch.setattr(code_review_agent, "query", fake_query)
    code = code_review_agent.run(_agent(), _node())

    assert code == "print(1)"
    assert [call["stage_name"] for call in calls] == ["feedback"]
    assert [call["stage_id"] for call in calls] == ["code_review"]
    assert calls[0]["model"] == "fast-review"


def test_deepseek_code_review_requests_json_mode_on_the_first_call(monkeypatch) -> None:
    calls: list[dict] = []
    agent = _agent(escalate=False)
    agent.acfg.feedback.model = "deepseek-v4.1-flash"

    def fake_query(**kwargs):
        calls.append(kwargs)
        return {"needs_revision": False, "reasoning": "No critical issue."}

    monkeypatch.setattr(code_review_agent, "query", fake_query)
    assert code_review_agent.run(agent, _node()) == "print(1)"
    assert calls[0]["structured_output_mode"] == "json"


def test_unpatchable_feedback_review_escalates_once(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        if kwargs["stage_name"] == "feedback":
            return {
                "needs_revision": True,
                "reasoning": "A critical issue exists, but the patch is missing.",
                "revised_code": None,
            }
        return {"needs_revision": False, "reasoning": "Main review approves the code."}

    monkeypatch.setattr(code_review_agent, "query", fake_query)
    code = code_review_agent.run(_agent(), _node())

    assert code == "print(1)"
    assert [call["stage_name"] for call in calls] == ["feedback", "code"]


@pytest.mark.parametrize("response", [
    {}, {"needs_revision": "false"},
    {"needs_revision": True, "reasoning": "Holdout accessed", "revised_code": None},
    {"needs_revision": True, "revised_code": "print(2)"},
    {"needs_revision": True, "revised_code": (
        "<<<<<<< SEARCH\nprint(1)\n=======\nprint(2)\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\nmissing_code()\n=======\nfixed()\n>>>>>>> REPLACE\n")},
])
def test_unresolved_review_never_returns_executable_original(monkeypatch, response):
    monkeypatch.setattr(code_review_agent, "query", lambda **kwargs: response)
    with pytest.raises(code_review_agent.CodeReviewFailed):
        code_review_agent.run(_agent(), _node())


def test_review_provider_failure_blocks_candidate(monkeypatch):
    def fail(**kwargs):
        raise TimeoutError("provider unavailable")
    monkeypatch.setattr(code_review_agent, "query", fail)
    with pytest.raises(code_review_agent.CodeReviewFailed):
        code_review_agent.run(_agent(), _node())


def test_focused_repair_preserves_features_globals_and_evaluator():
    source = "CONFIG = 1\ndef features(data): return data * 2\ndef train(data, artifact_dir): return None\n"
    fixed = source.replace("return None", "return artifact_dir")
    assert code_review_agent.validate_review_edit_scope(source, fixed, ["train"]) == fixed
    for mutation in (fixed.replace("data * 2", "data"), fixed.replace("CONFIG = 1", "CONFIG = 2")):
        with pytest.raises(code_review_agent.CodeReviewFailed, match="protected code"):
            code_review_agent.validate_review_edit_scope(source, mutation, ["train"])


def test_retry_receives_failed_patch_evidence(monkeypatch):
    calls = []
    def fake_query(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"needs_revision": True, "reasoning": "Repair an interface", "revised_code": None}
        return {"needs_revision": False, "reasoning": "No issue"}
    monkeypatch.setattr(code_review_agent, "query", fake_query)
    code_review_agent.run(_agent(), _node())
    assert "Required revision has no applicable patch" in calls[1]["user_message"]
