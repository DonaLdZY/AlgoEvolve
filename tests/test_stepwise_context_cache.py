from types import SimpleNamespace

from agents.coder import stepwise_coder
from agents.prompt_cache import TASK_CONTEXT_END_MARKER, task_section
from llm.openai import AGENT_INSTRUCTIONS_TITLE, _cache_friendly_messages, _prompt_to_messages


def _agent(*, max_tokens: int = 90000, keep_steps: int = 2, workspace=None):
    return SimpleNamespace(
        acfg=SimpleNamespace(
            draft=SimpleNamespace(
                stepwise_context_max_tokens=max_tokens,
                stepwise_compaction_keep_recent_steps=keep_steps,
                stepwise_compaction_max_tokens=1024,
            )
        ),
        cfg=SimpleNamespace(workspace_dir=workspace) if workspace is not None else object(),
    )


def test_stepwise_messages_append_without_rewriting_previous_prefix() -> None:
    conversation = stepwise_coder.StepwiseConversation(
        base_messages=[
            {"role": "system", "content": "fixed-system"},
            {"role": "user", "content": "fixed-task-context"},
            {"role": "assistant", "content": "fixed-ack"},
        ]
    )
    agent = _agent()

    first = conversation.messages_for("stage-one", agent)
    conversation.record("stage-one", "stage-one-result")
    second = conversation.messages_for("stage-two", agent)

    assert second[: len(first)] == first
    assert second[-2:] == [
        {"role": "assistant", "content": "stage-one-result"},
        {"role": "user", "content": "stage-two"},
    ]


def test_stepwise_compaction_keeps_stable_prefix_and_recent_turns(monkeypatch) -> None:
    conversation = stepwise_coder.StepwiseConversation(
        base_messages=[
            {"role": "system", "content": "fixed-system"},
            {"role": "user", "content": "fixed-task-context"},
            {"role": "assistant", "content": "fixed-ack"},
        ]
    )
    conversation.record("old-stage", "old-result " * 100)
    conversation.record("recent-stage", "recent-result " * 100)
    monkeypatch.setattr(
        stepwise_coder,
        "generate",
        lambda **_: "Exact identifiers and evaluator contracts from old-stage.",
    )

    messages = conversation.messages_for(
        "next-stage",
        _agent(max_tokens=1, keep_steps=1),
    )

    assert messages[:3] == conversation.base_messages
    assert conversation.compacted_history.startswith("Exact identifiers")
    assert "old-result" not in str(messages)
    assert {"role": "user", "content": "recent-stage"} in messages
    assert messages[-1] == {"role": "user", "content": "next-stage"}


def test_stepwise_compaction_failure_preserves_exact_history_for_legacy_rebuild(monkeypatch) -> None:
    conversation = stepwise_coder.StepwiseConversation(
        base_messages=[
            {"role": "system", "content": "fixed-system"},
            {"role": "user", "content": "fixed-task-context"},
            {"role": "assistant", "content": "fixed-ack"},
        ]
    )
    conversation.record("old-stage", "old-result " * 100)
    conversation.record("recent-stage", "recent-result " * 100)

    def fail_compaction(**_):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(stepwise_coder, "generate", fail_compaction)

    messages = conversation.messages_for(
        "next-stage",
        _agent(max_tokens=1, keep_steps=1),
    )

    assert messages is None
    assert conversation.legacy_rebuild_mode is True
    assert conversation.turns
    assert conversation.turns[0]["content"] == "old-stage"
    assert conversation.compacted_history == ""


def test_compacted_exact_snapshot_can_be_retrieved_by_id(monkeypatch, tmp_path) -> None:
    conversation = stepwise_coder.StepwiseConversation(
        base_messages=[
            {"role": "system", "content": "fixed-system"},
            {"role": "user", "content": "fixed-task-context"},
            {"role": "assistant", "content": "fixed-ack"},
        ]
    )
    conversation.record("old-stage", "exact-column-name: customer_id")
    conversation.record("recent-stage", "recent-result")
    monkeypatch.setattr(stepwise_coder, "generate", lambda **_: "compressed summary")

    conversation.messages_for(
        "next-stage",
        _agent(max_tokens=1, keep_steps=1, workspace=tmp_path),
    )

    snapshot_id = conversation.snapshots[0]["snapshot_id"]
    request = f"REQUEST_CONTEXT_SNAPSHOT: {snapshot_id} | need exact schema"
    assert conversation.requested_snapshot(request) == (snapshot_id, "need exact schema")
    assert "customer_id" in conversation.retrieve_snapshot(snapshot_id)


def test_openai_message_builder_preserves_native_chat_history() -> None:
    source = [
        {"role": "system", "content": "fixed"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "result"},
        {"role": "user", "content": "next"},
    ]

    assert _prompt_to_messages(source, model="deepseek-v4-pro") == source


def test_agent_instructions_follow_complete_task_context_with_inner_headings() -> None:
    user = (
        task_section(
            "# Task title\n## Evaluation\nAuthoritative metric details.",
            "## AutoRealize Structured Context\nExact schema details.",
        )
        + "\n# Implementation\ndynamic-code"
    )

    messages = _cache_friendly_messages("stage-specific rules", user)

    assert messages is not None
    content = messages[1]["content"]
    assert content.index("Authoritative metric details") < content.index(TASK_CONTEXT_END_MARKER)
    assert content.index("Exact schema details") < content.index(TASK_CONTEXT_END_MARKER)
    assert content.index(TASK_CONTEXT_END_MARKER) < content.index(AGENT_INSTRUCTIONS_TITLE)
    assert content.index(AGENT_INSTRUCTIONS_TITLE) < content.index("dynamic-code")
