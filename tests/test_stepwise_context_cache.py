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


def test_stepwise_reserves_reply_and_compacts_even_two_recent_steps(monkeypatch, tmp_path):
    conversation = stepwise_coder.StepwiseConversation(base_messages=[
        {"role": "user", "content": "UNCHANGED_FULL_TASK"}])
    conversation.record("data", "full original data module")
    conversation.record("model", "full original model module")
    agent = _agent(max_tokens=0, keep_steps=2, workspace=tmp_path)
    agent.acfg.code = SimpleNamespace(context_window_tokens=131072, max_tokens=32768)
    monkeypatch.setattr(conversation, '_estimated_tokens', lambda *_: 100000)
    calls = []
    monkeypatch.setattr(stepwise_coder, 'generate', lambda **kw: calls.append(kw) or "Preserved interfaces and rules")
    messages = conversation.messages_for("inference", agent)
    assert calls and messages[0] == conversation.base_messages[0]
    assert "UNCHANGED_FULL_TASK" not in str(calls[0]['prompt'])
    snapshot = conversation.retrieve_snapshot(conversation.snapshots[0]['snapshot_id'])
    assert "full original data module" in snapshot
    assert "full original model module" in str(messages)


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


def test_merge_receives_exact_stages_after_compaction_and_repairs_syntax(monkeypatch):
    conversation = stepwise_coder.StepwiseConversation(
        base_messages=[{'role':'user','content':'UNCHANGED_TASK'}],
        compacted_history='only a summary, no full code')
    stages = [{'name':'data','plan':'p','code':'def load_data():\n    return 42'},
              {'name':'model','plan':'p','code':'def train(data):\n    return data'}]
    monkeypatch.setattr(stepwise_coder.MetaAgent, '_build_merge_prompt', lambda *a,**kw:{
        'system':'merge','user':'# Instructions\nKeep complete source\n# Step results\nignored legacy block'})
    seen=[]
    stage_ids=[]
    bad='```python\ndef broken(:\n    pass\n```'
    def generate(**kw):
        seen.append(kw['prompt'])
        stage_ids.append(kw["stage_id"])
        return bad if len(seen)==1 else '```python\ndef load_data():\n    return 42\ndef train(data):\n    return data\n```'
    monkeypatch.setattr(stepwise_coder,'generate',generate)
    agent=_agent();agent.acfg.code=SimpleNamespace(temp=0)
    _,code=stepwise_coder.MetaAgent().merge('task','data',stages,{},agent,
        stepwise_coder.StepwiseContext(),conversation=conversation)
    assert seen[0][0] == conversation.base_messages[0]
    assert all(stage['code'] in str(seen[0]) or stage['code'] in seen[0][-1]['content'] for stage in stages)
    assert 'only a summary' not in str(seen[0])
    assert seen[1][:len(seen[0])] == seen[0]
    assert seen[1][-2]['content'] == bad
    assert 'invalid syntax' in seen[1][-1]['content']
    assert stage_ids == ["stepwise_solution_merge", "stepwise_solution_merge"]
    compile(code,'merged','exec')


def test_stage_snapshot_retrieval_does_not_consume_single_code_attempt(monkeypatch):
    conversation=stepwise_coder.StepwiseConversation(base_messages=[{'role':'user','content':'fixed'}])
    monkeypatch.setattr(conversation,'retrieve_snapshot',lambda value:'EXACT_PRIOR_CODE' if value=='a'*16 else None)
    stage=stepwise_coder.StepAgent('stage','intro','description',[])
    monkeypatch.setattr(stage,'_build_prompt',lambda **kw:{'system':'stage','user':'# Instructions\nDo stage\n# Current step: stage'})
    seen=[]
    stage_ids=[]
    def generate(**kw):
        seen.append(kw['prompt'])
        stage_ids.append(kw["stage_id"])
        return 'REQUEST_CONTEXT_SNAPSHOT: '+('a'*16)+' | exact schema' if len(seen)==1 else '```python\nvalue=42\n```'
    monkeypatch.setattr(stepwise_coder,'generate',generate)
    agent=_agent();agent.acfg.code=SimpleNamespace(temp=0)
    _,code=stage.generate('task','data',[],{},agent,stepwise_coder.StepwiseContext(),retries=1,conversation=conversation)
    assert len(seen)==2 and 'EXACT_PRIOR_CODE' in str(seen[1])
    assert stage_ids == ["stepwise_candidate_generation", "stepwise_candidate_generation"]
    namespace = {}
    exec(code, namespace)
    assert namespace['value'] == 42
