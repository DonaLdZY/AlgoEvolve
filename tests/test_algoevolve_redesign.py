from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.coder import stepwise_coder
from agents import code_review_agent, result_parse_agent
from agents.prompt_policy import (
    autonomous_method_selection_guidance,
    dynamic_expansion_instruction,
    normalize_output_language,
    output_language_instruction,
)
from engine.conditions import is_branch_stagnant, is_globally_stagnant
from engine.agent_search import AgentSearch
from engine.expansion_profile import ExpansionProfile, complexity_for_sibling
from engine.search_node import Journal, SearchNode
from engine.search_runtime_state import ActiveSearchAction
from engine.solution_manager import persist_resumable_checkpoint, write_solution_manifest
from engine.solution_protocol import interface_for, preflight_code
from llm.openai import _cache_friendly_messages
from utils.metric import MetricValue


def _prompt_agent(language: str = "english"):
    root = SearchNode(code="", plan="root", stage="root", step=0)
    journal = Journal([root])
    return SimpleNamespace(
        task_desc="Minimize assignment cost under capacity constraints.",
        coldstart_description="",
        data_preview="",
        autorealize_context="",
        acfg=SimpleNamespace(output_language=language),
        journal=journal,
        virtual_root=root,
    )


def test_interrupt_checkpoint_indexes_existing_topk_without_recopying_artifacts(
    tmp_path: Path,
) -> None:
    root = SearchNode(code="", stage="root", step=0, id="root")
    candidate = SearchNode(
        code="print('candidate')",
        stage="draft",
        step=1,
        parent=root,
        id="candidate",
        metric=MetricValue(1.0, maximize=True),
        is_buggy=False,
        is_valid=True,
    )
    candidate.search_eligible = True
    log_dir = tmp_path / "logs"
    workspace_dir = tmp_path / "workspace"
    log_dir.mkdir()
    workspace_dir.mkdir()
    agent = SimpleNamespace(
        journal=Journal([root, candidate]),
        top_candidates=[],
        top_k=3,
        best_node=None,
        metric_maximize=True,
        cfg=SimpleNamespace(
            log_dir=log_dir,
            workspace_dir=workspace_dir,
            runtime=SimpleNamespace(search_state_filename="search_state.json"),
        ),
    )

    payload = persist_resumable_checkpoint(
        agent,
        status="interrupted_resumable",
        reason="external_interrupt",
        materialize_artifacts=False,
    )

    assert payload["artifacts_materialized"] is False
    assert payload["top_solutions"][0]["node_id"] == "candidate"
    assert payload["top_solutions"][0]["materialized"] is False
    assert payload["provisional_top"][0]["source"] == "journal"
    assert not (workspace_dir / "top_solution").exists()
    assert not (workspace_dir / "checkpoint_candidates").exists()
    assert (log_dir / "checkpoint_manifest.json").is_file()
    assert (workspace_dir / "checkpoint_manifest.json").is_file()


@pytest.mark.parametrize(
    ("ordinal", "expected"),
    [(1, "simple"), (2, "simple"), (3, "normal"), (4, "normal"), (5, "complex"), (99, "complex")],
)
def test_sibling_complexity_thresholds(ordinal: int, expected: str) -> None:
    assert complexity_for_sibling(ordinal) == expected


def test_complexity_prompt_does_not_assign_a_method_family() -> None:
    guidance = " ".join(autonomous_method_selection_guidance()).lower()
    assert "whitelist or mandate" in guidance
    assert "freely choose a better unmentioned method" in guidance

    agent = _prompt_agent("english")
    profile = ExpansionProfile.create(5, operator="draft", task_family="decision")
    instruction = dynamic_expansion_instruction(agent, profile, operator="draft")
    assert "complexity profile: complex" in instruction.lower()
    assert "select it from the actual problem structure" in instruction.lower()
    assert "hybrid rl" not in instruction.lower()


def test_sibling_ordinals_are_atomic_and_action_profile_round_trips() -> None:
    parent = SearchNode(code="", plan="root", stage="root")
    with ThreadPoolExecutor(max_workers=8) as pool:
        ordinals = list(pool.map(lambda _: parent.reserve_sibling_ordinal(), range(20)))
    assert sorted(ordinals) == list(range(1, 21))

    profile = ExpansionProfile.create(5, operator="draft", task_family="decision")
    action = ActiveSearchAction("a", parent.id, expansion_profile=profile)
    restored = ActiveSearchAction.from_payload(action.to_payload())
    assert restored is not None
    assert restored.expansion_profile == profile


def test_language_instruction_is_emphatic_in_selected_language() -> None:
    assert normalize_output_language("中文") == "chinese"
    assert "必须使用中文" in output_language_instruction("chinese")
    assert "Write every user-facing" in output_language_instruction("english")
    with pytest.raises(ValueError):
        normalize_output_language("fr")

    agent = _prompt_agent("chinese")
    profile = ExpansionProfile.create(5, operator="draft", task_family="decision")
    instruction = dynamic_expansion_instruction(agent, profile, operator="draft")
    assert "Complexity profile: complex" in instruction
    assert "actual problem structure" in instruction
    assert "必须使用中文" in instruction


def test_finite_solution_interfaces_and_dangerous_code_preflight() -> None:
    prediction = interface_for(task_family="prediction", method_family="tree_boosting")
    assert preflight_code(
        "def train(data, artifact_dir): return artifact_dir\n"
        "def predict(model_path, data): return data\n",
        prediction,
    ).ok
    missing = preflight_code("def predict(model_path, data): return data\n", prediction)
    assert missing.missing_functions == ("train",)
    wrong_signature = preflight_code(
        "def train(dataset, output_dir): return output_dir\n"
        "def predict(data, model_path): return data\n",
        prediction,
    )
    assert wrong_signature.ok is False
    assert len(wrong_signature.invalid_signatures) == 2

    decision = interface_for(task_family="decision", method_family="heuristic")
    decision_report = preflight_code("def solve(model_path, data): return data\n", decision)
    assert decision_report.ok
    assert decision_report.expected_signatures == ("def solve(model_path, data): ...",)
    rl = interface_for(task_family="decision", method_family="reinforcement_learning")
    assert preflight_code(
        "def train_policy(data, artifact_dir): return artifact_dir\n"
        "def rollout(model_path, data): return data\n",
        rl,
    ).ok
    dangerous = preflight_code(
        "def solve(model_path, data):\n    return eval(data)\n",
        decision,
    )
    assert dangerous.ok is False
    assert dangerous.dangerous_findings


def test_preflight_rejection_regenerates_same_draft_instead_of_dropping_it(
    monkeypatch,
) -> None:
    root = SearchNode(code="", plan="root", stage="root", step=0)
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.task_desc = "Minimize assignment cost under capacity constraints."
    agent.data_preview = ""
    agent.autorealize_context = ""
    agent.coldstart_description = ""
    agent.runtime_checkpoint_callback = None
    agent.scfg = SimpleNamespace(num_drafts=8)
    agent.acfg = SimpleNamespace(
        draft=SimpleNamespace(fast_first_draft_skip_pre_review=False),
        retries=SimpleNamespace(preflight_regeneration_max_attempts=2),
    )

    rejected = SearchNode(
        code="def solve(data, artifact_dir): return data\n",
        plan="simulated annealing",
        stage="draft",
        parent=root,
    )
    monkeypatch.setattr(code_review_agent, "run", lambda _agent, node: node.code)
    monkeypatch.setattr(
        code_review_agent,
        "regenerate_after_preflight_failure",
        lambda *_args, **_kwargs: (
            "same simulated annealing method with corrected interface",
            "def solve(model_path, data): return data\n",
        ),
    )
    from agents import draft_agent

    monkeypatch.setattr(draft_agent, "run", lambda *_args, **_kwargs: rejected)

    _returned_to_root, result = agent._run_single_step(
        root,
        exec_callback=lambda *_args, **_kwargs: None,
        execute_immediately=False,
        expansion_profile=ExpansionProfile.create(
            1,
            operator="draft",
            task_family="decision",
        ),
    )

    assert result is rejected
    assert result.pending_execution is True
    assert result.preflight_report["ok"] is True
    assert result.plan.startswith("same simulated annealing")
    assert "def solve(model_path, data)" in result.code


def test_preflight_regeneration_prompt_ends_with_exact_rejection_evidence(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    agent = SimpleNamespace(
        task_desc="Minimize assignment cost.",
        data_preview="stable data contract",
        autorealize_context="",
        acfg=SimpleNamespace(
            output_language="english",
            code=SimpleNamespace(model="deepseek-chat", temp=0.0),
            retries=SimpleNamespace(code_generation_extract_max_attempts=1),
        ),
        cfg=SimpleNamespace(),
    )
    node = SearchNode(
        code="def solve(data, artifact_dir): return data\n",
        plan="simulated annealing",
        stage="draft",
    )
    interface = interface_for(task_family="decision", method_family="heuristic")
    report = preflight_code(node.code, interface)

    def fake_generate(_agent, prompt):
        captured["prompt"] = prompt
        return "corrected", "def solve(model_path, data): return data\n"

    monkeypatch.setattr(code_review_agent, "plan_and_code_query", fake_generate)
    plan, code = code_review_agent.regenerate_after_preflight_failure(
        agent,
        node,
        interface,
        report,
        attempt=1,
    )

    prompt = captured["prompt"]
    assert isinstance(prompt, dict)
    assert "# End stable task/data context" in str(prompt["user"])
    assert str(prompt["user"]).rfind("Latest deterministic rejection evidence") > str(
        prompt["user"]
    ).rfind("Current rejected candidate")
    assert "def solve(model_path, data): ..." in str(prompt["user"])
    assert plan == "corrected"
    assert "solve(model_path, data)" in code


def test_meta_merge_failure_never_concatenates_partial_code(monkeypatch) -> None:
    monkeypatch.setattr(
        stepwise_coder.MetaAgent,
        "_build_merge_prompt",
        lambda *args, **kwargs: "merge prompt",
    )
    monkeypatch.setattr(stepwise_coder, "generate", lambda **kwargs: "no code block")
    agent = SimpleNamespace(
        acfg=SimpleNamespace(code=SimpleNamespace(temp=0.0)),
        cfg=SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="refusing to execute concatenated"):
        stepwise_coder.MetaAgent().merge(
            task_desc="task",
            data_preview_str="data",
            step_results=[{"name": "one", "plan": "p", "code": "print('partial')"}],
            prompt_base={"Instructions": {}},
            agent_instance=agent,
            context=stepwise_coder.StepwiseContext(),
        )


def test_global_stagnation_compares_against_pre_window_best() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    values = [10.0, 10.1, 10.1, 10.1]
    nodes = [
        SearchNode(
            code="",
            plan=str(value),
            stage="draft" if index == 0 else "improve",
            parent=root if index == 0 else None,
            metric=MetricValue(value, maximize=True),
            is_buggy=False,
            is_valid=True,
            search_eligible=True,
        )
        for index, value in enumerate(values)
    ]
    agent = SimpleNamespace(
        journal=Journal([root, *nodes]),
        stagnation_threshold=2,
        metric_maximize=True,
        scfg=SimpleNamespace(metric_improvement_threshold=0.05),
    )
    assert is_globally_stagnant(agent) is True
    nodes[-1].metric = MetricValue(10.3, maximize=True)
    assert is_globally_stagnant(agent) is False


def test_branch_stagnation_treats_repeated_equal_scores_as_plateau() -> None:
    nodes = [
        SearchNode(
            code="",
            plan=str(value),
            stage="draft",
            metric=MetricValue(value, maximize=True),
            is_buggy=False,
            is_valid=True,
            search_eligible=True,
        )
        for value in [1.0, 1.2, 1.2, 1.2, 1.2]
    ]
    agent = SimpleNamespace(
        branch_successful_nodes={1: nodes},
        metric_maximize=True,
        scfg=SimpleNamespace(metric_improvement_threshold=0.01),
    )
    assert is_branch_stagnant(agent, 1, threshold=3) is True


def test_solution_manifest_records_cross_system_entrypoint(tmp_path) -> None:
    agent = _prompt_agent()
    agent.cfg = SimpleNamespace(workspace_dir=tmp_path)
    node = SearchNode(
        code="def solve(model_path, data): return data\n",
        plan="greedy assignment",
        stage="draft",
        method_family="heuristic",
    )
    payload = write_solution_manifest(agent, node, tmp_path)
    saved = json.loads((tmp_path / "solution_manifest.json").read_text(encoding="utf-8"))
    assert payload["interface_version"] == "algoevolve.solution.v1"
    assert saved["entrypoint"] == "solve"
    assert saved["stateful"] is False


def test_provider_cache_prefix_stays_identical_before_dynamic_agent_rules() -> None:
    task = "# Task description\nlarge stable task\n\n# End stable task/data context\n"
    first = _cache_friendly_messages("draft rules", task + "\n# Dynamic\none")
    second = _cache_friendly_messages("review rules", task + "\n# Dynamic\ntwo")
    assert first is not None and second is not None
    assert first[0] == second[0]
    marker = "# End stable task/data context"
    assert first[1]["content"].split(marker, 1)[0] == second[1]["content"].split(marker, 1)[0]


def test_only_suspicious_or_uncertain_results_trigger_adjudicator() -> None:
    agent = SimpleNamespace(
        task_desc="Minimize an assignment problem cost under capacity constraints.",
        coldstart_description="",
        data_preview="",
        metric_maximize=False,
        acfg=SimpleNamespace(
            retries=SimpleNamespace(result_adjudicator_on_anomaly=True),
        ),
    )
    accepted = {"verdict": "accept", "is_bug": False, "confidence": 0.9}
    assert result_parse_agent._needs_result_adjudication(agent, accepted, 0.0) is True
    assert result_parse_agent._needs_result_adjudication(agent, accepted, 12.0) is False
    rejected = {"verdict": "reject", "is_bug": True, "confidence": 0.9}
    assert result_parse_agent._needs_result_adjudication(agent, rejected, 0.0) is False
    uncertain = {"verdict": "uncertain", "is_bug": False, "confidence": 0.5}
    assert result_parse_agent._needs_result_adjudication(agent, uncertain, 12.0) is True
