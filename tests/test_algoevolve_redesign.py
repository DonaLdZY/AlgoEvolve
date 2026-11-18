from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.coder import stepwise_coder
from agents import code_review_agent, result_parse_agent, draft_agent
from agents.prompts.shared import get_internet_clarification
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


def test_draft_metric_guidance_requires_contract_repair_instead_of_invention() -> None:
    guidance = draft_agent.EVALUATION_CONTRACT_GUIDANCE.lower()

    assert "do not score candidates" in guidance
    assert "exact missing formula or parameter" in guidance
    assert "most conservative supported scalar" not in guidance


def test_network_guidance_preserves_task_restrictions_and_does_not_assume_assets() -> None:
    guidance = "\n".join(get_internet_clarification("C:/models")).lower()
    assert "follow the active task" in guidance
    assert "do not reinterpret a network restriction" in guidance
    assert "not proof that the files exist or load" in guidance
    assert "can and should use online resources" not in guidance
    assert "guaranteed to exist" not in guidance


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


@pytest.mark.parametrize("call", [
    "train()",
    "train('input', 'artifact', 'unexpected')",
    "train(dataset='input', artifact_dir='artifact')",
    "predict('model.pkl')",
])
def test_preflight_catches_invalid_cli_interface_calls_before_execution(call):
    code = (
        "def train(data, artifact_dir='artifact'): return data\n"
        "def predict(model_path, data): return data\n"
        f"if __name__ == '__main__':\n    {call}\n"
    )
    report = preflight_code(code, interface_for(task_family="prediction", method_family="tree"))
    assert not report.ok
    assert len(report.invalid_calls) == 1
    assert "module call" in report.invalid_calls[0]


@pytest.mark.parametrize("suffix", [
    "train('input')",
    "train(data='input', artifact_dir='artifact')",
    "train(*runtime_args, **runtime_kwargs)",
    "def helper(train):\n    return train()\n",
    "class Runner:\n    def train(self): return 1\n    def run(self): return self.train()\n",
    "train = factory()\ntrain()",
])
def test_preflight_preserves_defaulted_dynamic_and_shadowed_calls(suffix):
    code = (
        "def train(data, artifact_dir='artifact'): return data\n"
        "def predict(model_path, data): return data\n" + suffix + "\n"
    )
    assert preflight_code(
        code, interface_for(task_family="prediction", method_family="tree"),
    ).ok


def test_preflight_rejects_ast_parseable_but_uncompilable_candidate():
    report = preflight_code(
        "def train(data, artifact_dir): return data\n"
        "def predict(model_path, data): return data\n"
        "return None\n",
        interface_for(task_family="prediction", method_family="tree"),
    )
    assert not report.ok and report.syntax_error


@pytest.mark.parametrize("formal_task", [False, True])
@pytest.mark.parametrize("initially_valid", [False, True])
def test_preflight_rejection_regenerates_same_draft_instead_of_dropping_it(
    monkeypatch,
    formal_task,
    initially_valid,
) -> None:
    root = SearchNode(code="", plan="root", stage="root", step=0)
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.task_desc = "Minimize assignment cost under capacity constraints."
    agent.data_preview = ""
    agent.autorealize_context = ""
    agent.has_autorealize_context = formal_task
    agent.coldstart_description = ""
    agent.runtime_checkpoint_callback = None
    agent.scfg = SimpleNamespace(num_drafts=8)
    agent.cfg = SimpleNamespace(runtime=SimpleNamespace(resume_run=False))
    agent.acfg = SimpleNamespace(
        draft=SimpleNamespace(fast_first_draft_skip_pre_review=True),
        retries=SimpleNamespace(preflight_regeneration_max_attempts=2),
    )

    rejected = SearchNode(
        code=("def solve(model_path, data): return data\ndef final_evaluate(data, artifact_dir): return {}\n"
              if initially_valid else "def solve(data, artifact_dir): return data\n"),
        plan="simulated annealing",
        stage="draft",
        parent=root,
    )
    reviews = []
    def review(_agent, node):
        reviews.append(node.id)
        if formal_task and initially_valid:
            return "def solve(data, model_path): return data\n"
        return node.code
    monkeypatch.setattr(code_review_agent, "run", review)
    monkeypatch.setattr(
        code_review_agent,
        "regenerate_after_preflight_failure",
        lambda *_args, **_kwargs: (
            "same simulated annealing method with corrected interface",
            "def solve(model_path, data): return data\ndef final_evaluate(data, artifact_dir): return {'status': 'not_applicable', 'reason': 'No empirical holdout'}\n",
        ),
    )
    from agents import draft_agent

    monkeypatch.setattr(draft_agent, "run", lambda *_args, **_kwargs: rejected)

    _returned_to_root, result = agent._run_single_step(
        root,
        exec_callback=lambda *_args, **_kwargs: None,
        execute_immediately=False,
        fast_draft_mode=True,
        expansion_profile=ExpansionProfile.create(
            1,
            operator="draft",
            task_family="decision",
        ),
    )

    assert result is rejected
    assert result.pending_execution is True
    assert result.preflight_report["ok"] is True
    assert result.plan.startswith("simulated annealing" if initially_valid else "same simulated annealing")
    assert "def solve(model_path, data)" in result.code
    assert bool(reviews) is (formal_task or not initially_valid)
    if formal_task and initially_valid:
        assert result.preexecution_review_warnings == [
            "Reviewer edit did not improve deterministic preflight; original code was retained."
        ]


def test_preflight_regeneration_stops_when_failure_signature_does_not_change(monkeypatch):
    root = SearchNode(code="", plan="root", stage="root", step=0)
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.task_desc = "Minimize assignment cost."
    agent.data_preview = ""
    agent.autorealize_context = ""
    agent.has_autorealize_context = False
    agent.coldstart_description = ""
    agent.runtime_checkpoint_callback = None
    agent.scfg = SimpleNamespace(num_drafts=2)
    agent.cfg = SimpleNamespace(runtime=SimpleNamespace(resume_run=False))
    agent.acfg = SimpleNamespace(
        draft=SimpleNamespace(fast_first_draft_skip_pre_review=False),
        retries=SimpleNamespace(preflight_regeneration_max_attempts=2),
    )
    rejected = SearchNode(
        code="def solve(data, model_path): return data\n",
        plan="simulated annealing",
        stage="draft",
        parent=root,
    )
    reviews, regenerations = [], []
    monkeypatch.setattr(code_review_agent, "run", lambda _agent, node: reviews.append(node.id) or node.code)
    def regenerate(_agent, node, _interface, _preflight, *, attempt):
        regenerations.append(attempt)
        return node.plan, node.code
    monkeypatch.setattr(code_review_agent, "regenerate_after_preflight_failure", regenerate)
    from agents import draft_agent
    monkeypatch.setattr(draft_agent, "run", lambda *_args, **_kwargs: rejected)

    with pytest.raises(RuntimeError, match="failed mandatory pre-execution"):
        agent._run_single_step(
            root,
            exec_callback=lambda *_args, **_kwargs: None,
            execute_immediately=False,
            fast_draft_mode=False,
            expansion_profile=ExpansionProfile.create(1, operator="draft", task_family="decision"),
        )

    assert reviews == [rejected.id]
    assert regenerations == [1]


def test_code_review_prompt_requires_oof_calibration_and_temporal_purge():
    from agents.prompts.validation_template_prompts import get_code_review_guidelines

    prompt = "\n".join(get_code_review_guidelines(submission_required=False))
    assert "same development rows" in prompt
    assert "earliest validation cutoff" in prompt


def test_exhausted_review_can_continue_only_when_deterministic_preflight_passes(monkeypatch):
    root = SearchNode(code="", plan="root", stage="root", step=0)
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.task_desc = "Minimize assignment cost."
    agent.data_preview = ""
    agent.autorealize_context = ""
    agent.has_autorealize_context = False
    agent.coldstart_description = ""
    agent.runtime_checkpoint_callback = None
    agent.scfg = SimpleNamespace(num_drafts=2)
    agent.cfg = SimpleNamespace(runtime=SimpleNamespace(resume_run=False))
    agent.acfg = SimpleNamespace(
        draft=SimpleNamespace(fast_first_draft_skip_pre_review=False),
        retries=SimpleNamespace(preflight_regeneration_max_attempts=0, review_gate_policy="continue_on_exhaustion"),
    )
    candidate = SearchNode(
        code=(
            "def train(data, artifact_dir): return artifact_dir\n"
            "def predict(model_path, data): return data\n"
            "def solve(model_path, data): return data\n"
        ),
        plan="heuristic", stage="draft", parent=root,
    )
    monkeypatch.setattr(draft_agent, "run", lambda *_args, **_kwargs: candidate)
    monkeypatch.setattr(code_review_agent, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(code_review_agent.CodeReviewFailed("provider exhausted")))
    _root, result = agent._run_single_step(
        root, exec_callback=lambda *_args, **_kwargs: None, execute_immediately=False,
        fast_draft_mode=False, expansion_profile=ExpansionProfile.create(1, operator="draft", task_family="decision"),
    )
    assert result is candidate
    assert result.pending_execution is True
    assert result.preexecution_review_warnings == ["Pre-execution LLM review exhausted without approval: provider exhausted"]


def test_exhausted_review_does_not_bypass_deterministic_preflight(monkeypatch):
    root = SearchNode(code="", plan="root", stage="root", step=0)
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.task_desc = "Minimize assignment cost."
    agent.data_preview = ""
    agent.autorealize_context = ""
    agent.has_autorealize_context = False
    agent.coldstart_description = ""
    agent.runtime_checkpoint_callback = None
    agent.scfg = SimpleNamespace(num_drafts=2)
    agent.cfg = SimpleNamespace(runtime=SimpleNamespace(resume_run=False))
    agent.acfg = SimpleNamespace(
        draft=SimpleNamespace(fast_first_draft_skip_pre_review=False),
        retries=SimpleNamespace(preflight_regeneration_max_attempts=0, review_gate_policy="continue_on_exhaustion"),
    )
    candidate = SearchNode(code="def solve(data, artifact_dir): return data\n", plan="heuristic", stage="draft", parent=root)
    monkeypatch.setattr(draft_agent, "run", lambda *_args, **_kwargs: candidate)
    monkeypatch.setattr(code_review_agent, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(code_review_agent.CodeReviewFailed("provider exhausted")))
    with pytest.raises(code_review_agent.CodeReviewFailed):
        agent._run_single_step(
            root, exec_callback=lambda *_args, **_kwargs: None, execute_immediately=False,
            fast_draft_mode=False, expansion_profile=ExpansionProfile.create(1, operator="draft", task_family="decision"),
        )


@pytest.mark.parametrize("suffix", [
    "", "def final_evaluate(recipe): return recipe\n",
    "def final_evaluate(data, artifact_dir, holdout_positions): return {}\n",
    "def final_evaluate(data, artifact_dir, *, recipe): return {}\n",
    "async def final_evaluate(data, artifact_dir): return {}\n",
    "def final_evaluate(data, artifact_dir): return {}\nfinal_evaluate(None, None)\n",
    "def final_evaluate(data, artifact_dir): return {}\ndef main():\n    callback = final_evaluate\n",
])
def test_formal_preflight_blocks_missing_wrong_or_candidate_called_final(suffix):
    code = "def train(data, artifact_dir): pass\ndef predict(model_path, data): pass\n" + suffix
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    assert not preflight_code(code, interface, require_final_evaluation=True).ok


def test_final_preflight_allows_engine_only_entrypoint():
    code = ("def train(data, artifact_dir): pass\ndef predict(model_path, data): pass\n"
            "def final_evaluate(data, artifact_dir): return {'status': 'not_applicable', 'reason': 'No holdout'}\n"
            "if __name__ == '__main__': train(None, None)\n")
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    assert preflight_code(code, interface, require_final_evaluation=True).ok


@pytest.mark.parametrize(
    "body",
    [
        "if __name__ == '__main__':\n    evaluate_holdout(data, partition='holdout')\n",
        "def train(data, artifact_dir):\n    return evaluate(data, split='test')\n",
        "def predict(model_path, data):\n    return evaluate_holdout(data, _allow_holdout=True)\n",
    ],
)
def test_preflight_blocks_search_time_holdout_access(body):
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    code = (
        "def train(data, artifact_dir): return artifact_dir\n"
        "def predict(model_path, data): return data\n"
        + body
    )
    report = preflight_code(code, interface)
    assert report.ok is False
    assert report.policy_findings


def test_preflight_allows_holdout_only_inside_engine_final_evaluate():
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    code = (
        "def train(data, artifact_dir): return artifact_dir\n"
        "def predict(model_path, data): return data\n"
        "def final_evaluate(data, artifact_dir):\n"
        "    return evaluate_holdout(data, partition='holdout', _allow_holdout=True)\n"
    )
    report = preflight_code(code, interface, require_final_evaluation=True)
    assert report.ok is True
    assert not report.policy_findings


def test_preflight_requires_completed_final_result_contract_keys():
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    code = (
        "def train(data, artifact_dir): return artifact_dir\n"
        "def predict(model_path, data): return data\n"
        "def final_evaluate(data, artifact_dir):\n"
        "    return {'status': 'completed', 'holdout_mae': 0.4}\n"
    )

    report = preflight_code(code, interface, require_final_evaluation=True)

    assert report.ok is False
    finding = next(item for item in report.policy_findings if "completed return must include fields" in item)
    assert all(name in finding for name in ("metrics", "model_path", "inference_checks"))


def test_preflight_does_not_accept_contract_names_mentioned_outside_completed_return():
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    code = (
        "def train(data, artifact_dir): return artifact_dir\n"
        "def predict(model_path, data): return data\n"
        "def final_evaluate(data, artifact_dir):\n"
        "    required = ('metrics', 'model_path', 'inference_checks')\n"
        "    return {'status': 'completed', 'holdout_mae': 0.4}\n"
    )

    report = preflight_code(code, interface, require_final_evaluation=True)

    assert report.ok is False
    finding = next(item for item in report.policy_findings if "completed return must include fields" in item)
    assert all(name in finding for name in ("metrics", "model_path", "inference_checks"))


def test_preflight_accepts_complete_final_result_contract_keys():
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    code = (
        "def train(data, artifact_dir): return artifact_dir\n"
        "def predict(model_path, data): return data\n"
        "def final_evaluate(data, artifact_dir):\n"
        "    return {'status': 'completed', 'metrics': {'mae': 0.4}, "
        "'model_path': 'model.pkl', 'inference_checks': {}}\n"
    )

    report = preflight_code(code, interface, require_final_evaluation=True)

    assert report.ok is True


def test_preflight_ignores_completed_helper_results_when_checking_final_contract():
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    code = (
        "def train(data, artifact_dir): return artifact_dir\n"
        "def predict(model_path, data): return data\n"
        "def evaluate_holdout(data, artifact_dir, split):\n"
        "    return {'status': 'completed', 'split': split, 'mae': 0.4, 'rows': []}\n"
        "def final_evaluate(data, artifact_dir):\n"
        "    holdout = evaluate_holdout(data, artifact_dir, 'holdout')\n"
        "    report = {'status': 'completed', 'holdout_mae': holdout['mae'], "
        "'model_path': 'model.pkl', 'inference_checks': []}\n"
        "    return report\n"
    )

    report = preflight_code(code, interface, require_final_evaluation=True)

    assert report.ok is False
    finding = next(item for item in report.policy_findings if "completed return must include fields" in item)
    assert finding.endswith("metrics")


def test_preflight_accepts_final_contract_with_internal_evaluation_helpers():
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    code = (
        "def train(data, artifact_dir): return artifact_dir\n"
        "def predict(model_path, data): return data\n"
        "def evaluate_holdout(data, artifact_dir, split):\n"
        "    return {'status': 'completed', 'split': split, 'mae': 0.4, 'rows': []}\n"
        "def final_evaluate(data, artifact_dir):\n"
        "    holdout = evaluate_holdout(data, artifact_dir, 'holdout')\n"
        "    return {'status': 'completed', 'metrics': {'mae': holdout['mae']}, "
        "'model_path': 'model.pkl', 'inference_checks': []}\n"
    )

    report = preflight_code(code, interface, require_final_evaluation=True)

    assert report.ok is True


def test_preflight_defers_mutated_final_result_to_runtime_contract_validation():
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    code = (
        "def train(data, artifact_dir): return artifact_dir\n"
        "def predict(model_path, data): return data\n"
        "def final_evaluate(data, artifact_dir):\n"
        "    report = {'status': 'completed', 'model_path': 'model.pkl', "
        "'inference_checks': []}\n"
        "    report['metrics'] = {'mae': 0.4}\n"
        "    return report\n"
    )

    report = preflight_code(code, interface, require_final_evaluation=True)

    assert report.ok is True


def test_preflight_allows_top_level_helper_reachable_only_from_final_evaluate():
    interface = interface_for(task_family="prediction", method_family="tree_boosting")
    code = (
        "def train(data, artifact_dir): return artifact_dir\n"
        "def predict(model_path, data): return data\n"
        "def final_evaluate(data, artifact_dir): return score_final(data)\n"
        "def score_final(data):\n"
        "    return evaluate_holdout(data, partition='holdout', _allow_holdout=True)\n"
    )

    report = preflight_code(code, interface, require_final_evaluation=True)

    assert report.ok is True
    assert not report.policy_findings


@pytest.mark.parametrize(
    "code",
    [
        (
            "def train(data, artifact_dir): return score_final(data)\n"
            "def predict(model_path, data): return data\n"
            "def final_evaluate(data, artifact_dir): return score_final(data)\n"
            "def score_final(data): return evaluate_holdout(data, partition='holdout')\n"
        ),
        (
            "def train(data, artifact_dir): return score_final(data)\n"
            "def predict(model_path, data): return data\n"
            "def score_final(data): return evaluate_holdout(data, partition='holdout')\n"
            "if __name__ == '__main__': train(None, None)\n"
        ),
    ],
)
def test_preflight_rejects_holdout_helpers_reachable_from_search(code):
    interface = interface_for(task_family="prediction", method_family="tree_boosting")

    report = preflight_code(code, interface, require_final_evaluation=True)

    assert report.ok is False
    assert report.policy_findings


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

    def fake_generate(_agent, prompt, *, stage_id):
        assert stage_id == "preflight_code_regeneration"
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
