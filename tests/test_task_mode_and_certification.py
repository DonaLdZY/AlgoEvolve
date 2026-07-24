from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from agents.coder.stepwise_coder import create_default_step_agents
from agents.planner.base_planner import get_component_descriptions, get_planning_allowed_modules
from agents.prompts import infer_task_mode
from engine.execution import (
    _append_nonfatal_decision_warning,
    record_independent_score,
    update_node_certification,
)
from engine.search_node import SearchNode
from engine.solution_manager import update_best_solution
from utils.metric import MetricValue


def _context(*, structure: str, rl_required: bool, first_method: str) -> str:
    return f"""## AutoRealize Structured Context

## Method Strategy
- problem_paradigm: `static_optimization`
- problem_structure: `{structure}`
- rl_required: `{str(rl_required).lower()}`
- first_draft_method: `{first_method}`

## Evaluation Contract Reference
- metric_direction: minimize

## Output Contract Reference
- output_filename: submission.csv
"""


def _agent(tmp_path: Path, context: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_desc="Produce a decision artifact.",
        data_preview=context,
        coldstart_description="",
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        acfg=SimpleNamespace(generate_submission=False),
        branch_successful_nodes={},
        top_candidates=[],
        top_k=3,
        metric_maximize=False,
        best_node=None,
        provisional_best_node=None,
        save_node_lock=threading.Lock(),
    )


def test_static_optimization_with_required_rl_routes_to_rl() -> None:
    context = _context(
        structure="decision_optimization",
        rl_required=True,
        first_method="reinforcement_learning",
    )
    assert infer_task_mode("", "", context) == "rl"
    assert [step.name for step in create_default_step_agents("rl")] == [
        "problem_and_evaluator",
        "decision_method",
        "solve_rollout_and_artifact",
    ]


def test_optional_rl_does_not_force_optimization_route_into_rl() -> None:
    context = _context(
        structure="decision_optimization",
        rl_required=False,
        first_method="optimization",
    )
    assert infer_task_mode("", "", context) == "optimization"
    steps = create_default_step_agents("optimization")
    names = [step.name for step in steps]
    assert names == [
        "problem_and_evaluator",
        "decision_method",
        "solve_rollout_and_artifact",
    ]
    prompt_text = "\n".join(step.description + "\n" + "\n".join(step.guidelines) for step in steps)
    assert "Dropout" not in prompt_text
    assert "train/validation/test splits" not in prompt_text


def test_native_sequential_control_routes_to_rl() -> None:
    context = _context(
        structure="native_sequential_control",
        rl_required=False,
        first_method="task_appropriate",
    )
    assert infer_task_mode("", "", context) == "rl"


def test_diff_planner_modules_match_each_stepwise_route(tmp_path: Path) -> None:
    descriptions = get_component_descriptions()
    cases = {
        "prediction": SimpleNamespace(
            task_desc="Predict next month's sales.",
            coldstart_description="",
            data_preview="",
        ),
        "optimization": _agent(
            tmp_path,
            _context(
                structure="decision_optimization",
                rl_required=False,
                first_method="optimization",
            ),
        ),
        "rl": _agent(
            tmp_path,
            _context(
                structure="native_sequential_control",
                rl_required=False,
                first_method="task_appropriate",
            ),
        ),
    }

    for task_mode, agent in cases.items():
        expected = [step.name for step in create_default_step_agents(task_mode)]
        actual = get_planning_allowed_modules(agent_instance=agent)
        assert actual == expected
        assert all(name in descriptions for name in actual)


def test_stateless_optimization_zero_score_can_be_delivery_ready(tmp_path: Path) -> None:
    context = _context(
        structure="decision_optimization",
        rl_required=False,
        first_method="optimization",
    )
    agent = _agent(tmp_path, context)
    node = SearchNode(
        code="def predict(model_path, data):\n    return data\n",
        stage="draft",
        _term_out=["Final Validation Score: 0.0\n"],
        metric=MetricValue(0.0, maximize=False),
        is_buggy=False,
        is_valid=True,
    )

    update_node_certification(agent, node)

    assert node.search_eligible is True
    assert node.method_mode == "non_rl_solver"
    assert node.artifact_ready is True
    assert node.delivery_ready is True
    assert node.delivery_certified is False

    assert record_independent_score(agent, node, 0.0) is True
    assert node.delivery_certified is True
    assert node.certification_source == "trusted_evaluator"


def test_nonfatal_delivery_warning_preserves_reviewer_validity(tmp_path: Path) -> None:
    context = _context(
        structure="decision_optimization",
        rl_required=False,
        first_method="optimization",
    )
    agent = _agent(tmp_path, context)
    node = SearchNode(
        code="def predict(model_path, data):\n    return data\n",
        stage="debug",
        _term_out=["Final Validation Score: 619406.0\n"],
        metric=MetricValue(619406.0, maximize=False),
        is_buggy=False,
        is_valid=True,
    )

    _append_nonfatal_decision_warning(node, "No node-specific submission file was found.")
    update_node_certification(agent, node)

    assert node.is_valid is True
    assert node.contract_valid is True
    assert node.delivery_ready is True
    assert "No node-specific submission file was found." in node.analysis


def test_rl_without_policy_artifact_is_review_evidence_not_a_third_gate(tmp_path: Path) -> None:
    context = _context(
        structure="decision_optimization",
        rl_required=True,
        first_method="reinforcement_learning",
    )
    agent = _agent(tmp_path, context)
    node = SearchNode(
        code="def predict(model_path, data):\n    return data\n",
        stage="draft",
        _term_out=["Method Usage Summary: pure_rl\nFinal Validation Score: 1.0\n"],
        metric=MetricValue(1.0, maximize=False),
        is_buggy=False,
        is_valid=True,
    )

    update_node_certification(agent, node)
    update_best_solution(agent, node)

    assert node.search_eligible is True
    assert node.method_mode == "pure_rl"
    assert node.delivery_ready is True
    assert agent.provisional_best_node is node
    assert agent.best_node is node
    assert any("no persisted model/policy artifact" in note for note in node.certification_notes)


def test_invalid_search_candidate_never_replaces_delivery_best(tmp_path: Path) -> None:
    context = _context(
        structure="decision_optimization",
        rl_required=False,
        first_method="optimization",
    )
    agent = _agent(tmp_path, context)
    candidate = SearchNode(
        code="def predict(model_path, data):\n    return data\n",
        stage="draft",
        metric=MetricValue(1.0, maximize=False),
        is_buggy=False,
        is_valid=False,
        search_eligible=True,
        delivery_ready=False,
    )
    ready = SearchNode(
        code="def predict(model_path, data):\n    return data\n",
        stage="improve",
        metric=MetricValue(5.0, maximize=False),
        is_buggy=False,
        is_valid=True,
        search_eligible=True,
        delivery_ready=True,
        method_mode="non_rl_solver",
    )

    update_best_solution(agent, candidate)
    update_best_solution(agent, ready)

    assert agent.provisional_best_node is candidate
    assert agent.best_node is ready
