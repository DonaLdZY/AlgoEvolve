from __future__ import annotations

import json
from types import SimpleNamespace

from agents.aggregation_agent import _collect_branch_representatives
from agents.prompts.impl_guideline import get_impl_guideline, get_impl_guideline_from_agent
from agents.prompts.validation_template_prompts import get_code_review_guidelines
from engine.executor import ExecutionResult
from engine.search_node import Journal, SearchNode
from engine.solution_manager import get_branch_top_nodes
from engine.solution_protocol import interface_for, preflight_code
from utils.autorealize_context import _render_from_pack, select_autorealize_context_for_stage
from utils.evaluation_protocol import extract_contract_identity
from utils.metric import MetricValue
from utils.serialize import dumps_json, loads_json


def _node(score: float, branch: int = 1, **kwargs) -> SearchNode:
    return SearchNode(
        code="", stage="draft", branch_id=branch, is_buggy=False,
        is_valid=True, search_eligible=True, metric=MetricValue(score, maximize=False), **kwargs,
    )


def test_fusion_representatives_minimize_cost_and_ignore_rejected_or_mismatched_nodes():
    low, high, other = _node(4), _node(20), _node(9, branch=2)
    rejected = _node(-100)
    rejected.search_eligible = False
    opposite = _node(0)
    opposite.metric = MetricValue(0, maximize=True)
    invalid = _node(float("nan"))
    agent = SimpleNamespace(
        branch_successful_nodes={1: [high, rejected, opposite, invalid, low], 2: [other]},
        metric_maximize=False,
    )
    assert get_branch_top_nodes(agent, 1, top_k=3) == [low, high]
    assert _collect_branch_representatives(agent) == [low, other]
    assert get_branch_top_nodes(agent, 1, top_k=-1) == []


def test_fusion_provenance_survives_execution_and_journal_roundtrip():
    node = _node(4, fusion_sources=["source-a", "source-b"])
    node.evaluation_protocol = {"expected_evaluation_id": "evaluation:official", "split_fingerprint": "stale"}
    node.absorb_exec_result(ExecutionResult(
        term_out=[
            "Evaluation Protocol Summary: malformed\n",
            'Evaluation Protocol Summary: {"evaluation_id":"evaluation:official","split_fingerprint":"split:actual","evaluation_count":12}\n',
            "Final Validation Score: 4\n",
        ], exec_time=1.5, exc_type=None,
    ))
    restored = loads_json(dumps_json(Journal([node])), Journal).nodes[0]
    assert restored.fusion_sources == ["source-a", "source-b"]
    assert restored.evaluation_protocol["split_fingerprint"] == "split:actual"
    assert restored.evaluation_protocol["expected_evaluation_id"] == "evaluation:official"
    assert restored.score_recomputed is False
    node.absorb_exec_result(ExecutionResult(term_out=["failed retry\n"], exec_time=0.2, exc_type="ValueError"))
    assert node.evaluation_protocol == {"expected_evaluation_id": "evaluation:official"}


def test_execution_contract_survives_json_pack_rendering_and_stage_routing():
    identity = {"schema_version": "autorealize.execution_contract.v1", "contract_id": "task:123", "evaluation_id": "evaluation:456", "readiness": "ready_with_assumptions"}
    rendered = _render_from_pack({"execution_contract": identity, "task_goal": "Predict sales"})
    for stage in ("fast_draft", "solver_design", "merge", "code_review"):
        routed = select_autorealize_context_for_stage(rendered, stage)
        assert extract_contract_identity(routed) == {"expected_contract_id": "task:123", "expected_evaluation_id": "evaluation:456"}
    unrelated = '## Execution Contract\nNo IDs available.\n## Other\n```json\n' + json.dumps(identity) + '\n```'
    assert extract_contract_identity(unrelated) == {}


def test_structured_decision_route_has_no_prediction_wrapper_requirement():
    agent = SimpleNamespace(
        task_desc="Produce the requested output.", coldstart_description="",
        data_preview="## AutoRealize Structured Context\n\n## Method Strategy\n- problem_structure: decision_optimization\n",
        start_time=0, current_step=1,
        acfg=SimpleNamespace(time_limit=3600, steps=4),
        cfg=SimpleNamespace(exec=SimpleNamespace(timeout=60)),
    )
    result = get_impl_guideline_from_agent(agent)
    guidelines = "\n".join(result["Implementation guideline"])
    assert "solve(model_path, data)" in guidelines
    assert "MUST define `def train(data, artifact_dir)" not in guidelines
    assert "Decision / Optimization Solution Protocol (conditional, strict)" in result
    assert list(result)[-1] == "Current execution budget"


def test_resource_budget_changes_leave_long_implementation_prefix_identical():
    first = get_impl_guideline(3600, 20, 120)
    second = get_impl_guideline(1200, 8, 60)
    assert first["Implementation guideline"] == second["Implementation guideline"]
    assert first["Current execution budget"] != second["Current execution budget"]


def test_generation_and_review_require_selected_recipe_for_reusable_training():
    generation = "\n".join(get_impl_guideline(3600, 20, 120)["Implementation guideline"])
    review = "\n".join(get_code_review_guidelines())
    assert "a_new_artifact_dir" in generation
    for rules in (generation, review):
        assert "selected hyperparameters" in rules
        assert "relative to the solution" in rules
        assert "artifact manifest" in rules
        assert "fallback" in rules or "fall back" in rules


def test_preflight_uses_exported_entrypoints_despite_estimator_method_names():
    interface = interface_for(task_family="prediction", method_family="linear")
    report = preflight_code("""
class Estimator:
    def predict(self, data): return data
    def train(self, data): return self

def train(data, artifact_dir): return Estimator()
def predict(model_path, data): return data
""", interface)
    assert report.ok
    assert report.discovered_functions == ("predict", "train")
    positional = preflight_code("def train(data, artifact_dir, /): pass\ndef predict(model_path, data, /): pass", interface)
    assert positional.ok


def test_preflight_rejects_class_and_nested_functions_as_module_exports():
    interface = interface_for(task_family="prediction", method_family="linear")
    report = preflight_code("""
class Estimator:
    def predict(model_path, data): return data

def factory():
    def train(data, artifact_dir): pass
    return train
""", interface)
    assert not report.ok
    assert report.missing_functions == ("predict", "train")
    assert report.discovered_functions == ("factory",)


def test_preflight_still_checks_safety_inside_classes_and_nested_functions():
    interface = interface_for(task_family="prediction", method_family="linear")
    report = preflight_code("""
class Estimator:
    def predict(self, data): return eval(data)

def train(data, artifact_dir): pass
def predict(model_path, data):
    def unsafe(): exec(data)
    return data
""", interface)
    assert not report.ok
    assert len(report.dangerous_findings) == 2
    assert not report.missing_functions
    assert not report.invalid_signatures
