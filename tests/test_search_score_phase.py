from types import SimpleNamespace

import pytest

from engine.execution import update_node_certification, validate_executed_node
from engine.final_evaluation import require_unexposed_search, finalize_selected_candidate
from engine.journal_snapshot import NodeSnapshot
import json
from engine.search_node import SearchNode
from utils.evaluation_protocol import extract_evaluation_protocol
from utils.metric import MetricValue


@pytest.mark.parametrize("protocol,accepted", [
    ({}, False),
    ({"evaluation_phase": "final", "scoring_split": "holdout", "final_evaluation_accessed": True}, False),
    ({"evaluation_phase": "search", "scoring_split": "development", "final_evaluation_accessed": "false"}, False),
    ({"evaluation_phase": "search", "scoring_split": "development", "final_evaluation_accessed": False}, True),
])
def test_formal_task_requires_search_score_provenance(tmp_path, protocol, accepted):
    agent = SimpleNamespace(has_autorealize_context=True, cfg=SimpleNamespace(workspace_dir=tmp_path),
                            task_desc="Forecast usage", data_preview="", coldstart_description="")
    node = SearchNode(code="def predict(model_path, data): return data", stage="draft", is_buggy=False,
                      is_valid=True, metric=MetricValue(3.994, maximize=False), evaluation_protocol=protocol)
    node._term_out = []
    update_node_certification(agent, node)
    assert node.search_eligible is accepted


def test_protocol_preserves_phase_and_structured_cutoffs():
    protocol = extract_evaluation_protocol('Evaluation Protocol Summary: {"evaluation_phase":"search","scoring_split":"development","final_evaluation_accessed":false,"cutoff":{"fit_end":21024,"development_end":28032}}')
    assert protocol["cutoff"]["development_end"] == 28032
    assert protocol["final_evaluation_accessed"] is False


def test_decision_summary_protocol_reaches_normal_search_certification(tmp_path):
    expected = {"evaluation_phase": "search", "scoring_split": "not_applicable_static_optimization",
                "final_evaluation_accessed": False, "split_fingerprint": "not_applicable_static_optimization"}
    output = "Decision Validation Summary: " + json.dumps({
        "status": "blocked_data_gap", "evaluation_protocol": expected,
        "software_contract_check_passed": 12, "software_contract_check_total": 12,
    })
    protocol = extract_evaluation_protocol(output)
    assert protocol == expected
    agent = SimpleNamespace(has_autorealize_context=True, cfg=SimpleNamespace(workspace_dir=tmp_path),
                            task_desc="Vehicle routing optimization", data_preview="", coldstart_description="")
    node = SearchNode(code="def solve(model_path, data): return data", stage="draft", is_buggy=False,
                      is_valid=True, metric=MetricValue(100, maximize=True), evaluation_protocol=protocol)
    node._term_out = [output]
    update_node_certification(agent, node)
    assert node.search_eligible
    assert not node.score_recomputed


@pytest.mark.parametrize("conflicting", [True, "false", 0])
def test_conflicting_protocol_wrappers_are_not_silently_accepted(conflicting):
    standalone = {"evaluation_phase": "search", "scoring_split": "development", "final_evaluation_accessed": False}
    nested = {**standalone, "final_evaluation_accessed": conflicting}
    output = ("Evaluation Protocol Summary: " + json.dumps(standalone) + "\n"
              "Decision Validation Summary: " + json.dumps({"evaluation_protocol": nested}))
    protocol = extract_evaluation_protocol(output)
    assert "evaluation_phase" not in protocol
    assert protocol["final_evaluation_accessed"] is (conflicting is True)


def test_decision_summary_does_not_infer_missing_phase_or_false_boolean():
    protocol = extract_evaluation_protocol('Decision Validation Summary: {"status":"blocked_data_gap","evaluation_protocol":{"final_evaluation_accessed":"false"}}')
    assert protocol == {"final_evaluation_accessed": "false"}


def test_search_exposure_is_preserved_and_cannot_resume_or_finalize(tmp_path):
    agent = SimpleNamespace(has_autorealize_context=True, cfg=SimpleNamespace(workspace_dir=tmp_path),
                            task_desc="Classify failures", data_preview="", coldstart_description="",
                            accept_search_results=True)
    node = SearchNode(code="def predict(model_path, data): return data", stage="draft", is_buggy=True,
                      is_valid=False, metric=MetricValue(0.8, maximize=True),
                      evaluation_protocol={"final_evaluation_accessed": True})
    node._term_out = ["saved execution evidence"]
    validate_executed_node(agent, node)
    assert not agent.accept_search_results
    assert agent.search_integrity_failure == "search_evaluation_exposure"
    marker = json.loads((tmp_path / "search_evaluation_exposure.json").read_text(encoding="utf-8"))
    restored = NodeSnapshot.from_payload(marker["node"]).to_node()
    assert restored.id == node.id
    assert restored._term_out == ["saved execution evidence"]
    assert not restored.search_eligible
    with pytest.raises(RuntimeError, match="cannot resume"):
        require_unexposed_search(tmp_path)
    with pytest.raises(RuntimeError, match="Cannot finalize"):
        finalize_selected_candidate(agent.cfg, node, None)
