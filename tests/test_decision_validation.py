from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "decision_validation.py"
_SPEC = importlib.util.spec_from_file_location(
    "mlevolve_decision_validation_for_test",
    _MODULE_PATH,
)
assert _SPEC and _SPEC.loader
_DV = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DV)

decision_signal_summary = _DV.decision_signal_summary
extract_decision_validation_summary = _DV.extract_decision_validation_summary


def test_extracts_last_candidate_reported_decision_summary() -> None:
    output = (
        'Decision Validation Summary: {"feasible": false, "violations": ["first"]}\n'
        'Decision Validation Summary: {"feasible": true, "final_score_source": "score_solution"}\n'
    )

    summary = extract_decision_validation_summary(output)

    assert summary == {"feasible": True, "final_score_source": "score_solution"}


def test_signal_summary_labels_candidate_values_without_trust_verdict() -> None:
    summary = {
        "evaluator_self_tests_passed": True,
        "feasible": False,
        "violations": ["capacity"],
        "final_score_source": "total_penalized_cost",
        "score_components": {"objective_cost": 10.0, "penalty": 100.0},
    }

    signals = decision_signal_summary(summary)

    assert signals == {
        "reported_final_score_source": "total_penalized_cost",
        "reported_score_component_count": 2,
        "reported_evaluator_self_tests_passed": True,
        "reported_feasible": False,
        "reported_violation_count": 1,
    }
    assert not any("trusted" in key for key in signals)


def test_invalid_or_missing_summary_produces_no_program_verdict() -> None:
    assert extract_decision_validation_summary(
        "Decision Validation Summary: not-json\nFinal Validation Score: 0\n"
    ) is None
    assert decision_signal_summary(None) == {}
