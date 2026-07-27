from __future__ import annotations

from types import SimpleNamespace

from agents.planner.planner_with_memory import _build_refinement_guidance


def test_refinement_guidance_serializes_decision_signals() -> None:
    record = SimpleNamespace(
        record_id="node-1",
        description="Use a constrained solver",
        method="mixed_integer_programming",
    )

    guidance = _build_refinement_guidance(
        [(record, 0.9)],
        [],
        metadata_map={
            "node-1": {
                "decision_signals": {
                    "reported_feasible": True,
                    "objective_value": 42.5,
                }
            }
        },
    )

    assert '"reported_feasible": true' in guidance
    assert '"objective_value": 42.5' in guidance
