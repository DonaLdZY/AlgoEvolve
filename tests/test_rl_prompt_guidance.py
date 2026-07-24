from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_rl_stepwise_prompt_has_diagnostics_without_internal_baseline():
    stepwise = _read("agents/coder/stepwise_coder.py")
    shared = _read("agents/prompts/shared.py")
    combined = stepwise + "\n" + shared

    assert "baseline" not in combined.lower()
    assert "RL Design Summary" in stepwise
    assert "Candidate/Action Probe Summary" in stepwise
    assert "Env Smoke Trace" in stepwise
    assert "Method Usage Summary" in stepwise
    assert "unused_rl_scaffold" in stepwise
    assert "Curriculum Plan" in stepwise
    assert "small to full instances" in stepwise
    assert "mask.any()" in shared
    assert "all-invalid mask" in shared
    assert "original full-scale environment" in shared
    assert "城市配送" not in combined
    assert "承运商" not in combined


def test_rl_review_prompt_flags_unused_rl_scaffolds_without_task_specific_gates():
    review = _read("agents/prompts/validation_template_prompts.py")

    assert "unused RL scaffolds" in review
    assert "saved policy artifact" in review
    assert "final evaluated rollout" in review
    assert "all-invalid mask" in review
    assert "curriculum learning" in review
    assert "coverage_ok" not in review
    assert "hard_violations" not in review
    assert "baseline gap" not in review.lower()
