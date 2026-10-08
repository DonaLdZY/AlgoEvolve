"""Post-execution validation: validate_executed_node (csv existence, metric=0.0, register success)."""

import logging
import json
import math
import os
import re
from contextlib import nullcontext
from pathlib import Path
import time

from engine.model_artifacts import find_model_artifacts
from engine.search_node import SearchNode
from agents.prompts import infer_task_mode
from agents.prompt_policy import method_family_for_node

logger = logging.getLogger("AlgoEvolve")

_ZERO_METRIC_ANALYSIS = (
    "Performance is 0.0 (complete failure). This indicates fundamental issues that need debugging:\n"
    "1. Model architecture may be incorrect or not learning\n"
    "2. Data preprocessing might be broken (wrong format, normalization issues)\n"
    "3. Loss function or evaluation metric calculation may be faulty\n"
    "4. Training loop might not be updating weights properly\n"
    "5. Input data might not be loaded correctly\n\n"
    "Please review the code carefully to identify the root cause."
)


def _append_nonfatal_decision_warning(node: SearchNode, message: str) -> None:
    if node.analysis:
        if message not in node.analysis:
            node.analysis = f"{node.analysis}\n\n[Non-fatal warning] {message}"
    else:
        node.analysis = f"[Non-fatal warning] {message}"
    node.parser_analysis = node.analysis


def _has_scorable_decision_run(agent, node: SearchNode) -> bool:
    mode = infer_task_mode(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
        autorealize_context=getattr(agent, "data_preview", ""),
    )
    if mode not in {"optimization", "rl"}:
        return False
    metric = getattr(node, "metric", None)
    return metric is not None and getattr(metric, "value", None) is not None


def _has_predict_api(node: SearchNode) -> bool:
    return bool(
        re.search(
            r"def\s+predict\s*\(\s*model_path(?:\s*:\s*[^,)=]+)?(?:\s*=\s*[^,)]*)?\s*,\s*data(?:\s*:\s*[^,)=]+)?(?:\s*=\s*[^,)]*)?\s*[,)]",
            node.code or "",
        )
    )


def _method_mode(agent, node: SearchNode) -> str:
    task_mode = infer_task_mode(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
        autorealize_context=getattr(agent, "data_preview", ""),
    )
    output = (node.term_out or "").lower()
    usage_excerpt = ""
    marker = output.rfind("method usage summary")
    if marker >= 0:
        usage_excerpt = output[marker : marker + 1200]
    for value in ("pure_rl", "hybrid_rl", "unused_rl_scaffold", "non_rl_solver"):
        if value in usage_excerpt:
            return value
    if task_mode == "rl":
        return "unused_rl_scaffold"
    if task_mode == "optimization":
        return "non_rl_solver"
    return "prediction"


def update_node_certification(agent, node: SearchNode) -> None:
    """Record evidence without adding a third delivery-eligibility review layer."""

    metric = getattr(node, "metric", None)
    protocol = getattr(node, "evaluation_protocol", {}) or {}
    if getattr(agent, "has_autorealize_context", False) and node.exc_type is None:
        phase_ok = (
            protocol.get("evaluation_phase") == "search"
            and isinstance(protocol.get("scoring_split"), str)
            and bool(protocol["scoring_split"].strip())
            and protocol.get("final_evaluation_accessed") is False
        )
        if not phase_ok:
            node.is_valid = False
            _append_nonfatal_decision_warning(node, "Search scoring provenance is missing or reports final-evaluation access; this candidate cannot be ranked. Required: evaluation_phase=search, actual scoring_split, final_evaluation_accessed=false.")
    metric_value = getattr(metric, "value", None) if metric is not None else None
    finite_metric = isinstance(metric_value, (int, float)) and math.isfinite(float(metric_value))
    artifacts = find_model_artifacts(agent.cfg.workspace_dir, str(node.id))
    node.method_mode = _method_mode(agent, node)
    node.method_family = method_family_for_node(
        node, task_family="prediction" if node.method_mode == "prediction" else "decision"
    )
    node.runtime_ok = node.exc_type is None and node.is_buggy is False
    node.contract_valid = node.is_valid is not False
    node.search_eligible = bool(node.runtime_ok and finite_metric and node.contract_valid)
    interface_kind = str(getattr(node, "solution_interface", "") or "")
    stateful_interface = interface_kind in {
        "prediction",
        "reinforcement_learning",
    } or node.method_mode in {"pure_rl", "hybrid_rl"}
    node.artifact_ready = bool(artifacts) if stateful_interface else True

    # Deprecated compatibility fields mirror acceptance/certification; they no
    # longer form an independent gate for best-node or Top-K selection.
    output_deferred = getattr(node, "output_scope", "unknown") == "development_validation_only"
    node.delivery_ready = node.search_eligible and not output_deferred
    node.delivery_certified = bool(node.delivery_ready and node.score_recomputed)
    node.certification_source = (
        "independent_evaluator"
        if node.delivery_certified
        else "candidate_reported_score"
        if node.search_eligible
        else "runtime_or_validation_failure"
    )
    notes: list[str] = []
    if output_deferred:
        notes.append("Development predictions were reviewed; real NEW output was not available. Independent inference delivery remains pending.")
    if node.search_eligible and not node.score_recomputed:
        notes.append("Final score was reported by candidate code and has not been independently recomputed.")
    if stateful_interface and not artifacts:
        notes.append("The stateful interface passed preflight, but no persisted model/policy artifact was found after execution.")
    if not node.contract_valid:
        notes.append("Output or task-contract validation is incomplete.")
    node.certification_notes = notes


def record_independent_score(
    agent,
    node: SearchNode,
    recomputed_score: float,
    *,
    source: str = "trusted_evaluator",
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> bool:
    """Apply a trusted evaluator replay result without changing search history."""

    metric = getattr(node, "metric", None)
    reported = getattr(metric, "value", None) if metric is not None else None
    matches = bool(
        isinstance(reported, (int, float))
        and math.isfinite(float(recomputed_score))
        and math.isclose(float(reported), float(recomputed_score), rel_tol=rel_tol, abs_tol=abs_tol)
    )
    node.score_recomputed = matches
    if not matches:
        node.is_valid = False
        _append_nonfatal_decision_warning(
            node,
            f"Independent evaluator mismatch: candidate reported {reported}, {source} recomputed {recomputed_score}.",
        )
    update_node_certification(agent, node)
    if matches and node.delivery_certified:
        node.certification_source = source
        node.certification_notes = [
            note
            for note in node.certification_notes
            if "not been independently recomputed" not in note
        ]
    return matches


def validate_executed_node(agent, node: SearchNode):
    """Finalize deterministic hard facts after the Result Review verdict."""
    update_node_certification(agent, node)
    protocol = getattr(node, "evaluation_protocol", {}) or {}
    if protocol.get("final_evaluation_accessed") is True:
        # Preserve evidence outside feedback memory and stop admitting work before
        # the rejected node can become a debug parent or expose its metrics again.
        agent.accept_search_results = False
        node.search_eligible = False
        node.delivery_ready = False
        node.delivery_certified = False
        from engine.journal_snapshot import NodeSnapshot
        marker = Path(agent.cfg.workspace_dir) / "search_evaluation_exposure.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        with getattr(agent, "journal_lock", nullcontext()):
            try:
                with marker.open("x", encoding="utf-8") as stream:
                    json.dump({"reason": "search_evaluation_exposure", "recorded_at": time.time(),
                               "node": NodeSnapshot.from_node(node).to_payload(),
                               "resume_allowed": False, "final_evaluation_allowed": False},
                              stream, ensure_ascii=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                pass
            finally:
                agent.search_integrity_failure = "search_evaluation_exposure"
        return

    if node.search_eligible and hasattr(node, 'branch_id') and node.branch_id:
        if node.branch_id not in agent.branch_successful_nodes:
            agent.branch_successful_nodes[node.branch_id] = []
        if all(existing.id != node.id for existing in agent.branch_successful_nodes[node.branch_id]):
            agent.branch_successful_nodes[node.branch_id].append(node)
