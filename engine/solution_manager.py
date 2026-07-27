"""Top-K candidate management and result persistence (update_top_candidates, save_top_candidates, get_branch_top_nodes, save_best_solution, update_best_solution, write_metric_file)."""

import shutil
import logging
import json
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List

from engine.model_artifacts import find_model_artifacts
from engine.search_node import SearchNode
from engine.solution_protocol import interface_for, solution_manifest
from agents.prompt_policy import infer_task_family

logger = logging.getLogger("AlgoEvolve")


_STAGE_LABELS = {
    'fusion_draft': 'fusion_draft (multi-branch aggregation)',
    'draft': 'draft (initial solution)',
    'improve': 'improve (refinement)',
    'evolution': 'evolution (intra-branch evolution)',
    'fusion': 'fusion (cross-branch fusion)',
    'debug': 'debug (bug fixing)',
}


def format_stage_display(stage: str) -> str:
    """Map stage value to human-readable label."""
    return _STAGE_LABELS.get(stage, stage)


def write_metric_file(filepath, node, metric_maximize: bool) -> None:
    """Write metric.txt with metric value, maximize, branch_id, stage, from_topk, exec/created time."""
    with open(filepath, "w") as f:
        f.write(f"Metric: {node.metric.value}\n")
        f.write(f"Maximize: {metric_maximize}\n")

        if hasattr(node, 'branch_id') and node.branch_id is not None:
            f.write(f"Branch ID: {node.branch_id}\n")
        else:
            f.write(f"Branch ID: N/A\n")

        if hasattr(node, 'stage') and node.stage:
            f.write(f"Stage: {format_stage_display(node.stage)}\n")
        else:
            f.write(f"Stage: N/A\n")

        if hasattr(node, 'from_topk'):
            f.write(f"From Top-K: {node.from_topk}\n")
        else:
            f.write(f"From Top-K: False\n")

        f.write(f"Search Eligible: {bool(getattr(node, 'search_eligible', False))}\n")
        f.write(f"Result Review Verdict: {getattr(node, 'review_verdict', '') or 'N/A'}\n")
        f.write(f"Score Recomputed: {bool(getattr(node, 'score_recomputed', False))}\n")
        f.write(f"Evidence Source: {getattr(node, 'certification_source', '') or 'N/A'}\n")
        f.write(f"Method Mode: {getattr(node, 'method_mode', 'unknown')}\n")

        if node.exec_time is not None:
            f.write(f"Execution Time(s): {node.exec_time:.2f}\n")
        else:
            f.write(f"Execution Time(s): N/A\n")

        if hasattr(node, 'created_time') and node.created_time:
            f.write(f"Created Time: {node.created_time}\n")
        else:
            f.write(f"Created Time: N/A\n")


def copy_model_artifacts(agent, node: SearchNode, target_dir) -> None:
    """Copy node-specific model artifacts into a solution directory."""
    artifacts = find_model_artifacts(agent.cfg.workspace_dir, str(node.id))
    artifacts_dir = target_dir / "model_artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    if not artifacts:
        logger.warning(f"No model artifacts found for node {node.id}")
        return

    artifacts_dir.mkdir(exist_ok=True, parents=True)
    manifest_lines = []
    for src in artifacts:
        try:
            dst = artifacts_dir / src.name
            shutil.copy2(src, dst)
            manifest_lines.append(f"- {src.name} <- {src}")
        except Exception as e:
            logger.error(f"Failed to copy model artifact for node {node.id}: {src} -> {e}")
    if manifest_lines:
        (target_dir / "model_artifacts_manifest.md").write_text(
            "# Model Artifacts\n\n" + "\n".join(manifest_lines) + "\n",
            encoding="utf-8",
        )
        primary = artifacts_dir / artifacts[0].name
        (target_dir / "model_path.txt").write_text(str(primary), encoding="utf-8")


def write_solution_manifest(agent, node: SearchNode, target_dir: Path) -> dict:
    """Write the finite cross-system interface contract next to exported code."""

    interface = interface_for(
        task_family=infer_task_family(agent),
        method_family=str(getattr(node, "method_family", "unknown") or "unknown"),
    )
    model_path_file = target_dir / "model_path.txt"
    artifact_path = (
        model_path_file.read_text(encoding="utf-8").strip()
        if model_path_file.exists()
        else None
    )
    payload = solution_manifest(
        interface,
        artifact_path=artifact_path or None,
        node_id=str(node.id),
        method_family=str(getattr(node, "method_family", "unknown") or "unknown"),
    )
    (target_dir / "solution_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def save_best_solution(agent, result_node, submission_file_path) -> None:
    """Save best solution code, submission, and meta to disk (thread-safe via agent.save_node_lock)."""
    best_solution_dir = agent.cfg.workspace_dir / "best_solution"
    best_submission_dir = agent.cfg.workspace_dir / "best_submission"
    generate_submission = getattr(agent.acfg, "generate_submission", True)

    with agent.save_node_lock:
        best_solution_dir.mkdir(exist_ok=True, parents=True)

        if generate_submission:
            best_submission_dir.mkdir(exist_ok=True, parents=True)
            if submission_file_path.exists():
                shutil.copy(
                    submission_file_path,
                    best_submission_dir / "submission.csv",
                )
            else:
                logger.warning(
                    f"Best node {result_node.id} has no submission file to copy: {submission_file_path}"
                )

        with open(best_solution_dir / "solution.py", "w") as f:
            f.write(result_node.code)

        with open(best_solution_dir / "node_id.txt", "w") as f:
            f.write(str(result_node.id))

        copy_model_artifacts(agent, result_node, best_solution_dir)
        write_solution_manifest(agent, result_node, best_solution_dir)

        write_metric_file(
            best_solution_dir / "metric.txt",
            result_node,
            agent.metric_maximize,
        )


def update_top_candidates(agent, new_node: SearchNode) -> None:
    """Maintain a top-N list of best candidates by metric (higher is better if maximize else lower).
    Only consider nodes that are not buggy and have a valid metric value.
    Each branch contributes at most 5 candidates to ensure diversity.
    """
    if (
        not new_node
        or not getattr(new_node, "search_eligible", False)
        or getattr(new_node, "is_valid", None) is False
        or not new_node.metric
        or new_node.metric.value is None
    ):
        return

    # Avoid duplicates (by node id)
    existing_ids = {n.id for n in agent.top_candidates}
    if new_node.id not in existing_ids:
        agent.top_candidates.append(new_node)

    if agent.metric_maximize is None:
        logger.warning("metric_maximize not initialized, using default value True")
        maximize = True
    else:
        maximize = agent.metric_maximize

    branch_nodes = defaultdict(list)

    for node in agent.top_candidates:
        branch_id = getattr(node, 'branch_id', None)
        if branch_id is None:
            branch_id = -1
        branch_nodes[branch_id].append(node)

    branch_top_nodes = []
    max_per_branch = 5

    for branch_id, nodes in branch_nodes.items():
        nodes.sort(
            key=lambda n: (
                n.metric.value
                if (n.metric and n.metric.value is not None)
                else (float('-inf') if maximize else float('inf'))
            ),
            reverse=maximize
        )
        branch_top_nodes.extend(nodes[:max_per_branch])

    branch_top_nodes.sort(
        key=lambda n: (
            n.metric.value
            if (n.metric and n.metric.value is not None)
            else (float('-inf') if maximize else float('inf'))
        ),
        reverse=maximize
    )

    agent.top_candidates = branch_top_nodes[:agent.top_k]


def save_top_candidates(agent) -> None:
    """Persist top-N candidates' code and submissions into workspace directories for offline inspection.
    All top-N files are organized under a single 'top_solution/' directory for better organization.
    Does not change best_node logic. Thread-safe with save_node_lock.
    """
    generate_submission = getattr(agent.acfg, "generate_submission", True)
    with agent.save_node_lock:
        top_solution_dir = agent.cfg.workspace_dir / "top_solution"
        top_solution_dir.mkdir(exist_ok=True, parents=True)

        for rank, node in enumerate(agent.top_candidates, start=1):
            rank_dir = top_solution_dir / f"top{rank}"
            rank_dir.mkdir(exist_ok=True, parents=True)

            # Save code and meta
            try:
                with open(rank_dir / "solution.py", "w") as f:
                    f.write(node.code)
                with open(rank_dir / "node_id.txt", "w") as f:
                    f.write(str(node.id))
                write_metric_file(
                    rank_dir / "metric.txt",
                    node,
                    agent.metric_maximize,
                )
                copy_model_artifacts(agent, node, rank_dir)
                write_solution_manifest(agent, node, rank_dir)
            except Exception as e:
                logger.error(f"Failed to save top{rank} solution files for node {node.id}: {e}")

            if not generate_submission:
                continue

            # Copy submission to the same directory
            submission_file_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
            target_submission_path = rank_dir / "submission.csv"

            if submission_file_path.exists():
                try:
                    shutil.copy(submission_file_path, target_submission_path)
                    logger.info(f"Saved top{rank} submission for node {node.id}")
                except Exception as e:
                    logger.error(f"Failed to copy top{rank} submission for node {node.id}: {e}")
            else:
                # Best-effort search for alternative matching file
                submission_dir = agent.cfg.workspace_dir / "submission"
                if submission_dir.exists():
                    for file in submission_dir.iterdir():
                        if node.id in file.name and file.name.endswith('.csv'):
                            try:
                                shutil.copy(file, target_submission_path)
                                logger.info(
                                    f"Found alternative submission for top{rank} node {node.id}: {file.name}")
                            except Exception as e:
                                logger.error(f"Failed to copy alternative submission for node {node.id}: {e}")
                            break


def rebuild_top_candidates(agent) -> list[SearchNode]:
    """Rebuild the deliverable Top-K from the durable journal after resume/interrupt."""
    agent.top_candidates = []
    for node in agent.journal.nodes:
        update_top_candidates(agent, node)
    return list(agent.top_candidates)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _remove_stale_rank_dirs(root: Path, keep_count: int) -> None:
    if not root.exists():
        return
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"top(\d+)", child.name)
        if match and int(match.group(1)) > keep_count:
            shutil.rmtree(child)


def _save_checkpoint_candidates(agent, nodes: list[SearchNode]) -> list[dict]:
    """Export searchable candidates without presenting them as delivery-ready Top-K."""
    root = agent.cfg.workspace_dir / "checkpoint_candidates"
    rows: list[dict] = []
    with agent.save_node_lock:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        for rank, node in enumerate(nodes[: agent.top_k], start=1):
            rank_dir = root / f"top{rank}"
            rank_dir.mkdir(parents=True, exist_ok=True)
            (rank_dir / "solution.py").write_text(node.code or "", encoding="utf-8")
            (rank_dir / "node_id.txt").write_text(str(node.id), encoding="utf-8")
            write_metric_file(rank_dir / "metric.txt", node, agent.metric_maximize)
            copy_model_artifacts(agent, node, rank_dir)
            write_solution_manifest(agent, node, rank_dir)

            submission_source = (
                agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
            )
            if submission_source.exists():
                shutil.copy2(submission_source, rank_dir / "submission.csv")
            rows.append(
                {
                    "rank": rank,
                    "node_id": str(node.id),
                    "metric": node.metric.value,
                    "maximize": node.metric.maximize,
                    "stage": node.stage,
                    "method_mode": getattr(node, "method_mode", "unknown"),
                    "delivery_ready": bool(getattr(node, "delivery_ready", False)),
                    "solution_path": str(rank_dir / "solution.py"),
                    "submission_path": str(rank_dir / "submission.csv"),
                    "model_artifacts_dir": str(rank_dir / "model_artifacts"),
                }
            )
    return rows


def persist_resumable_checkpoint(
    agent,
    *,
    status: str,
    reason: str,
    active_actions: list[dict] | None = None,
    manifest_filename: str = "checkpoint_manifest.json",
    materialize_artifacts: bool = True,
) -> dict:
    """Commit an artifact index after journal/search-state writes.

    User interruption uses ``materialize_artifacts=False`` because accepted Top-K
    nodes are already exported when they enter Top-K. Re-copying every model and
    solution here delays process termination without improving resumability.
    """
    top_nodes = rebuild_top_candidates(agent)
    if materialize_artifacts and top_nodes:
        save_top_candidates(agent)
        _remove_stale_rank_dirs(
            agent.cfg.workspace_dir / "top_solution",
            len(top_nodes),
        )
        best = top_nodes[0]
        agent.best_node = best
        submission_path = (
            agent.cfg.workspace_dir / "submission" / f"submission_{best.id}.csv"
        )
        save_best_solution(agent, best, submission_path)

    provisional_nodes = [
        node
        for node in agent.journal.nodes
        if getattr(node, "search_eligible", False)
        and node.metric
        and node.metric.value is not None
    ]
    provisional_nodes.sort(key=lambda node: node.metric, reverse=True)
    checkpoint_candidate_rows = (
        _save_checkpoint_candidates(agent, provisional_nodes)
        if materialize_artifacts
        else [
            {
                "rank": rank,
                "node_id": str(node.id),
                "metric": node.metric.value,
                "maximize": node.metric.maximize,
                "stage": node.stage,
                "method_mode": getattr(node, "method_mode", "unknown"),
                "delivery_ready": bool(getattr(node, "delivery_ready", False)),
                "materialized": False,
                "source": "journal",
            }
            for rank, node in enumerate(provisional_nodes[: agent.top_k], start=1)
        ]
    )

    top_rows = []
    for rank, node in enumerate(top_nodes, start=1):
        rank_dir = agent.cfg.workspace_dir / "top_solution" / f"top{rank}"
        top_rows.append(
            {
                "rank": rank,
                "node_id": str(node.id),
                "metric": node.metric.value,
                "maximize": node.metric.maximize,
                "stage": node.stage,
                "method_mode": getattr(node, "method_mode", "unknown"),
                "delivery_certified": bool(
                    getattr(node, "delivery_certified", False)
                ),
                "solution_path": str(rank_dir / "solution.py"),
                "submission_path": str(rank_dir / "submission.csv"),
                "model_artifacts_dir": str(rank_dir / "model_artifacts"),
                "materialized": (rank_dir / "solution.py").is_file(),
            }
        )

    payload = {
        "schema_version": "algoevolve.checkpoint.v1",
        "checkpoint_id": uuid.uuid4().hex,
        "status": status,
        "reason": reason,
        "resumable": status == "interrupted_resumable",
        "artifacts_materialized": materialize_artifacts,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "completed_nodes": max(0, len(agent.journal) - 1),
        "journal_path": str(agent.cfg.log_dir / "journal.json"),
        "filtered_journal_path": str(agent.cfg.log_dir / "filtered_journal.json"),
        "search_state_path": str(
            agent.cfg.log_dir
            / str(
                getattr(
                    getattr(agent.cfg, "runtime", None),
                    "search_state_filename",
                    "search_state.json",
                )
                or "search_state.json"
            )
        ),
        "best_solution_dir": str(agent.cfg.workspace_dir / "best_solution"),
        "top_solution_dir": str(agent.cfg.workspace_dir / "top_solution"),
        "top_solutions": top_rows,
        "checkpoint_candidates_dir": str(
            agent.cfg.workspace_dir / "checkpoint_candidates"
        ),
        "provisional_top": checkpoint_candidate_rows,
        "active_actions": active_actions or [],
    }
    log_manifest = agent.cfg.log_dir / manifest_filename
    workspace_manifest = agent.cfg.workspace_dir / manifest_filename
    _atomic_write_json(log_manifest, payload)
    _atomic_write_json(workspace_manifest, payload)
    logger.info(
        "Committed %s checkpoint: top=%s provisional=%s active_actions=%s",
        status,
        len(top_rows),
        len(payload["provisional_top"]),
        len(payload["active_actions"]),
    )
    return payload


def get_branch_top_nodes(agent, branch_id: int, top_k: int = 3) -> List[SearchNode]:
    """Return top-k nodes for a branch, sorted by metric."""
    if branch_id not in agent.branch_successful_nodes:
        logger.info(f"Branch {branch_id} has no successful nodes")
        return []

    successful_nodes = agent.branch_successful_nodes[branch_id]

    if not successful_nodes:
        logger.info(f"Branch {branch_id} has no successful nodes")
        return []

    maximize = agent.metric_maximize if agent.metric_maximize is not None else True

    sorted_nodes = sorted(
        successful_nodes,
        key=lambda n: n.metric.value if n.metric and n.metric.value is not None else (
            float('-inf') if maximize else float('inf')),
        reverse=maximize
    )

    result = sorted_nodes[:top_k]

    logger.info(f"Branch {branch_id}: found {len(successful_nodes)} successful nodes, returning top {len(result)}")
    for i, node in enumerate(result):
        logger.debug(f"  Top {i + 1}: Node {node.id}, Metric: {node.metric.value}")

    return result


def update_best_solution(agent, node):
    """Update Top-K/best from the accepted Result Review set."""
    if not node.metric or node.metric.value is None:
        return

    submission_file_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    if getattr(node, "search_eligible", False):
        provisional = getattr(agent, "provisional_best_node", None)
        if provisional is None or provisional.metric < node.metric:
            agent.provisional_best_node = node
            logger.info(
                "[provisional-best] updated: node %s, metric=%s",
                node.id,
                node.metric.value,
            )

    if not getattr(node, "search_eligible", False) or getattr(node, "is_valid", None) is False:
        return

    update_top_candidates(agent, node)
    save_top_candidates(agent)

    if agent.best_node is None or agent.best_node.metric < node.metric:
        agent.best_node = node
        save_best_solution(agent, node, submission_file_path)
        certification = "certified" if getattr(node, "delivery_certified", False) else "provisional-score"
        logger.info(
            "[best] updated: node %s, metric=%s, status=%s",
            node.id,
            node.metric.value,
            certification,
        )
    else:
        logger.debug(f"Node {node.id} not the best (current best: {agent.best_node.id})")
