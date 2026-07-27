"""Search conditions: should_trigger_branch_fusion, is_branch_stagnant, is_globally_stagnant."""

import logging
import time
from agents.prompt_policy import infer_task_family, method_family_for_node

logger = logging.getLogger("AlgoEvolve")


def should_trigger_branch_fusion(agent) -> bool:
    """Gate root aggregation by half-time, evidence diversity, stagnation, and headroom."""
    if agent.fusion_draft_count >= agent.max_fusion_drafts:
        return False

    if not agent.search_start_time:
        return False

    scfg = agent.scfg
    elapsed_time = time.time() - agent.search_start_time
    total_time = max(0.0, float(getattr(agent.acfg, "time_limit", 0) or 0))
    if total_time > 0 and elapsed_time < total_time / 2:
        return False
    remaining_time = total_time - elapsed_time if total_time > 0 else float("inf")
    if remaining_time < max(0, int(getattr(scfg, "fusion_min_remaining_seconds", 300))):
        return False

    successful_branches = [
        bid for bid, nodes in agent.branch_successful_nodes.items()
        if len(nodes) >= scfg.fusion_min_successful_nodes
    ]
    if len(successful_branches) < scfg.fusion_min_branches:
        return False

    task_family = infer_task_family(agent)
    method_families = {
        method_family_for_node(node, task_family=task_family)
        for branch_id in successful_branches
        for node in agent.branch_successful_nodes.get(branch_id, [])
    }
    if len(method_families) < 2:
        return False

    if not is_globally_stagnant(agent):
        return False

    logger.info(
        f"Branch fusion conditions met at {elapsed_time/3600:.1f}h "
        f"with {len(successful_branches)} successful branches and families={sorted(method_families)}"
    )
    return True


def should_trigger_node_fusion(agent, parent_node) -> bool:
    """Use the same evidence gates for a cross-branch Fusion child."""

    if not agent.search_start_time:
        return False
    elapsed = time.time() - agent.search_start_time
    total = max(0.0, float(getattr(agent.acfg, "time_limit", 0) or 0))
    if total > 0 and elapsed < total / 2:
        return False
    if total > 0 and total - elapsed < max(
        0, int(getattr(agent.scfg, "fusion_min_remaining_seconds", 300))
    ):
        return False
    family = infer_task_family(agent)
    parent_family = method_family_for_node(parent_node, task_family=family)
    alternatives = {
        method_family_for_node(node, task_family=family)
        for branch_id, nodes in agent.branch_successful_nodes.items()
        if branch_id != getattr(parent_node, "branch_id", None)
        for node in nodes
        if getattr(node, "search_eligible", False)
    }
    return bool(alternatives - {parent_family})


def is_branch_stagnant(agent, branch_id: int, threshold: int = 3) -> bool:
    """Compare recent branch attempts with the best score that preceded them."""
    if branch_id not in agent.branch_successful_nodes:
        return False

    successful_nodes = agent.branch_successful_nodes[branch_id]
    if len(successful_nodes) <= threshold:
        return False

    maximize = agent.metric_maximize if agent.metric_maximize is not None else True

    scored = [
        node
        for node in successful_nodes
        if node.metric and node.metric.value is not None
    ]
    if len(scored) <= threshold:
        return False
    historical = scored[:-threshold]
    recent = scored[-threshold:]
    historical_best = (
        max(node.metric.value for node in historical)
        if maximize
        else min(node.metric.value for node in historical)
    )
    recent_best = (
        max(node.metric.value for node in recent)
        if maximize
        else min(node.metric.value for node in recent)
    )
    improvement = (
        recent_best - historical_best
        if maximize
        else historical_best - recent_best
    )
    stagnant = improvement <= float(agent.scfg.metric_improvement_threshold)
    if stagnant:
        logger.info(
            "Branch %s stagnant: recent_best=%s historical_best=%s threshold=%s",
            branch_id,
            recent_best,
            historical_best,
            agent.scfg.metric_improvement_threshold,
        )
    return stagnant


def is_globally_stagnant(agent) -> bool:
    """Compare the recent window with the best score that existed before it."""
    window_size = agent.stagnation_threshold
    scored_nodes = [
        node
        for node in agent.journal.nodes
        if getattr(node, "search_eligible", False)
        and node.metric
        and node.metric.value is not None
    ]
    if len(scored_nodes) <= window_size:
        return False

    historical = scored_nodes[:-window_size]
    recent = scored_nodes[-window_size:]
    maximize = True if agent.metric_maximize is None else bool(agent.metric_maximize)
    historical_best = (
        max(node.metric.value for node in historical)
        if maximize
        else min(node.metric.value for node in historical)
    )
    recent_best = (
        max(node.metric.value for node in recent)
        if maximize
        else min(node.metric.value for node in recent)
    )
    improvement = (
        recent_best - historical_best
        if maximize
        else historical_best - recent_best
    )
    if improvement > agent.scfg.metric_improvement_threshold:
        return False

    logger.info(f"Global stagnation detected: no improvement beyond threshold in last {window_size} nodes")
    return True
