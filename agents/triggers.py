import logging

from engine.search_node import SearchNode

logger = logging.getLogger("AlgoEvolve")


def get_patience_counter(agent, parent_node: SearchNode) -> tuple:
    if not hasattr(parent_node, 'branch_id') or parent_node.branch_id is None:
        return 0, 0, None

    branch_successful_nodes = agent.branch_successful_nodes.get(parent_node.branch_id, [])
    branch_all_nodes = agent.branch_all_nodes.get(parent_node.branch_id, [])

    if len(branch_successful_nodes) == 0:
        return 0, len(branch_all_nodes), None

    valid_nodes = [n for n in branch_successful_nodes if n.metric and n.metric.value is not None]
    if not valid_nodes:
        return 0, len(branch_all_nodes), None

    best_node = max(valid_nodes, key=lambda n: n.metric)
    branch_best_score = best_node.metric.value

    try:
        best_idx_success = branch_successful_nodes.index(best_node)
    except ValueError:
        logger.warning(f"Best node {best_node.id} not found in branch_successful_nodes list")
        return 0, len(branch_all_nodes), branch_best_score

    try:
        best_idx_all = branch_all_nodes.index(best_node)
    except ValueError:
        logger.warning(f"Best node {best_node.id} not found in branch_all_nodes list")
        best_idx_all = 0

    success_patience = len(branch_successful_nodes) - best_idx_success - 1
    total_patience = len(branch_all_nodes) - best_idx_all - 1

    logger.info(
        f"🔥 Patience counters: success={success_patience}, total={total_patience} "
        f"(best at pos {best_idx_all+1}/{len(branch_all_nodes)} overall, "
        f"{best_idx_success+1}/{len(branch_successful_nodes)} successful, "
        f"metric={branch_best_score:.4f}, id={best_node.id[:8]})"
    )

    return max(0, success_patience), max(0, total_patience), branch_best_score


def register_node(agent, node: SearchNode, prompt, parent_node=None, new_branch: bool = False):
    import time

    node.prompt_input = agent._serialize_prompt(prompt)
    node.created_time = time.strftime("%Y-%m-%dT%H:%M:%S")

    state_lock = getattr(agent, "tree_state_lock", None)

    def _register_branch_state() -> None:
        if new_branch:
            node.branch_id = agent.next_branch_id
            agent.next_branch_id += 1
            agent.branch_all_nodes[node.branch_id] = [node]
            agent.branch_successful_nodes[node.branch_id] = []
        else:
            node.branch_id = parent_node.branch_id
            if node.branch_id in agent.branch_all_nodes:
                agent.branch_all_nodes[node.branch_id].append(node)

    if state_lock is None:
        _register_branch_state()
    else:
        with state_lock:
            _register_branch_state()
