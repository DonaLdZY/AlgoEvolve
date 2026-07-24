from __future__ import annotations

import json
import random
import time
from pathlib import Path

from engine.agent_search import AgentSearch
from engine.search_node import Journal, SearchNode
from engine.search_runtime_state import (
    ActiveSearchAction,
    SearchRuntimeState,
    SearchRuntimeStateStore,
    load_search_runtime_state,
    prune_completed_actions,
    reconcile_runtime_locks,
    repair_journal_for_resume,
    retain_generated_actions,
    retain_one_action_per_parent,
    restore_generated_node,
)
from utils.metric import MetricValue


def _journal() -> tuple[Journal, SearchNode, SearchNode]:
    root = SearchNode(
        code="",
        stage="root",
        step=0,
        metric=MetricValue(None, maximize=False),
        id="root",
    )
    parent = SearchNode(
        code="parent",
        stage="draft",
        parent=root,
        metric=MetricValue(10, maximize=False),
        is_buggy=False,
        is_valid=True,
        id="parent",
        branch_id=1,
    )
    return Journal([root, parent]), root, parent


def test_search_state_round_trip_preserves_generated_node_rng_and_elapsed(tmp_path: Path) -> None:
    path = tmp_path / "search_state.json"
    journal, _, parent = _journal()
    generated = SearchNode(
        code="generated",
        stage="improve",
        parent=parent,
        local_best_node=parent,
        id="generated",
        branch_id=1,
    )

    random.seed(12345)
    store = SearchRuntimeStateStore(path, checkpoint_seconds=60)
    action = store.register_action(
        parent.id,
        action_id="action-1",
        parent_from_topk=True,
    )
    store.update_action(action.action_id, status="generated", generated_node=generated)
    saved_random_state = random.getstate()
    store.checkpoint()

    payload = json.loads(path.read_text(encoding="utf-8"))
    restored = load_search_runtime_state(path)

    assert "config" not in payload
    assert restored.cumulative_search_elapsed_seconds >= 0
    assert len(restored.active_actions) == 1
    restored_action = restored.active_actions[0]
    assert restored_action.parent_node_id == parent.id
    assert restored_action.status == "generated"
    assert restored_action.parent_from_topk is True
    assert restored_action.generated_node is not None
    assert restored_action.generated_node.id == generated.id
    assert restored_action.generated_node.code == "generated"

    random.seed(999)
    restored_store = SearchRuntimeStateStore(path, initial_state=restored)
    assert restored_store.restore_random_state() is True
    assert random.getstate() == saved_random_state


def test_repair_reserves_active_children_and_clears_stale_locks() -> None:
    journal, root, parent = _journal()
    root.lock = True
    parent.lock = True
    root.expected_child_count = 99
    parent.expected_child_count = 99
    actions = [
        ActiveSearchAction(action_id="a1", parent_node_id=parent.id),
        ActiveSearchAction(action_id="a2", parent_node_id=parent.id),
    ]

    repair_journal_for_resume(journal, actions)

    assert root.lock is False
    assert root.expected_child_count == 1
    assert parent.lock is True
    assert parent.expected_child_count == 2


def test_live_lock_reconciliation_tracks_only_active_action_parents() -> None:
    journal, root, parent = _journal()
    root.lock = True
    parent.lock = True

    active_ids = reconcile_runtime_locks(
        journal.nodes,
        [ActiveSearchAction(action_id="a1", parent_node_id=parent.id)],
    )

    assert active_ids == {parent.id}
    assert root.lock is False
    assert parent.lock is True

    reconcile_runtime_locks(journal.nodes, [])
    assert root.lock is False
    assert parent.lock is False


def test_restore_generated_node_reuses_same_id_code_and_parent() -> None:
    journal, _, parent = _journal()
    generated = SearchNode(
        code="exact interrupted code",
        stage="improve",
        parent=parent,
        local_best_node=parent,
        id="generated-child",
        branch_id=1,
    )
    store_state = SearchRuntimeStateStore(Path("unused-search-state.json"), enabled=False)
    action = store_state.register_action(parent.id, action_id="a1")
    updated = store_state.update_action(
        action.action_id,
        status="executing",
        generated_node=generated,
    )
    assert updated is not None

    repair_journal_for_resume(journal, [updated])
    restored_node = restore_generated_node(updated, journal)

    assert restored_node is not None
    assert restored_node.id == "generated-child"
    assert restored_node.code == "exact interrupted code"
    assert restored_node.parent is parent
    assert restored_node in parent.children
    assert restored_node.pending_execution is True
    assert restored_node.runtime_action_id == "a1"
    assert parent.expected_child_count == len(parent.children)


def test_completed_generated_action_is_pruned() -> None:
    journal, _, parent = _journal()
    generated = SearchNode(
        code="done",
        stage="improve",
        parent=parent,
        id="already-in-journal",
    )
    journal.append(generated)
    store = SearchRuntimeStateStore(Path("unused-search-state.json"), enabled=False)
    action = store.register_action(parent.id, action_id="a1")
    action = store.update_action("a1", status="generated", generated_node=generated)
    assert action is not None

    assert prune_completed_actions([action], {node.id for node in journal.nodes}) == []


def test_interrupt_retains_only_actions_with_generated_code(tmp_path: Path) -> None:
    journal, _, parent = _journal()
    generated = SearchNode(
        code="resume this exact code",
        stage="improve",
        parent=parent,
        id="generated-child",
    )
    state_path = tmp_path / "search_state.json"
    store = SearchRuntimeStateStore(state_path, checkpoint_seconds=60)
    scheduled = store.register_action(parent.id, action_id="scheduled")
    materialized = store.register_action(parent.id, action_id="materialized")
    materialized = store.update_action(
        materialized.action_id,
        status="executing",
        generated_node=generated,
    )
    assert materialized is not None

    retained = retain_generated_actions([scheduled, materialized])
    store.close(clear_actions=False, replacement_actions=retained)
    repair_journal_for_resume(journal, retained)
    restored = load_search_runtime_state(state_path)

    assert [action.action_id for action in retained] == ["materialized"]
    assert [action.action_id for action in restored.active_actions] == ["materialized"]
    assert retained[0].generated_node is not None
    assert retained[0].generated_node.code == "resume this exact code"
    assert parent.lock is True
    assert parent.expected_child_count == len(parent.children) + 1


def test_resume_keeps_only_one_generated_action_per_parent() -> None:
    _, _, parent = _journal()
    store = SearchRuntimeStateStore(Path("unused-search-state.json"), enabled=False)
    first = store.register_action(parent.id, action_id="first")
    second = store.register_action(parent.id, action_id="second")
    first_node = SearchNode(code="first code", stage="debug", parent=parent, id="first-node")
    second_node = SearchNode(code="second code", stage="debug", parent=parent, id="second-node")
    first = store.update_action(first.action_id, status="generated", generated_node=first_node)
    second = store.update_action(second.action_id, status="generated", generated_node=second_node)
    assert first is not None and second is not None

    retained = retain_one_action_per_parent([first, second])

    assert [action.action_id for action in retained] == ["first"]


def test_agent_search_elapsed_resume_does_not_restart_at_zero() -> None:
    agent = AgentSearch.__new__(AgentSearch)
    before = time.time()
    agent.restore_search_elapsed(321.5)
    elapsed = time.time() - agent.search_start_time

    assert elapsed >= 321.5
    assert agent.search_start_time <= before - 321.4


def test_runtime_state_uses_new_config_outside_snapshot() -> None:
    state = SearchRuntimeState(cumulative_search_elapsed_seconds=100)
    payload = state.to_payload()

    new_config = {"steps": 80, "parallel_search_num": 6, "num_improves": 9}
    assert "config" not in payload
    assert new_config == {"steps": 80, "parallel_search_num": 6, "num_improves": 9}
