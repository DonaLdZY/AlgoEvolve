from types import SimpleNamespace

import pytest

from engine import evaluation
from engine.journal_snapshot import JournalSnapshot
from engine.search_node import Journal, SearchNode
from utils.metric import MetricValue


def _node(
    node_id: str,
    stage: str,
    *,
    parent: SearchNode | None = None,
    metric: float | None = None,
    buggy: bool | None = False,
) -> SearchNode:
    return SearchNode(
        code="",
        plan=node_id,
        stage=stage,
        parent=parent,
        id=node_id,
        metric=MetricValue(metric, maximize=False),
        is_buggy=buggy,
    )


def test_virtual_visit_changes_uct_before_review() -> None:
    root = _node("root", "root")
    child = _node("child", "draft", parent=root, metric=10.0)
    root.visits = 10
    child.visits = 2
    child.total_reward = 2.0
    before = child.uct_value()
    persisted_before = child._uct

    reservation = evaluation.ExpansionReservation(child)

    assert root.virtual_visits == 1
    assert child.virtual_visits == 1
    assert child.uct_value() < before
    assert child._uct == persisted_before
    reservation.cancel()
    assert root.virtual_visits == 0
    assert child.virtual_visits == 0


def test_review_converts_virtual_visit_to_real_reward_path() -> None:
    root = _node("root", "root")
    parent = _node("parent", "draft", parent=root, metric=10.0)
    result = _node("result", "improve", parent=parent, metric=9.0)
    reservation = evaluation.ExpansionReservation(parent)

    assert reservation.settle_result(result, 2.5) is True
    assert reservation.active is False
    for node in (root, parent, result):
        assert node.visits == 1
        assert node.total_reward == pytest.approx(2.5)
        assert getattr(node, "virtual_visits", 0) == 0
    assert reservation.settle_failure() is False


def test_failed_expansion_is_one_negative_completed_visit() -> None:
    root = _node("root", "root")
    parent = _node("parent", "draft", parent=root, metric=10.0)
    reservation = evaluation.ExpansionReservation(parent)

    reservation.settle_failure(-1.0)

    for node in (root, parent):
        assert node.visits == 1
        assert node.total_reward == pytest.approx(-1.0)
        assert getattr(node, "virtual_visits", 0) == 0


def test_virtual_visits_are_not_serialized_into_checkpoint_journal() -> None:
    root = _node("root", "root")
    child = _node("child", "draft", parent=root, metric=10.0)
    reservation = evaluation.ExpansionReservation(child)

    payload = JournalSnapshot.from_journal(Journal([root, child])).to_payload()

    assert all("virtual_visits" not in node for node in payload["nodes"])
    reservation.cancel()


def test_every_review_immediately_commits_reward_without_ending_chain() -> None:
    root = _node("root", "root")
    parent = _node("parent", "draft", parent=root, metric=10.0)
    result = _node("result", "improve", parent=parent, metric=9.0)
    result.local_best_node = parent
    reservation = evaluation.ExpansionReservation(parent)
    result._expansion_reservation = reservation
    agent = SimpleNamespace(
        search_start_time=None,
        metric_maximize=False,
        best_metric=10.0,
        best_node=parent,
        current_node_list=[],
        scfg=SimpleNamespace(
            metric_improvement_threshold=0.0001,
            max_improve_failure=3,
            back_debug_depth=3,
            max_debug_depth=20,
        ),
    )

    ended = evaluation.check_improvement(agent, result, parent)

    assert ended is False
    assert result.continue_improve is True
    assert result in agent.current_node_list
    for node in (root, parent, result):
        assert node.visits == 1
        assert node.total_reward == pytest.approx(2.5)
