import pytest

from engine.search_node import SearchNode
from utils.metric import MetricValue


def child(root, plan, score, maximize, **kwargs):
    return SearchNode(code="print('candidate')", plan=plan, stage="draft", parent=root,
                      metric=MetricValue(score, maximize=maximize), is_buggy=False, **kwargs)


@pytest.mark.parametrize("maximize,best,worst", [(False, 1.0, 9.0), (True, 9.0, 1.0)])
def test_memory_orders_by_quality_and_reports_the_same_best(maximize, best, worst):
    root = SearchNode(code="", stage="root")
    child(root, "worse design", worst, maximize)
    child(root, "better design", best, maximize)
    memory = root.fetch_child_memory()
    assert memory.index("Design: better design") < memory.index("Design: worse design")
    assert f"best: {best:.4f}" in memory
    assert ("maximize" if maximize else "minimize") in memory


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_score_does_not_crash_or_become_best(score):
    root = SearchNode(code="", stage="root")
    child(root, "not scored", score, False)
    memory = root.fetch_child_memory()
    assert "no comparable finite metric" in memory
    assert "best:" not in memory


def test_unknown_or_mixed_directions_are_not_guessed():
    root = SearchNode(code="", stage="root")
    child(root, "unknown", 0, None)
    child(root, "lower", 1, False)
    child(root, "higher", 9, True)
    assert "best:" not in root.fetch_child_memory()


def test_parent_direction_excludes_mismatched_children_and_ties_are_stable():
    root = SearchNode(code="", stage="root", metric=MetricValue(10, maximize=False))
    child(root, "same score B", 1, False, ctime=10, id="b")
    child(root, "same score A", 1, False, ctime=10, id="a")
    child(root, "wrong direction", 100, True)
    child(root, "missing score", None, False)
    memory = root.fetch_child_memory()
    assert memory == root.fetch_child_memory()
    assert "best: 1.0000; minimize" in memory
    assert memory.index("Design: same score") < memory.index("Design: wrong direction")
