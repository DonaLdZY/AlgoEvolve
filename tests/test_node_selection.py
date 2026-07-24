from __future__ import annotations

from types import SimpleNamespace

from engine import node_selection
from engine.search_node import Journal, SearchNode


def test_fully_expanded_root_with_only_locked_children_returns_without_spinning(
    monkeypatch,
) -> None:
    children = [
        SimpleNamespace(id=f"draft-{index}", lock=True, stage="draft")
        for index in range(8)
    ]
    root = SimpleNamespace(
        id="root",
        stage="root",
        children=children,
        is_terminal=False,
        reached_child_limit=lambda scfg: True,
    )
    agent = SimpleNamespace(
        scfg=SimpleNamespace(),
        acfg=SimpleNamespace(branch_fusion_trigger_prob=0.0),
        is_root=lambda node: node is root,
    )
    monkeypatch.setattr(node_selection, "should_trigger_branch_fusion", lambda _agent: False)

    assert node_selection.select(agent, root) is None


def test_equal_unvisited_root_children_do_not_use_insertion_order(monkeypatch) -> None:
    root = SearchNode(code="", plan="root", stage="root", id="root")
    first = SearchNode(code="", plan="first", stage="draft", parent=root, id="first")
    second = SearchNode(code="", plan="second", stage="draft", parent=root, id="second")
    root.children = {first, second}
    selected_ties: list[list[str]] = []

    def choose_last(items):
        selected_ties.append([item.id for item in items])
        return items[-1]

    monkeypatch.setattr(node_selection.random, "choice", choose_last)
    agent = SimpleNamespace(
        current_step=0,
        cfg=SimpleNamespace(
            agent=SimpleNamespace(
                decay=SimpleNamespace(
                    exploration_constant=1.414,
                    phase_ratios=[0.3, 0.7],
                    alpha=0.01,
                    lower_bound=0.5,
                )
            )
        ),
        scfg=SimpleNamespace(num_drafts=8, num_improves=5),
        acfg=SimpleNamespace(steps=50),
        journal=Journal([root, first, second]),
        is_root=lambda node: node is root,
    )

    selected = node_selection._best_child(agent, root)

    assert selected in {first, second}
    assert set(selected_ties[0]) == {"first", "second"}


def test_equal_unvisited_children_prefer_less_expanded_subtree() -> None:
    root = SearchNode(code="", plan="root", stage="root", id="root")
    deep = SearchNode(code="", plan="deep", stage="draft", parent=root, id="deep")
    shallow = SearchNode(
        code="", plan="shallow", stage="draft", parent=root, id="shallow"
    )
    descendant = SearchNode(
        code="", plan="descendant", stage="improve", parent=deep, id="descendant"
    )
    root.children = {deep, shallow}
    deep.children = {descendant}
    agent = SimpleNamespace(
        current_step=0,
        cfg=SimpleNamespace(
            agent=SimpleNamespace(
                decay=SimpleNamespace(
                    exploration_constant=1.414,
                    phase_ratios=[0.3, 0.7],
                    alpha=0.01,
                    lower_bound=0.5,
                )
            )
        ),
        scfg=SimpleNamespace(num_drafts=8, num_improves=5),
        acfg=SimpleNamespace(steps=50),
        journal=Journal([root, deep, shallow, descendant]),
        is_root=lambda node: node is root,
    )

    assert node_selection._best_child(agent, root) is shallow


def test_unreviewed_pending_child_cannot_be_selected_for_expansion() -> None:
    root = SearchNode(code="", plan="root", stage="root", id="root")
    reviewed = SearchNode(
        code="reviewed",
        plan="reviewed",
        stage="draft",
        parent=root,
        id="reviewed",
    )
    pending = SearchNode(
        code="pending",
        plan="pending",
        stage="debug",
        parent=reviewed,
        id="pending",
    )
    pending.pending_execution = True
    agent = SimpleNamespace(
        journal=Journal([root, reviewed]),
        is_root=lambda node: node is root,
    )

    assert node_selection._best_child(agent, reviewed) is None


def test_locked_non_root_child_cannot_receive_parallel_expansions() -> None:
    root = SearchNode(code="", plan="root", stage="root", id="root")
    parent = SearchNode(
        code="parent",
        plan="parent",
        stage="draft",
        parent=root,
        id="parent",
    )
    child = SearchNode(
        code="child",
        plan="child",
        stage="debug",
        parent=parent,
        id="child",
    )
    child.lock = True
    agent = SimpleNamespace(
        journal=Journal([root, parent, child]),
        is_root=lambda node: node is root,
    )

    assert node_selection._best_child(agent, parent) is None


def test_locked_root_falls_through_to_an_available_reviewed_child() -> None:
    root = SearchNode(code="", plan="root", stage="root", id="root")
    child = SearchNode(
        code="child",
        plan="child",
        stage="draft",
        parent=root,
        id="child",
        is_buggy=False,
    )
    root.lock = True
    agent = SimpleNamespace(
        current_step=1,
        cfg=SimpleNamespace(
            agent=SimpleNamespace(
                decay=SimpleNamespace(
                    exploration_constant=1.414,
                    phase_ratios=[0.3, 0.7],
                    alpha=0.01,
                    lower_bound=0.5,
                )
            )
        ),
        journal=Journal([root, child]),
        scfg=SimpleNamespace(num_drafts=8, num_improves=5),
        acfg=SimpleNamespace(steps=50, initial_drafts=3),
        is_root=lambda node: node is root,
    )

    assert node_selection.select(agent, root) is child


def test_buggy_parent_at_debug_limit_is_not_expanded_again(monkeypatch) -> None:
    root = SearchNode(code="", plan="root", stage="root", id="root")
    buggy = SearchNode(
        code="buggy",
        plan="buggy",
        stage="improve",
        parent=root,
        id="buggy",
        is_buggy=True,
    )
    buggy.expected_child_count = 1
    agent = SimpleNamespace(
        journal=Journal([root, buggy]),
        scfg=SimpleNamespace(num_bugs=1),
        acfg=SimpleNamespace(branch_fusion_trigger_prob=0.0),
        is_root=lambda node: node is root,
    )
    monkeypatch.setattr(node_selection, "should_trigger_branch_fusion", lambda _agent: False)

    assert node_selection.select(agent, buggy) is None
