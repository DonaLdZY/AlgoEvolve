from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

from agents.triggers import register_node
from engine import agent_search
from engine.agent_search import AgentSearch
from engine.search_node import Journal, SearchNode
from utils.metric import MetricValue


def test_parallel_generation_assigns_unique_branch_ids_and_children() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    agent = SimpleNamespace(
        tree_state_lock=threading.RLock(),
        next_branch_id=1,
        branch_all_nodes={},
        branch_successful_nodes={},
        _serialize_prompt=lambda prompt: str(prompt),
    )

    def generate(index: int) -> int:
        node = SearchNode(
            code=f"print({index})",
            plan=f"draft {index}",
            stage="draft",
            parent=root,
        )
        register_node(agent, node, "prompt", new_branch=True)
        return node.branch_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        branch_ids = list(pool.map(generate, range(20)))

    assert sorted(branch_ids) == list(range(1, 21))
    assert len(root.children) == 20
    assert set(agent.branch_all_nodes) == set(range(1, 21))


def test_follow_up_search_reselects_from_global_root() -> None:
    run_source = (Path(__file__).resolve().parents[1] / "run.py").read_text(
        encoding="utf-8"
    )

    assert "def schedule_new_step(executor)" in run_source
    assert "parent = agent.select_parent(None)" in run_source
    assert "reservation = ExpansionReservation(parent, agent.tree_state_lock)" in run_source
    assert "def fill_worker_slots()" in run_source
    assert "generation_executor" not in run_source
    assert "pending_execution_headroom" not in run_source


def test_each_worker_owns_one_complete_expansion_and_root_draft_mode() -> None:
    run_source = (Path(__file__).resolve().parents[1] / "run.py").read_text(
        encoding="utf-8"
    )

    assert "parallel closed-loop workers with virtual-visit UCT reservations" in run_source
    assert "executor.submit(\n            step_task" in run_source
    assert "execute_immediately=False" not in run_source
    assert "return_result_node=True" in run_source
    assert "draft_single_call=draft_single_call" in run_source
    assert "fast_draft_mode=fast_draft_mode" in run_source
    assert "parent.id not in journal_ids" in run_source
    assert "parent.id in active_parent_ids" in run_source


def test_interruption_freezes_late_search_results_and_preserves_actions() -> None:
    run_source = (Path(__file__).resolve().parents[1] / "run.py").read_text(
        encoding="utf-8"
    )

    assert "agent.accept_search_results = False" in run_source
    assert "search_state.close(clear_actions=False)" in run_source
    assert 'status="interrupted_resumable"' in run_source


def test_deferred_code_execution_releases_before_result_review(monkeypatch) -> None:
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(
        code="print('candidate')",
        plan="candidate",
        stage="draft",
        parent=root,
        metric=MetricValue(None, maximize=False),
    )
    node.pending_execution = True
    agent = AgentSearch.__new__(AgentSearch)
    agent.runtime_checkpoint_callback = None
    agent.accept_search_results = True
    agent.best_node = None
    agent.journal = Journal([root])
    agent.journal_lock = threading.Lock()
    reviewed: list[object] = []

    def fail_if_reviewed(*_args, **_kwargs):
        raise AssertionError("result review must not run in the code execution worker")

    monkeypatch.setattr(agent_search.result_parse_agent, "run", fail_if_reviewed)
    exec_result = object()

    returned = agent.execute_deferred_code(
        node,
        lambda *_args, **_kwargs: exec_result,
        runtime_action_id="action",
    )

    assert returned is exec_result
    assert node.pending_execution is True

    def review(_agent, *, node, exec_result):
        reviewed.append(exec_result)
        node.metric = MetricValue(3.0, maximize=False)
        node.is_buggy = False
        node.is_valid = True
        return node

    monkeypatch.setattr(agent_search.result_parse_agent, "run", review)
    monkeypatch.setattr(agent_search.execution, "validate_executed_node", lambda *_args: None)
    monkeypatch.setattr(agent_search.evaluation, "check_improvement", lambda *_args: False)
    monkeypatch.setattr(agent_search.solution_manager, "update_best_solution", lambda *_args: None)

    finalized = agent.finalize_deferred_node(node, returned)

    assert reviewed == [exec_result]
    assert finalized.pending_execution is False
    assert finalized in agent.journal.nodes
