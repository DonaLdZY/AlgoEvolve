import contextvars
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from config import save_run
from engine.journal_snapshot import JournalSnapshot
from engine.search_node import Journal, SearchNode, filter_on_path
from utils.metric import MetricValue
from utils.serialize import dumps_json, loads_json


def _journal() -> tuple[Journal, SearchNode, SearchNode, SearchNode]:
    root = SearchNode(
        code="root",
        stage="root",
        metric=MetricValue(100, maximize=False),
        is_buggy=False,
        branch_id=2,
        visits=4,
        total_reward=7.5,
        _uct=2.25,
    )
    best = SearchNode(
        code="best",
        stage="improve",
        parent=root,
        metric=MetricValue(10, maximize=False),
        is_buggy=False,
        is_valid=True,
        runtime_ok=True,
        search_eligible=True,
        contract_valid=True,
        artifact_ready=True,
        delivery_ready=True,
        delivery_certified=False,
        certification_source="candidate_reported_score",
        certification_notes=["Independent evaluator not available."],
        method_mode="non_rl_solver",
        expected_child_count=3,
    )
    other = SearchNode(
        code="other",
        stage="debug",
        parent=root,
        metric=MetricValue(50, maximize=False),
        is_buggy=True,
    )
    root.local_best_node = best
    return Journal([root, best, other]), root, best, other


def test_snapshot_ignores_runtime_only_dynamic_attributes() -> None:
    journal, root, _, _ = _journal()
    root.runtime_thread = threading.Thread(target=lambda: None)
    root.runtime_lock = threading.Lock()
    root.runtime_callback = lambda: None
    root.runtime_context = contextvars.copy_context()

    payload = json.loads(dumps_json(journal))
    serialized = json.dumps(payload)

    assert payload["__version"] == "3"
    assert "runtime_thread" not in serialized
    assert "runtime_lock" not in serialized
    assert "runtime_callback" not in serialized
    assert "runtime_context" not in serialized


def test_snapshot_round_trip_rebuilds_graph_metrics_and_locks() -> None:
    journal, root, best, _ = _journal()
    root_lock = root.child_count_lock

    restored = loads_json(dumps_json(journal), Journal)
    restored_by_id = {node.id: node for node in restored.nodes}
    restored_root = restored_by_id[root.id]
    restored_best = restored_by_id[best.id]

    assert restored_best.parent is restored_root
    assert restored_best in restored_root.children
    assert restored_root.local_best_node is restored_best
    assert restored_best.metric.value == 10.0
    assert restored_best.metric.maximize is False
    assert restored_root.branch_id == 2
    assert restored_root.visits == 4
    assert restored_root.total_reward == 7.5
    assert restored_root._uct == 2.25
    assert restored_best.expected_child_count == 3
    assert restored_best.search_eligible is True
    assert restored_best.delivery_ready is True
    assert restored_best.delivery_certified is False
    assert restored_best.method_mode == "non_rl_solver"
    assert restored_best.certification_notes == ["Independent evaluator not available."]
    assert restored_root.child_count_lock is not root_lock
    assert restored_root.child_count_lock is not restored_best.child_count_lock
    assert hasattr(restored_root.child_count_lock, "acquire")


def test_filtered_snapshot_omits_execution_details_without_mutating_source() -> None:
    journal, root, best, other = _journal()
    root._term_out = ["full stdout"]
    root.exc_stack = [("trace.py", 10, "main")]

    filtered = filter_on_path(journal, [root.id, best.id])

    assert [node.id for node in filtered.nodes] == [root.id, best.id]
    assert other.id not in {node.id for node in filtered.nodes}
    assert filtered.nodes[0]._term_out == "<OMITTED>"
    assert filtered.nodes[0].exc_stack == "<OMITTED>"
    assert filtered.nodes[1].parent is filtered.nodes[0]
    assert root._term_out == ["full stdout"]
    assert root.exc_stack == [("trace.py", 10, "main")]


def test_version_two_payload_remains_loadable() -> None:
    journal, root, best, _ = _journal()
    version_three = JournalSnapshot.from_journal(journal).to_payload()
    legacy_payload = dict(version_three)
    legacy_payload["__version"] = "2"

    restored = loads_json(json.dumps(legacy_payload), Journal)
    restored_by_id = {node.id: node for node in restored.nodes}

    assert restored_by_id[best.id].parent is restored_by_id[root.id]
    assert restored_by_id[root.id].local_best_node is restored_by_id[best.id]
    assert restored_by_id[best.id].metric.value == 10.0
    assert hasattr(restored_by_id[root.id].child_count_lock, "acquire")


def test_save_run_succeeds_while_runtime_thread_is_pending(tmp_path: Path) -> None:
    journal, root, _, _ = _journal()
    release = threading.Event()
    pending_thread = threading.Thread(target=release.wait, daemon=True)
    pending_thread.start()
    root.pending_insight_thread = pending_thread
    cfg = SimpleNamespace(
        log_dir=tmp_path,
        runtime=SimpleNamespace(
            save_journal=True,
            save_filtered_journal=True,
            save_resolved_config=False,
            save_best_solution=False,
        ),
    )

    try:
        save_run(cfg, journal)
    finally:
        release.set()
        pending_thread.join(timeout=2)

    full_payload = json.loads((tmp_path / "journal.json").read_text(encoding="utf-8"))
    filtered_payload = json.loads(
        (tmp_path / "filtered_journal.json").read_text(encoding="utf-8")
    )
    assert full_payload["__version"] == "3"
    assert filtered_payload["__version"] == "3"
    assert "pending_insight_thread" not in json.dumps(full_payload)
