from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf
import service_api

from service_api import (
    JobStore,
    StartAlgoEvolveRequest,
    _base_config_value,
    _inspect_interrupted_checkpoint,
    _mark_service_budget_completed,
    _native_exit_reason,
    _resolve_best_node,
    _resolve_interrupted_checkpoint_layout,
    _tail_text,
    _with_cli_override,
)


def test_service_reads_runtime_defaults_from_default_yaml() -> None:
    assert _base_config_value("runtime.job_status_tail_chars", None) == 60000
    assert _base_config_value("runtime.service_log_tail_chars", None) == 200000
    assert _base_config_value("runtime.termination_wait_seconds", None) == 20
    assert _base_config_value("runtime.save_search_state", None) is True
    assert _base_config_value("runtime.search_state_filename", None) == "search_state.json"
    assert _base_config_value("runtime.restore_inflight_actions", None) is True
    assert _base_config_value("runtime.interruption_checkpoint_wait_seconds", None) == 30
    assert _base_config_value("runtime.exit_immediately_after_interrupt_checkpoint", None) is True
    assert _base_config_value("runtime.service_startup_buffer_seconds", None) == 1800


def test_job_status_tail_and_request_default_are_config_driven() -> None:
    store = JobStore()
    job = store.create("task", "logs", "workspace")
    store.update(job.job_id, stdout_tail="abcdefgh", job_status_tail_chars=4)

    assert store.status(job.job_id).stdout_tail == "efgh"
    assert _tail_text("abcdefgh", 0) == ""
    request = StartAlgoEvolveRequest(
        task_id="task",
        log_dir="logs",
        workspace_dir="workspace",
        config_path="other/task-config.yaml",
    )
    assert request.graceful_shutdown_buffer_secs is None
    assert request.config_path == "other/task-config.yaml"
    assert request.resources is None


def test_windows_stack_overflow_exit_code_has_native_diagnostic() -> None:
    expected = "Windows STATUS_STACK_OVERFLOW (0xC00000FD)"
    assert _native_exit_reason(3221225725) == expected
    assert _native_exit_reason(-1073741571) == expected
    assert _native_exit_reason(1) is None


def test_service_run_timestamp_override_stays_a_string_in_omegaconf() -> None:
    timestamp = "20260722_015005"
    args = _with_cli_override(
        [],
        "runtime.run_timestamp",
        json.dumps(timestamp),
    )

    parsed = OmegaConf.from_dotlist(args)

    assert parsed.runtime.run_timestamp == timestamp
    assert isinstance(parsed.runtime.run_timestamp, str)


def test_stop_endpoint_returns_while_checkpointing_continues(monkeypatch) -> None:
    class FakeProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

    started: list[tuple[object, tuple[object, ...]]] = []

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            assert daemon is True
            self.target = target
            self.args = args

        def start(self):
            started.append((self.target, self.args))

    job = service_api.store.create("stop-test", "logs", "workspace")
    service_api.store.set_process(job.job_id, FakeProcess())
    monkeypatch.setattr(service_api.threading, "Thread", FakeThread)

    response = service_api.stop_job(service_api.StopRequest(job_id=job.job_id))

    assert response["status"] == "stopping"
    assert service_api.store.get(job.job_id).stop_requested is True
    assert started == [
        (
            service_api._stop_process_after_checkpoint_window,
            (
                service_api.store.get(job.job_id).process,
                30.0,
                Path("logs"),
                Path("workspace"),
            ),
        )
    ]


def test_stop_worker_kills_process_tree_as_soon_as_new_checkpoint_is_ready(
    monkeypatch,
) -> None:
    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.signals: list[object] = []

        @staticmethod
        def poll():
            return None

        def send_signal(self, value):
            self.signals.append(value)

    process = FakeProcess()
    markers = iter([("old",), ("new",)])
    terminated: list[int] = []
    monkeypatch.setattr(service_api.os, "name", "nt")
    monkeypatch.setattr(
        service_api,
        "_interrupted_checkpoint_marker",
        lambda *_args: next(markers),
    )
    monkeypatch.setattr(
        service_api,
        "_inspect_interrupted_checkpoint",
        lambda *_args: (True, "manifest", ""),
    )
    monkeypatch.setattr(
        service_api,
        "terminate_process_tree",
        lambda pid: terminated.append(pid),
    )

    service_api._stop_process_after_checkpoint_window(
        process,
        30.0,
        Path("logs"),
        Path("workspace"),
    )

    assert process.signals == [service_api.signal.CTRL_BREAK_EVENT]
    assert terminated == [12345]


def test_interrupted_checkpoint_requires_both_manifests_and_resume_state(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    workspace_dir = tmp_path / "workspace"
    log_dir.mkdir()
    workspace_dir.mkdir()
    manifest = {
        "status": "interrupted_resumable",
        "resumable": True,
        "active_actions": [{"action_id": "a1"}],
    }
    (log_dir / "journal.json").write_text("{}", encoding="utf-8")
    (log_dir / "search_state.json").write_text("{}", encoding="utf-8")
    (log_dir / "run_status.json").write_text(
        json.dumps({"status": "interrupted_resumable"}),
        encoding="utf-8",
    )
    (log_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    ready, _, error = _inspect_interrupted_checkpoint(log_dir, workspace_dir)
    assert ready is False
    assert "workspace" in error

    (workspace_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    ready, manifest_path, error = _inspect_interrupted_checkpoint(log_dir, workspace_dir)
    assert ready is True
    assert manifest_path == str(log_dir / "checkpoint_manifest.json")
    assert error == ""


def test_interrupted_checkpoint_finds_legacy_numeric_timestamp_directory(tmp_path: Path) -> None:
    expected_log = tmp_path / "logs" / "20260722_015005_demo"
    expected_workspace = tmp_path / "workspaces" / "20260722_015005_demo"
    actual_log = tmp_path / "logs" / "20260722015005_demo"
    actual_workspace = tmp_path / "workspaces" / "20260722015005_demo"
    actual_log.mkdir(parents=True)
    actual_workspace.mkdir(parents=True)
    manifest = {"status": "interrupted_resumable", "resumable": True}
    for path in (actual_log / "journal.json", actual_log / "search_state.json"):
        path.write_text("{}", encoding="utf-8")
    (actual_log / "run_status.json").write_text(
        json.dumps({"status": "interrupted_resumable"}),
        encoding="utf-8",
    )
    for path in (
        actual_log / "checkpoint_manifest.json",
        actual_workspace / "checkpoint_manifest.json",
    ):
        path.write_text(json.dumps(manifest), encoding="utf-8")

    resolved_log, resolved_workspace = _resolve_interrupted_checkpoint_layout(
        expected_log,
        expected_workspace,
    )
    ready, manifest_path, error = _inspect_interrupted_checkpoint(
        expected_log,
        expected_workspace,
    )

    assert resolved_log == actual_log
    assert resolved_workspace == actual_workspace
    assert ready is True
    assert manifest_path == str(actual_log / "checkpoint_manifest.json")
    assert error == ""


def test_interrupted_checkpoint_rejects_non_resumable_manifest(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    workspace_dir = tmp_path / "workspace"
    log_dir.mkdir()
    workspace_dir.mkdir()
    for path in (log_dir / "journal.json", log_dir / "search_state.json"):
        path.write_text("{}", encoding="utf-8")
    (log_dir / "run_status.json").write_text(
        json.dumps({"status": "interrupted_resumable"}),
        encoding="utf-8",
    )
    bad_manifest = {"status": "stopped", "resumable": False}
    for path in (
        log_dir / "checkpoint_manifest.json",
        workspace_dir / "checkpoint_manifest.json",
    ):
        path.write_text(json.dumps(bad_manifest), encoding="utf-8")

    ready, _, error = _inspect_interrupted_checkpoint(log_dir, workspace_dir)
    assert ready is False
    assert "not committed as resumable" in error


def test_service_watchdog_normalizes_saved_run_to_completed(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "interrupted_resumable",
                "termination_reason": "external_interrupt",
                "completed_steps": 12,
                "total_steps": 50,
            }
        ),
        encoding="utf-8",
    )

    _mark_service_budget_completed(log_dir, 10800)

    payload = json.loads((log_dir / "run_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["termination_reason"] == "time_limit_exhausted"
    assert payload["completed_steps"] == 12
    assert payload["service_forced_finalize"] is True


def test_best_node_uses_accepted_search_candidate_without_delivery_gate(tmp_path: Path) -> None:
    nodes = [
        {
            "id": "candidate-a",
            "metric": 12.0,
            "maximize": False,
            "is_buggy": False,
            "search_eligible": True,
            "delivery_ready": False,
        },
        {
            "id": "candidate-b",
            "metric": 8.0,
            "maximize": False,
            "is_buggy": False,
            "search_eligible": True,
            "delivery_ready": False,
        },
    ]

    node_id, kind = _resolve_best_node(tmp_path, tmp_path, nodes)

    assert node_id == "candidate-b"
    assert kind == "delivery"
