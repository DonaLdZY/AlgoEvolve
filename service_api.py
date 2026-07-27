from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.resource_limits import (
    accelerator_visibility_env,
    apply_process_tree_cpu_affinity,
    choose_cpu_ids,
    cpu_enforcement_capabilities,
    cpu_limit_environment,
    create_process_tree_memory_limiter,
    detect_resource_inventory,
    format_bytes,
    memory_enforcement_capabilities,
    process_tree_memory_bytes,
    relieve_process_tree_memory_pressure,
    terminate_process_tree,
    validate_accelerator_selection,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKDIR = str(ROOT_DIR)


def now_ts() -> float:
    return time.time()


def _is_interrupted_exit_code(exit_code: int | None) -> bool:
    # Windows CTRL_C_EVENT/CTRL_BREAK_EVENT is reported as 0xC000013A.
    return exit_code in {3221225786, -1073741510, 130, -2, -15}


def _native_exit_reason(exit_code: int | None) -> str | None:
    reasons = {
        3221225725: "Windows STATUS_STACK_OVERFLOW (0xC00000FD)",
        -1073741571: "Windows STATUS_STACK_OVERFLOW (0xC00000FD)",
    }
    return reasons.get(exit_code)


def _memory_child_guard_threshold(memory_limit_bytes: int, hard_limit_active: bool) -> int:
    """Leave allocation headroom for the controller before a hard process-tree cap."""
    if memory_limit_bytes <= 0 or not hard_limit_active or memory_limit_bytes < 1024**3:
        return memory_limit_bytes
    reserve = max(512 * 1024**2, min(2 * 1024**3, memory_limit_bytes // 10))
    return max(memory_limit_bytes // 2, memory_limit_bytes - reserve)


class TaskResourceLimits(BaseModel):
    cpu_cores: int = Field(default=4, ge=1, le=4096)
    memory_limit_gb: float = Field(default=8.0, ge=0, le=1048576)
    accelerator_mode: Literal["all", "selected", "none"] = "all"
    accelerator_device_ids: list[str] = Field(default_factory=list)
    monitor_interval_seconds: float = Field(default=0.5, ge=0.1, le=10.0)


class StartAlgoEvolveRequest(BaseModel):
    task_id: str
    python_executable: str = "python"
    working_dir: str = DEFAULT_WORKDIR
    env_overrides: dict[str, str] = Field(default_factory=dict)
    config_path: str = ""
    args: list[str] = Field(default_factory=list)
    log_dir: str
    workspace_dir: str
    resume: bool = False
    graceful_shutdown_buffer_secs: int | None = Field(default=None, ge=0, le=3600)
    resources: TaskResourceLimits | None = None


class StopRequest(BaseModel):
    job_id: str


class SnapshotRequest(BaseModel):
    log_dir: str = ""
    workspace_dir: str = ""
    run_dir: str = ""
    task_name: str = ""


# Import compatibility for integrations that still use the old class name.
StartMLEvolveRequest = StartAlgoEvolveRequest


class JobStatus(BaseModel):
    job_id: str
    task_id: str
    status: str
    started_at: float
    updated_at: float
    log_dir: str
    workspace_dir: str
    exit_code: int | None = None
    last_error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    resource_limits: dict[str, Any] = Field(default_factory=dict)
    assigned_cpu_ids: list[int] = Field(default_factory=list)
    cpu_enforcement: dict[str, Any] = Field(default_factory=dict)
    current_memory_bytes: int = 0
    peak_memory_bytes: int = 0
    memory_enforcement: dict[str, Any] = Field(default_factory=dict)
    resource_violation: str | None = None
    resource_warning: str | None = None
    checkpoint_ready: bool = False
    resumable: bool = False
    checkpoint_manifest_path: str | None = None


@dataclass
class JobRuntime:
    job_id: str
    task_id: str
    log_dir: str
    workspace_dir: str
    process: subprocess.Popen[str] | None
    status: str
    started_at: float
    updated_at: float
    exit_code: int | None = None
    last_error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    stop_requested: bool = False
    job_status_tail_chars: int = 60000
    stop_wait_seconds: float = 30.0
    resource_limits: dict[str, Any] = field(default_factory=dict)
    assigned_cpu_ids: list[int] = field(default_factory=list)
    cpu_enforcement: dict[str, Any] = field(default_factory=dict)
    current_memory_bytes: int = 0
    peak_memory_bytes: int = 0
    memory_enforcement: dict[str, Any] = field(default_factory=dict)
    resource_violation: str | None = None
    resource_warning: str | None = None
    checkpoint_ready: bool = False
    resumable: bool = False
    checkpoint_manifest_path: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRuntime] = {}

    def create(
        self,
        task_id: str,
        log_dir: str,
        workspace_dir: str,
        *,
        resource_limits: dict[str, Any] | None = None,
        assigned_cpu_ids: list[int] | None = None,
    ) -> JobRuntime:
        with self._lock:
            for job in self._jobs.values():
                if job.task_id != task_id or job.status not in {"pending", "running", "stopping"}:
                    continue
                proc = job.process
                if proc is not None and proc.poll() is not None:
                    job.status = "failed" if (proc.returncode or 0) != 0 else "completed"
                    job.exit_code = proc.returncode
                    job.updated_at = now_ts()
                    continue
                raise HTTPException(status_code=400, detail="task already running in AlgoEvolve service")
            job_id = uuid.uuid4().hex
            ts = now_ts()
            runtime = JobRuntime(
                job_id=job_id,
                task_id=task_id,
                log_dir=log_dir,
                workspace_dir=workspace_dir,
                process=None,
                status="pending",
                started_at=ts,
                updated_at=ts,
                resource_limits=dict(resource_limits or {}),
                assigned_cpu_ids=list(assigned_cpu_ids or []),
            )
            self._jobs[job_id] = runtime
            return runtime

    def _get_unlocked(self, job_id: str) -> JobRuntime:
        runtime = self._jobs.get(job_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="job not found")
        return runtime

    def get(self, job_id: str) -> JobRuntime:
        with self._lock:
            return self._get_unlocked(job_id)

    def set_process(self, job_id: str, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            runtime = self._get_unlocked(job_id)
            runtime.process = proc
            runtime.status = "running"
            runtime.updated_at = now_ts()

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            runtime = self._get_unlocked(job_id)
            for key, value in kwargs.items():
                setattr(runtime, key, value)
            runtime.updated_at = now_ts()

    def status(self, job_id: str) -> JobStatus:
        runtime = self.get(job_id)
        return JobStatus(
            job_id=runtime.job_id,
            task_id=runtime.task_id,
            status=runtime.status,
            started_at=runtime.started_at,
            updated_at=runtime.updated_at,
            log_dir=runtime.log_dir,
            workspace_dir=runtime.workspace_dir,
            exit_code=runtime.exit_code,
            last_error=runtime.last_error,
            stdout_tail=_tail_text(runtime.stdout_tail, runtime.job_status_tail_chars),
            stderr_tail=_tail_text(runtime.stderr_tail, runtime.job_status_tail_chars),
            resource_limits=dict(runtime.resource_limits),
            assigned_cpu_ids=list(runtime.assigned_cpu_ids),
            cpu_enforcement=dict(runtime.cpu_enforcement),
            current_memory_bytes=int(runtime.current_memory_bytes),
            peak_memory_bytes=int(runtime.peak_memory_bytes),
            memory_enforcement=dict(runtime.memory_enforcement),
            resource_violation=runtime.resource_violation,
            resource_warning=runtime.resource_warning,
            checkpoint_ready=runtime.checkpoint_ready,
            resumable=runtime.resumable,
            checkpoint_manifest_path=runtime.checkpoint_manifest_path,
        )


store = JobStore()
app = FastAPI(title="AlgoEvolve Service API", version="0.1.0")


def _tail_text(text: str, limit: int = 200000) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _unquote_cli_value(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        try:
            return json.loads(text)
        except Exception:
            return text[1:-1]
    return text


def _extract_cli_override(args: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for item in args:
        if isinstance(item, str) and item.startswith(prefix):
            return _unquote_cli_value(item[len(prefix) :])
    return None


def _with_cli_override(args: list[str], key: str, value: Any) -> list[str]:
    if _extract_cli_override(args, key) is not None:
        return args
    rendered = str(value).lower() if isinstance(value, bool) else str(value)
    return [*args, f"{key}={rendered}"]


def _saved_config_value(log_dir: Path, dotted_key: str, default: Any) -> Any:
    path = log_dir / "config.yaml"
    if not path.exists():
        return default
    try:
        current: Any = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    except Exception:
        return default


def _resolved_log_filename(
    log_dir: Path,
    dotted_key: str,
    default_name: str,
    legacy_name: str,
) -> str:
    configured = str(_saved_config_value(log_dir, dotted_key, default_name) or default_name)
    if (log_dir / configured).exists() or configured != default_name:
        return configured
    if (log_dir / legacy_name).exists():
        return legacy_name
    return configured


def _base_config_value(dotted_key: str, default: Any) -> Any:
    path = ROOT_DIR / "config" / "config.yaml"
    if not path.exists():
        return default
    try:
        current: Any = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    except Exception:
        return default


def _resolve_resource_limits(req: StartAlgoEvolveRequest) -> TaskResourceLimits:
    if req.resources is not None:
        return req.resources

    payload: dict[str, Any] = {}
    config_path = Path(req.config_path).expanduser() if req.config_path.strip() else None
    if config_path is not None:
        if not config_path.is_absolute():
            config_path = Path(req.working_dir.strip() or DEFAULT_WORKDIR) / config_path
        try:
            config_data = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
            resources = config_data.get("resources") if isinstance(config_data, dict) else {}
            if isinstance(resources, dict):
                payload.update(resources)
        except Exception:
            pass

    defaults = TaskResourceLimits()
    for field_name in TaskResourceLimits.model_fields:
        raw = _extract_cli_override(req.args, f"resources.{field_name}")
        if raw is not None:
            try:
                payload[field_name] = yaml.safe_load(raw)
            except Exception:
                payload[field_name] = raw
        elif field_name not in payload:
            payload[field_name] = _base_config_value(
                f"resources.{field_name}",
                getattr(defaults, field_name),
            )
    return TaskResourceLimits.model_validate(payload)


def _extract_time_limit_secs(args: list[str]) -> int | None:
    for key in ("agent.time_limit", "exec.timeout"):
        raw = _extract_cli_override(args, key)
        if raw is None:
            continue
        try:
            return max(1, int(float(raw)))
        except Exception:
            return None
    return None


def _resolve_run_layout(req: StartAlgoEvolveRequest, run_timestamp: str) -> tuple[Path, Path, str]:
    exp_name = (_extract_cli_override(req.args, "exp_name") or "").strip()
    if not exp_name and not req.resume:
        raise HTTPException(status_code=400, detail="missing exp_name in AlgoEvolve args")

    log_root = Path(req.log_dir).expanduser().resolve()
    workspace_root = Path(req.workspace_dir).expanduser().resolve()
    if req.resume:
        final_name = exp_name or log_root.name or workspace_root.name
        return log_root, workspace_root, final_name

    final_name = f"{run_timestamp}_{exp_name}"
    if log_root == workspace_root:
        per_run_root = (log_root / final_name).resolve()
        return (per_run_root / "logs").resolve(), (per_run_root / "workspace").resolve(), final_name
    return (log_root / final_name).resolve(), (workspace_root / final_name).resolve(), final_name


def _timestamp_path_variants(path: Path) -> list[Path]:
    """Return equivalent run paths for quoted and legacy numeric timestamps."""

    def alternate_name(name: str) -> str | None:
        quoted = re.match(r"^(\d{8})_(\d{6})(_.+)$", name)
        if quoted:
            return f"{quoted.group(1)}{quoted.group(2)}{quoted.group(3)}"
        legacy = re.match(r"^(\d{14})(_.+)$", name)
        if legacy:
            stamp = legacy.group(1)
            return f"{stamp[:8]}_{stamp[8:]}{legacy.group(2)}"
        return None

    variants = [path]
    direct_name = alternate_name(path.name)
    if direct_name:
        variants.append(path.with_name(direct_name))
    parent_name = alternate_name(path.parent.name)
    if parent_name:
        variants.append(path.parent.with_name(parent_name) / path.name)
    return variants


def _timestamp_run_identity(path: Path) -> str:
    for name in (path.name, path.parent.name):
        quoted = re.match(r"^(\d{8})_(\d{6})_(.+)$", name)
        if quoted:
            return f"{quoted.group(1)}{quoted.group(2)}|{quoted.group(3)}"
        legacy = re.match(r"^(\d{14})_(.+)$", name)
        if legacy:
            return f"{legacy.group(1)}|{legacy.group(2)}"
    return ""


def _resolve_interrupted_checkpoint_layout(
    log_dir: Path,
    workspace_dir: Path,
) -> tuple[Path, Path]:
    """Locate checkpoints written before run timestamps were string-quoted."""
    for candidate_log in _timestamp_path_variants(log_dir):
        for candidate_workspace in _timestamp_path_variants(workspace_dir):
            log_identity = _timestamp_run_identity(candidate_log)
            workspace_identity = _timestamp_run_identity(candidate_workspace)
            if log_identity and workspace_identity and log_identity != workspace_identity:
                continue
            manifest_name = str(
                _saved_config_value(
                    candidate_log,
                    "runtime.checkpoint_manifest_filename",
                    "checkpoint_manifest.json",
                )
            )
            if (candidate_log / manifest_name).is_file() and (
                candidate_workspace / manifest_name
            ).is_file():
                return candidate_log, candidate_workspace
    return log_dir, workspace_dir


def _safe_read_text(path: Path, limit: int = 60000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        byte_limit = max(limit, limit * 4)
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - byte_limit))
            data = f.read()
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return text[-limit:]


def _safe_read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _is_search_budget_exhausted(log_dir: Path) -> bool:
    status_name = str(_saved_config_value(log_dir, "runtime.run_status_filename", "run_status.json"))
    status = _safe_read_json_dict(log_dir / status_name)
    reason = str(status.get("termination_reason") or "").strip().lower()
    if reason in {"time_limit_exhausted", "step_limit_exhausted", "steps_completed"}:
        return True
    brief_log_name = _resolved_log_filename(
        log_dir,
        "logging.brief_log_filename",
        "AlgoEvolve.log",
        "MLEvolve.log",
    )
    log_tail = _safe_read_text(log_dir / brief_log_name, limit=120000)
    return (
        "Search budget is exhausted" in log_tail
        or "AlgoEvolve search budget exhausted" in log_tail
        or "MLEvolve search budget exhausted" in log_tail
        or "Time limit reached (configured=" in log_tail
    )


def _mark_service_budget_completed(log_dir: Path, time_limit_secs: int) -> None:
    """Normalize a watchdog-finalized run to the same status as an engine timeout."""
    status_name = str(_saved_config_value(log_dir, "runtime.run_status_filename", "run_status.json"))
    status_path = log_dir / status_name
    payload = _safe_read_json_dict(status_path)
    payload.update(
        {
            "schema_version": payload.get("schema_version") or "algoevolve.run_status.v1",
            "status": "completed",
            "termination_reason": "time_limit_exhausted",
            "time_limit_secs": int(time_limit_secs),
            "service_forced_finalize": True,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _inspect_interrupted_checkpoint(
    log_dir: Path,
    workspace_dir: Path,
) -> tuple[bool, str | None, str]:
    """Verify that resume-critical files were durably committed."""
    log_dir, workspace_dir = _resolve_interrupted_checkpoint_layout(log_dir, workspace_dir)
    manifest_name = str(
        _saved_config_value(
            log_dir,
            "runtime.checkpoint_manifest_filename",
            "checkpoint_manifest.json",
        )
    )
    search_state_name = str(
        _saved_config_value(log_dir, "runtime.search_state_filename", "search_state.json")
    )
    run_status_name = str(
        _saved_config_value(log_dir, "runtime.run_status_filename", "run_status.json")
    )
    log_manifest_path = log_dir / manifest_name
    workspace_manifest_path = workspace_dir / manifest_name
    manifest = _safe_read_json_dict(log_manifest_path)
    workspace_manifest = _safe_read_json_dict(workspace_manifest_path)
    run_status = _safe_read_json_dict(log_dir / run_status_name)

    required_paths = [
        log_dir / "journal.json",
        log_dir / search_state_name,
        log_dir / run_status_name,
        log_manifest_path,
        workspace_manifest_path,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    status_ok = run_status.get("status") == "interrupted_resumable"
    checkpoint_ids = [
        str(payload.get("checkpoint_id") or "")
        for payload in (manifest, workspace_manifest)
    ]
    checkpoint_ids_ok = not any(checkpoint_ids) or (
        all(checkpoint_ids) and checkpoint_ids[0] == checkpoint_ids[1]
    )
    manifests_ok = all(
        payload.get("status") == "interrupted_resumable"
        and payload.get("resumable") is True
        for payload in (manifest, workspace_manifest)
    ) and checkpoint_ids_ok
    ready = not missing and status_ok and manifests_ok
    if ready:
        return True, str(log_manifest_path), ""

    details: list[str] = []
    if missing:
        details.append("missing=" + ", ".join(missing))
    if not status_ok:
        details.append(f"run_status={run_status.get('status') or 'missing'}")
    if not manifests_ok:
        details.append(
            "manifest was not committed as resumable or checkpoint ids differ"
        )
    return False, str(log_manifest_path) if log_manifest_path.exists() else None, "; ".join(details)


def _interrupted_checkpoint_marker(log_dir: Path, workspace_dir: Path) -> tuple[Any, ...]:
    """Return a cheap identity for detecting a newly committed manifest pair."""
    log_dir, workspace_dir = _resolve_interrupted_checkpoint_layout(log_dir, workspace_dir)
    manifest_name = str(
        _saved_config_value(
            log_dir,
            "runtime.checkpoint_manifest_filename",
            "checkpoint_manifest.json",
        )
    )
    marker: list[Any] = []
    for path in (log_dir / manifest_name, workspace_dir / manifest_name):
        payload = _safe_read_json_dict(path)
        try:
            stat = path.stat()
            marker.extend(
                (
                    str(path),
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                    stat.st_size,
                    str(payload.get("checkpoint_id") or ""),
                )
            )
        except OSError:
            marker.extend((str(path), None, None, None, ""))
    return tuple(marker)


def _safe_tail_lines(path: Path, limit: int = 400, byte_limit: int = 512_000) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - byte_limit))
            data = f.read()
        return data.decode("utf-8", errors="ignore").splitlines()[-limit:]
    except Exception:
        return []


def _safe_read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _parse_metric_obj(metric_obj: Any) -> tuple[float | None, bool | None]:
    if not isinstance(metric_obj, dict):
        return None, None
    value = metric_obj.get("value")
    maximize = metric_obj.get("maximize")
    try:
        value = None if value is None else float(value)
    except Exception:
        value = None
    if not isinstance(maximize, bool):
        maximize = None
    return value, maximize


def _read_pending_nodes(log_dir: Path) -> list[dict[str, Any]]:
    filename = str(_saved_config_value(log_dir, "runtime.pending_nodes_filename", "pending_nodes.json"))
    payload = _safe_read_json(log_dir / filename, {})
    if not isinstance(payload, dict):
        return []
    rows = payload.get("nodes")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id") or "").strip()
        if not node_id:
            continue
        row = dict(row)
        row["id"] = node_id
        row["pending_execution"] = bool(
            row.get("pending_execution") or row.get("status") in {"generating", "pending_execution", "executing"}
        )
        out.append(row)
    return out


def _best_metric_node_id(
    nodes: list[dict[str, Any]],
) -> str | None:
    best_metric = None
    best_id = None
    best_maximize: bool | None = None
    for node in nodes:
        metric = node.get("metric")
        maximize = node.get("maximize")
        if metric is None or node.get("is_buggy") is True:
            continue
        accepted = node.get("search_eligible")
        if accepted is None:
            accepted = node.get("delivery_ready") is True
        if accepted is not True:
            continue
        if best_metric is None:
            best_metric = metric
            best_id = node.get("id")
            best_maximize = maximize
            continue
        compare_maximize = True if best_maximize is None else best_maximize
        if (compare_maximize and metric > best_metric) or (
            not compare_maximize and metric < best_metric
        ):
            best_metric = metric
            best_id = node.get("id")
    return str(best_id) if best_id else None


def _resolve_best_node(
    log_dir: Path,
    workspace_dir: Path,
    nodes: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    best_file = workspace_dir / "best_solution" / "node_id.txt"
    if best_file.exists():
        try:
            best_id = best_file.read_text(encoding="utf-8", errors="ignore").strip()
            if best_id:
                return best_id, "delivery"
        except Exception:
            pass
    accepted_best = _best_metric_node_id(nodes)
    if accepted_best:
        return accepted_best, "delivery"
    return None, None


def _resolve_best_node_id(log_dir: Path, workspace_dir: Path, nodes: list[dict[str, Any]]) -> str | None:
    return _resolve_best_node(log_dir, workspace_dir, nodes)[0]


def _parse_log_events(log_path: Path, limit: int = 400) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    pattern = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<level>[A-Z]+):\s+(?P<msg>.*)$")
    rows: list[dict[str, Any]] = []
    for line in _safe_tail_lines(log_path, limit=limit, byte_limit=768_000):
        text = line.strip()
        if not text:
            continue
        match = pattern.match(text)
        if match:
            rows.append(
                {
                    "ts": match.group("ts"),
                    "component": "algoevolve.log",
                    "event": match.group("level"),
                    "message": match.group("msg"),
                }
            )
        else:
            rows.append({"ts": "", "component": "algoevolve.log", "event": "INFO", "message": text})
    return rows


def _pick_dirs(req: SnapshotRequest) -> tuple[Path | None, Path | None]:
    log_dir = None
    workspace_dir = None

    raw_log = req.log_dir.strip()
    if raw_log:
        candidate = Path(raw_log).expanduser().resolve()
        if candidate.exists():
            log_dir = candidate

    raw_ws = req.workspace_dir.strip()
    if raw_ws:
        candidate = Path(raw_ws).expanduser().resolve()
        if candidate.exists():
            workspace_dir = candidate

    return log_dir, workspace_dir


def _build_snapshot(req: SnapshotRequest) -> dict[str, Any]:
    log_dir, workspace_dir = _pick_dirs(req)
    if log_dir is None:
        return {"engine": "algoevolve", "nodes": [], "events": []}

    if workspace_dir is None and log_dir.parent.name == "logs":
        sibling = log_dir.parent.parent / "workspace"
        if sibling.exists():
            workspace_dir = sibling.resolve()

    # filtered_journal.json is a best-path projection. UI snapshots need the
    # full search tree when the journal is reasonably sized.
    journal_source = ""
    journal = {}
    journal_path = log_dir / "journal.json"
    filtered_journal_path = log_dir / "filtered_journal.json"
    journal_max_bytes = max(
        1,
        int(_saved_config_value(log_dir, "runtime.snapshot_journal_max_bytes", 150 * 1024 * 1024)),
    )
    snapshot_event_limit = max(1, int(_saved_config_value(log_dir, "runtime.snapshot_event_limit", 400)))
    snapshot_text_limit = max(
        1000,
        int(_saved_config_value(log_dir, "runtime.snapshot_text_tail_chars", 200000)),
    )
    brief_log_name = _resolved_log_filename(
        log_dir,
        "logging.brief_log_filename",
        "AlgoEvolve.log",
        "MLEvolve.log",
    )
    verbose_log_name = _resolved_log_filename(
        log_dir,
        "logging.verbose_log_filename",
        "AlgoEvolve.verbose.log",
        "MLEvolve.verbose.log",
    )
    try:
        if journal_path.exists() and journal_path.stat().st_size <= journal_max_bytes:
            journal = _safe_read_json(journal_path, {})
            if isinstance(journal, dict) and journal:
                journal_source = "journal"
    except Exception:
        journal = {}
    if not isinstance(journal, dict) or not journal:
        journal = _safe_read_json(filtered_journal_path, {})
        if isinstance(journal, dict) and journal:
            journal_source = "filtered_journal"
    node_rows: list[dict[str, Any]] = []
    node2parent = {}
    if isinstance(journal, dict):
        node2parent = journal.get("node2parent", {}) or {}
        for node in journal.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            metric, maximize = _parse_metric_obj(node.get("metric"))
            term_out = node.get("_term_out")
            result = ""
            if isinstance(term_out, list):
                result = "".join(str(part) for part in term_out)
            elif term_out is not None:
                result = str(term_out)
            node_id = str(node.get("id") or "")
            node_rows.append(
                {
                    "id": node_id,
                    "parent_id": node2parent.get(node_id),
                    "stage": node.get("stage"),
                    "plan": node.get("plan"),
                    "code": node.get("code"),
                    "result": result,
                    "insight": node.get("llm_insight") or node.get("analysis"),
                    "llm_insight": node.get("llm_insight"),
                    "parser_analysis": node.get("parser_analysis") or node.get("analysis"),
                    "decision_signals": node.get("decision_signals"),
                    "metric": metric,
                    "maximize": maximize,
                    "is_buggy": node.get("is_buggy"),
                    "is_valid": node.get("is_valid"),
                    "runtime_ok": node.get("runtime_ok"),
                    "search_eligible": node.get("search_eligible"),
                    "score_recomputed": node.get("score_recomputed"),
                    "contract_valid": node.get("contract_valid"),
                    "artifact_ready": node.get("artifact_ready"),
                    "delivery_ready": node.get("delivery_ready"),
                    "delivery_certified": node.get("delivery_certified"),
                    "certification_source": node.get("certification_source"),
                    "certification_notes": node.get("certification_notes"),
                    "method_mode": node.get("method_mode"),
                    "visits": node.get("visits"),
                    "total_reward": node.get("total_reward"),
                    "uct": node.get("_uct"),
                    "finish_time": node.get("finish_time"),
                    "exec_time": node.get("exec_time"),
                    "branch_id": node.get("branch_id"),
                    "from_topk": node.get("from_topk"),
                }
            )

    journal_node_ids = {str(node.get("id")) for node in node_rows if node.get("id")}
    pending_nodes = [
        node for node in _read_pending_nodes(log_dir)
        if str(node.get("id")) not in journal_node_ids
    ]
    best_node_id, best_node_kind = _resolve_best_node(
        log_dir,
        workspace_dir or Path("."),
        node_rows,
    )
    best_solution_code = _safe_read_text(
        (workspace_dir / "best_solution" / "solution.py") if workspace_dir else Path(""),
        limit=snapshot_text_limit,
    )
    best_metric_text = _safe_read_text((workspace_dir / "best_solution" / "metric.txt") if workspace_dir else Path(""), limit=20000)
    task_automl_root = log_dir.parent.parent if log_dir.parent.name == "logs" else None
    dependency_installations = _safe_read_text(
        log_dir / "dependency_installations.jsonl",
        limit=snapshot_text_limit,
    )
    dependency_summary = _safe_read_json(
        log_dir / "dependency_installations_summary.json",
        {},
    )
    if task_automl_root is not None:
        if not dependency_installations:
            dependency_installations = _safe_read_text(
                task_automl_root / "dependency_installations.jsonl",
                limit=snapshot_text_limit,
            )
        if not dependency_summary:
            dependency_summary = _safe_read_json(
                task_automl_root / "dependency_installations_summary.json",
                {},
            )

    return {
        "engine": "algoevolve",
        "log_dir": str(log_dir),
        "workspace_dir": str(workspace_dir) if workspace_dir else "",
        "events": _parse_log_events(log_dir / brief_log_name, limit=snapshot_event_limit),
        "nodes": node_rows,
        "pending_nodes": pending_nodes,
        "best_node_id": best_node_id,
        "best_node_kind": best_node_kind,
        "journal_source": journal_source,
        "best_solution_code": best_solution_code,
        "best_metric_text": best_metric_text,
        "ml_log": _safe_read_text(log_dir / brief_log_name, limit=snapshot_text_limit),
        "verbose_log": _safe_read_text(log_dir / verbose_log_name, limit=snapshot_text_limit),
        "frontend_stdout": _safe_read_text(log_dir / "_frontend_stdout.log", limit=snapshot_text_limit),
        "frontend_stderr": _safe_read_text(log_dir / "_frontend_stderr.log", limit=snapshot_text_limit),
        "service_stdout": _safe_read_text(log_dir / "_service_stdout.log", limit=snapshot_text_limit),
        "service_stderr": _safe_read_text(log_dir / "_service_stderr.log", limit=snapshot_text_limit),
        "resource_usage": _safe_read_json(log_dir / "resource_usage.json", {}),
        "dependency_installations": dependency_installations,
        "dependency_installation_summary": dependency_summary,
    }


def _monitor_task_resources(
    job_id: str,
    proc: subprocess.Popen[str],
    limits: TaskResourceLimits,
    cpu_ids: list[int],
    stop_event: threading.Event,
    hard_memory_limit_active: bool = False,
) -> None:
    memory_limit_bytes = int(float(limits.memory_limit_gb) * (1024**3))
    guard_threshold = _memory_child_guard_threshold(memory_limit_bytes, hard_memory_limit_active)
    guard_enabled = memory_limit_bytes > 0 and (
        not hard_memory_limit_active or guard_threshold < memory_limit_bytes
    )
    interval = max(0.1, float(limits.monitor_interval_seconds))
    peak_memory = 0
    while not stop_event.is_set() and proc.poll() is None:
        apply_process_tree_cpu_affinity(proc.pid, cpu_ids)
        current_memory = process_tree_memory_bytes(proc.pid)
        peak_memory = max(peak_memory, current_memory)
        store.update(
            job_id,
            current_memory_bytes=current_memory,
            peak_memory_bytes=peak_memory,
        )
        if guard_enabled and current_memory > guard_threshold:
            action = relieve_process_tree_memory_pressure(proc.pid, guard_threshold)
            if action.action == "terminated_child":
                warning = (
                    "memory_limit_child_guard: stopped memory-heavy execution child "
                    f"pid={action.child_pid} after task memory reached {format_bytes(action.observed_bytes)}; "
                    f"controller guard={format_bytes(action.limit_bytes)}, "
                    f"configured limit={format_bytes(memory_limit_bytes)}. AlgoEvolve controller continues."
                )
            elif action.action == "controller_over_limit":
                warning = (
                    "memory_limit_child_guard: controller memory exceeded the configured limit, "
                    "but the controller was preserved because whole-task termination is disabled. "
                    f"observed={format_bytes(action.observed_bytes)}, limit={format_bytes(action.limit_bytes)}"
                )
            else:
                warning = None
            if warning:
                store.update(job_id, resource_warning=warning)
        stop_event.wait(interval)


def _run_job(job_id: str, req: StartAlgoEvolveRequest, actual_log_dir: Path, actual_workspace_dir: Path, run_timestamp: str) -> None:
    limits = req.resources or _resolve_resource_limits(req)
    run_args = list(req.args)
    # OmegaConf interprets 20260722_015005 as an integer with a numeric
    # separator unless the dot-list value is explicitly quoted.
    run_args = _with_cli_override(run_args, "runtime.run_timestamp", json.dumps(run_timestamp))
    run_args = _with_cli_override(run_args, "runtime.resume_run", req.resume)

    def runtime_value(name: str, default: Any) -> Any:
        raw = _extract_cli_override(run_args, f"runtime.{name}")
        if raw is not None:
            return raw
        return _base_config_value(f"runtime.{name}", default)

    job_status_tail_chars = max(0, int(float(runtime_value("job_status_tail_chars", 60000))))
    service_log_tail_chars = max(0, int(float(runtime_value("service_log_tail_chars", 200000))))
    service_last_error_chars = max(1, int(float(runtime_value("service_last_error_chars", 300))))
    termination_wait_default = max(0.0, float(runtime_value("termination_wait_seconds", 20)))
    interruption_checkpoint_wait = max(
        0.0,
        float(runtime_value("interruption_checkpoint_wait_seconds", 30)),
    )
    store.update(
        job_id,
        job_status_tail_chars=job_status_tail_chars,
        stop_wait_seconds=interruption_checkpoint_wait,
    )
    cmd = [req.python_executable or "python", "run.py", *run_args]
    workdir = req.working_dir.strip() or DEFAULT_WORKDIR
    job_runtime = store.get(job_id)
    env = os.environ.copy()
    env.update(req.env_overrides or {})
    env.update(accelerator_visibility_env(limits.accelerator_mode, limits.accelerator_device_ids))
    parallel_workers_raw = _extract_cli_override(run_args, "agent.search.parallel_search_num")
    try:
        parallel_workers = max(1, int(float(parallel_workers_raw or 1)))
    except Exception:
        parallel_workers = 1
    cpu_enforcement = cpu_enforcement_capabilities()
    env.update(
        cpu_limit_environment(
            limits.cpu_cores,
            parallel_workers,
            capabilities=cpu_enforcement,
        )
    )
    assigned_cpu_ids_json = json.dumps(job_runtime.assigned_cpu_ids)
    env["ALGOEVOLVE_ASSIGNED_CPU_IDS"] = assigned_cpu_ids_json
    env["MLEVOLVE_ASSIGNED_CPU_IDS"] = assigned_cpu_ids_json
    env["ALGOEVOLVE_MEMORY_LIMIT_GB"] = str(limits.memory_limit_gb)
    env["MLEVOLVE_MEMORY_LIMIT_GB"] = str(limits.memory_limit_gb)
    config_path: Path | None = None
    if req.config_path.strip():
        config_path = Path(req.config_path).expanduser()
        if not config_path.is_absolute():
            config_path = Path(workdir) / config_path
    if config_path is not None:
        resolved_config_path = str(config_path.resolve())
        env["ALGOEVOLVE_CONFIG_PATH"] = resolved_config_path
        env["MLEVOLVE_CONFIG_PATH"] = resolved_config_path
    else:
        env.pop("ALGOEVOLVE_CONFIG_PATH", None)
        env.pop("MLEVOLVE_CONFIG_PATH", None)
    env["ALGOEVOLVE_RUN_TIMESTAMP"] = run_timestamp
    env["MLEVOLVE_RUN_TIMESTAMP"] = run_timestamp
    resume_flag = "1" if req.resume else "0"
    env["ALGOEVOLVE_RESUME_RUN"] = resume_flag
    env["MLEVOLVE_RESUME_RUN"] = resume_flag
    env.setdefault("PYTHONFAULTHANDLER", "1")

    memory_limit_bytes = int(float(limits.memory_limit_gb) * (1024**3))
    memory_limiter = None
    memory_setup_warning: str | None = None
    if memory_limit_bytes > 0:
        try:
            memory_limiter = create_process_tree_memory_limiter(memory_limit_bytes)
        except Exception as exc:
            fallback_name = "POSIX RLIMIT_AS plus child guard" if os.name == "posix" else "child-process guard"
            memory_setup_warning = f"hard memory limiter setup failed; using {fallback_name}: {exc}"
    if memory_limiter is not None:
        memory_enforcement = memory_limiter.describe()
    elif memory_limit_bytes > 0:
        memory_enforcement = {
            **memory_enforcement_capabilities(),
            "backend": "posix_rlimit_as_plus_child_guard" if os.name == "posix" else "process_tree_child_guard",
            "hard_limit": False,
            "hard_limit_supported": False,
            "over_limit_behavior": (
                "per_process_allocation_failure_then_child_guard"
                if os.name == "posix"
                else "terminate_memory_heavy_child_process"
            ),
            "per_process_address_space_limit": os.name == "posix",
            "limit_bytes": memory_limit_bytes,
        }
    else:
        memory_enforcement = {
            "backend": "disabled",
            "hard_limit": False,
            "total_process_tree": True,
            "over_limit_behavior": "unlimited",
            "whole_task_termination": False,
            "limit_bytes": 0,
        }
    store.update(
        job_id,
        cpu_enforcement=cpu_enforcement,
        memory_enforcement=memory_enforcement,
        resource_warning=memory_setup_warning,
    )
    memory_mode = str(memory_enforcement.get("backend") or "disabled")
    env["ALGOEVOLVE_MEMORY_LIMIT_BYTES"] = str(memory_limit_bytes)
    env["MLEVOLVE_MEMORY_LIMIT_BYTES"] = str(memory_limit_bytes)
    env["ALGOEVOLVE_MEMORY_ENFORCEMENT_MODE"] = memory_mode
    env["MLEVOLVE_MEMORY_ENFORCEMENT_MODE"] = memory_mode

    try:
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except Exception as exc:
        if memory_limiter is not None:
            memory_limiter.close()
        store.update(job_id, status="failed", last_error=f"start failed: {exc}")
        return

    store.set_process(job_id, proc)
    if memory_limiter is not None:
        try:
            memory_limiter.attach(proc.pid)
        except Exception as exc:
            memory_limiter.close()
            memory_limiter = None
            memory_enforcement = {
                **memory_enforcement_capabilities(),
                "backend": "process_tree_child_guard",
                "hard_limit": False,
                "hard_limit_supported": False,
                "limit_bytes": memory_limit_bytes,
                "setup_error": str(exc),
            }
            store.update(
                job_id,
                memory_enforcement=memory_enforcement,
                resource_warning=f"hard memory limiter attach failed; using child-process guard: {exc}",
            )
    cpu_errors = apply_process_tree_cpu_affinity(proc.pid, job_runtime.assigned_cpu_ids)
    if cpu_errors:
        message = "resource_limit_setup_failed: " + "; ".join(cpu_errors[:5])
        terminate_process_tree(proc.pid)
        out, err = proc.communicate()
        store.update(
            job_id,
            status="failed",
            exit_code=proc.returncode,
            last_error=message,
            resource_violation=message,
            stdout_tail=out or "",
            stderr_tail=err or "",
        )
        if memory_limiter is not None:
            memory_limiter.close()
        return

    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_task_resources,
        args=(job_id, proc, limits, job_runtime.assigned_cpu_ids, monitor_stop, memory_limiter is not None),
        daemon=True,
    )
    monitor_thread.start()
    time_limit_secs = _extract_time_limit_secs(run_args)
    timed_out = False
    try:
        if time_limit_secs is not None:
            configured_buffer = _extract_cli_override(run_args, "runtime.graceful_shutdown_buffer_seconds")
            try:
                shutdown_buffer = (
                    int(float(configured_buffer))
                    if configured_buffer is not None
                    else int(
                        req.graceful_shutdown_buffer_secs
                        if req.graceful_shutdown_buffer_secs is not None
                        else runtime_value("graceful_shutdown_buffer_seconds", 600)
                    )
                )
            except Exception:
                shutdown_buffer = int(runtime_value("graceful_shutdown_buffer_seconds", 600))
            configured_startup_buffer = _extract_cli_override(
                run_args,
                "runtime.service_startup_buffer_seconds",
            )
            try:
                startup_buffer = (
                    max(0, int(float(configured_startup_buffer)))
                    if configured_startup_buffer is not None
                    else max(0, int(runtime_value("service_startup_buffer_seconds", 1800)))
                )
            except Exception:
                startup_buffer = max(0, int(runtime_value("service_startup_buffer_seconds", 1800)))
            total_timeout = time_limit_secs + startup_buffer + max(0, shutdown_buffer)
            try:
                out, err = proc.communicate(timeout=total_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[arg-type]
                    else:
                        os.killpg(proc.pid, signal.SIGTERM)
                    configured_wait = _extract_cli_override(run_args, "runtime.termination_wait_seconds")
                    try:
                        termination_wait = (
                            max(0.0, float(configured_wait))
                            if configured_wait is not None
                            else termination_wait_default
                        )
                    except Exception:
                        termination_wait = termination_wait_default
                    out, err = proc.communicate(timeout=termination_wait)
                except Exception:
                    terminate_process_tree(proc.pid)
                    out, err = proc.communicate()
        else:
            out, err = proc.communicate()
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=max(1.0, float(limits.monitor_interval_seconds) * 2.0))
        if memory_limiter is not None:
            limiter_peak = memory_limiter.peak_memory_bytes()
            final_state = store.get(job_id)
            if limiter_peak > final_state.peak_memory_bytes:
                store.update(job_id, peak_memory_bytes=limiter_peak)
            memory_limiter.close()

    exit_code = proc.returncode
    current_job = store.get(job_id)
    stop_requested = bool(current_job.stop_requested or current_job.status in {"stopping", "stopped"})
    resource_violation = str(current_job.resource_violation or "").strip()
    native_exit_reason = _native_exit_reason(exit_code)
    checkpoint_ready = False
    checkpoint_path: str | None = None
    checkpoint_error = ""
    checkpoint_log_dir, checkpoint_workspace_dir = _resolve_interrupted_checkpoint_layout(
        actual_log_dir,
        actual_workspace_dir,
    )
    if resource_violation:
        status = "failed"
    elif stop_requested:
        checkpoint_ready, checkpoint_path, checkpoint_error = _inspect_interrupted_checkpoint(
            checkpoint_log_dir,
            checkpoint_workspace_dir,
        )
        status = "interrupted_resumable" if checkpoint_ready else "interrupted_incomplete"
    elif timed_out:
        status = "completed"
        _mark_service_budget_completed(actual_log_dir, int(time_limit_secs or 0))
    elif _is_interrupted_exit_code(exit_code):
        checkpoint_ready, checkpoint_path, checkpoint_error = _inspect_interrupted_checkpoint(
            checkpoint_log_dir,
            checkpoint_workspace_dir,
        )
        status = "interrupted_resumable" if checkpoint_ready else "interrupted_incomplete"
    elif exit_code == 0:
        status = "completed"
    else:
        status = "failed"
    last_error = None
    if resource_violation:
        last_error = resource_violation
    elif timed_out:
        completion_note = (
            "AlgoEvolve search budget was exhausted; the service finalized the saved "
            "search tree and current Top-K artifacts normally. "
            f"search_limit={time_limit_secs}s, startup_buffer={int(startup_buffer)}s, "
            f"grace={int(shutdown_buffer)}s."
        )
        out = "\n".join(part for part in ((out or "").rstrip(), completion_note) if part)
    elif status == "interrupted_resumable":
        last_error = (
            "AlgoEvolve was interrupted after committing the durable search tree, "
            "generated-code resume actions, and the existing Top-K artifact index. "
            "It can be resumed or reported now."
        )
    elif status == "interrupted_incomplete":
        actor = "user stop" if stop_requested else f"signal exit code {exit_code}"
        last_error = (
            f"AlgoEvolve was terminated after {actor}, but the resumable checkpoint "
            f"could not be verified: {checkpoint_error or 'unknown checkpoint error'}"
        )
    elif native_exit_reason:
        tail = (err or out or "").strip()
        secondary = tail.splitlines()[-1][:service_last_error_chars] if tail else ""
        last_error = (
            f"AlgoEvolve native crash: {native_exit_reason}. "
            f"Peak task memory={format_bytes(current_job.peak_memory_bytes)}, "
            f"configured limit={format_bytes(memory_limit_bytes)}."
        )
        if secondary:
            last_error += f" Last stderr/output: {secondary}"
    elif exit_code != 0:
        tail = (err or out or "").strip()
        last_error = tail.splitlines()[-1][:service_last_error_chars] if tail else f"AlgoEvolve exited with code {exit_code}"

    if checkpoint_ready:
        actual_log_dir = checkpoint_log_dir
        actual_workspace_dir = checkpoint_workspace_dir

    try:
        actual_log_dir.mkdir(parents=True, exist_ok=True)
        if out:
            (actual_log_dir / "_service_stdout.log").write_text(
                _tail_text(out, service_log_tail_chars),
                encoding="utf-8",
                errors="ignore",
            )
        if err:
            (actual_log_dir / "_service_stderr.log").write_text(
                _tail_text(err, service_log_tail_chars),
                encoding="utf-8",
                errors="ignore",
            )
        final_resource_state = store.get(job_id)
        (actual_log_dir / "resource_usage.json").write_text(
            json.dumps(
                {
                    "limits": final_resource_state.resource_limits,
                    "assigned_cpu_ids": final_resource_state.assigned_cpu_ids,
                    "cpu_enforcement": final_resource_state.cpu_enforcement,
                    "current_memory_bytes": final_resource_state.current_memory_bytes,
                    "peak_memory_bytes": final_resource_state.peak_memory_bytes,
                    "memory_enforcement": final_resource_state.memory_enforcement,
                    "resource_violation": final_resource_state.resource_violation,
                    "resource_warning": final_resource_state.resource_warning,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    store.update(
        job_id,
        status=status,
        exit_code=(0 if timed_out and status == "completed" else exit_code),
        last_error=last_error,
        stdout_tail=_tail_text(out or "", service_log_tail_chars),
        stderr_tail=_tail_text(err or "", service_log_tail_chars),
        log_dir=str(actual_log_dir),
        workspace_dir=str(actual_workspace_dir),
        checkpoint_ready=(status == "interrupted_resumable"),
        resumable=(status == "interrupted_resumable"),
        checkpoint_manifest_path=(
            checkpoint_path
            if status in {"interrupted_resumable", "interrupted_incomplete"}
            else None
        ),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/resources/inventory")
def resource_inventory(python_executable: str = "") -> dict[str, Any]:
    return detect_resource_inventory(python_executable or None)


@app.post("/jobs/start")
def start_job(req: StartAlgoEvolveRequest) -> dict[str, Any]:
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    actual_log_dir, actual_workspace_dir, final_run_name = _resolve_run_layout(req, run_timestamp)
    actual_log_dir.parent.mkdir(parents=True, exist_ok=True)
    actual_workspace_dir.parent.mkdir(parents=True, exist_ok=True)

    resources = _resolve_resource_limits(req)
    req.resources = resources
    try:
        cpu_ids = choose_cpu_ids(resources.cpu_cores)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if resources.accelerator_mode == "selected":
        inventory = detect_resource_inventory(req.python_executable)
        selection_errors = validate_accelerator_selection(
            resources.accelerator_mode,
            resources.accelerator_device_ids,
            inventory,
        )
        if selection_errors:
            raise HTTPException(status_code=400, detail=selection_errors)

    job = store.create(
        task_id=req.task_id,
        log_dir=str(actual_log_dir),
        workspace_dir=str(actual_workspace_dir),
        resource_limits=resources.model_dump(),
        assigned_cpu_ids=cpu_ids,
    )
    thread = threading.Thread(
        target=_run_job,
        args=(job.job_id, req, actual_log_dir, actual_workspace_dir, run_timestamp),
        daemon=True,
    )
    thread.start()
    return {
        "job_id": job.job_id,
        "status": "started",
        "engine": "algoevolve",
        "run_name": final_run_name,
        "log_dir": str(actual_log_dir),
        "workspace_dir": str(actual_workspace_dir),
        "resources": resources.model_dump(),
        "assigned_cpu_ids": cpu_ids,
        "cpu_enforcement": cpu_enforcement_capabilities(),
        "memory_enforcement": memory_enforcement_capabilities(),
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return store.status(job_id).model_dump()


def _stop_process_after_checkpoint_window(
    proc: subprocess.Popen[str],
    wait_seconds: float,
    log_dir: Path,
    workspace_dir: Path,
) -> None:
    baseline_marker = _interrupted_checkpoint_marker(log_dir, workspace_dir)
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[arg-type]
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        terminate_process_tree(proc.pid)
        return

    process_deadline = time.monotonic() + max(0.0, wait_seconds)
    while proc.poll() is None and time.monotonic() < process_deadline:
        checkpoint_ready, _, _ = _inspect_interrupted_checkpoint(log_dir, workspace_dir)
        if (
            checkpoint_ready
            and _interrupted_checkpoint_marker(log_dir, workspace_dir) != baseline_marker
        ):
            # The manifest pair is the final commit marker. Kill the whole
            # process tree now instead of waiting for running LLM threads.
            terminate_process_tree(proc.pid)
            return
        time.sleep(0.05)
    if proc.poll() is None:
        terminate_process_tree(proc.pid)


@app.post("/jobs/stop")
def stop_job(req: StopRequest) -> dict[str, Any]:
    job = store.get(req.job_id)
    proc = job.process
    if proc is None or proc.poll() is not None:
        return store.status(req.job_id).model_dump()
    if job.status == "stopping":
        return store.status(req.job_id).model_dump()
    store.update(req.job_id, status="stopping", stop_requested=True, last_error="stop requested by user")
    threading.Thread(
        target=_stop_process_after_checkpoint_window,
        args=(
            proc,
            float(job.stop_wait_seconds),
            Path(job.log_dir),
            Path(job.workspace_dir),
        ),
        daemon=True,
    ).start()
    return store.status(req.job_id).model_dump()


@app.post("/snapshot")
def snapshot(req: SnapshotRequest) -> dict[str, Any]:
    try:
        return _build_snapshot(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"snapshot failed: {exc}")
