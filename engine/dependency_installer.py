"""Controlled recovery for generated scripts that fail on a missing package."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


logger = logging.getLogger("AlgoEvolve")

INSTALL_DECLARATION_RE = re.compile(
    r"^\s*#\s*(?:ALGOEVOLVE|MLEVOLVE)_PIP_INSTALL"
    r"(?:\[(?P<module>[A-Za-z_][A-Za-z0-9_.]*)\])?"
    r"\s*:\s*(?P<command>.+?)\s*$",
    re.MULTILINE,
)
MISSING_MODULE_RE = re.compile(
    r"ModuleNotFoundError\s*:\s*No module named\s+['\"](?P<module>[^'\"]+)['\"]"
)
DIRECT_INSTALL_TEXT_RE = re.compile(
    r"(?im)^\s*[!%]\s*(?:pip|conda|mamba|uv)\s+(?:install|add)\b"
)


@contextmanager
def _exclusive_file_lock(path: Path, timeout_seconds: int):
    """Serialize pip mutations across AutoML processes sharing one interpreter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    deadline = time.monotonic() + max(1, timeout_seconds)
    locked = False
    try:
        while not locked:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for dependency installation lock: {path}"
                    )
                time.sleep(0.1)
        yield
    finally:
        if locked:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


@dataclass(frozen=True)
class DependencyRecoveryDecision:
    retry: bool
    message: str
    distribution: str = ""
    requirement: str = ""


def extract_missing_module(stderr: str) -> str:
    matches = list(MISSING_MODULE_RE.finditer(stderr or ""))
    if not matches:
        return ""
    return matches[-1].group("module").split(".", 1)[0].strip()


def extract_install_declarations(code: str) -> list[str]:
    return [match.group("command").strip() for match in INSTALL_DECLARATION_RE.finditer(code or "")]


def extract_scoped_install_declarations(code: str) -> list[tuple[str, str]]:
    return [
        (
            str(match.group("module") or "").split(".", 1)[0].strip().lower(),
            match.group("command").strip(),
        )
        for match in INSTALL_DECLARATION_RE.finditer(code or "")
    ]


def parse_declared_requirement(command: str) -> Requirement | None:
    """Accept one package requirement, never shell flags, URLs, or multiple packages."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    if len(tokens) == 3 and tokens[:2] == ["pip", "install"]:
        raw_requirement = tokens[2]
    elif len(tokens) == 5 and tokens[1:4] == ["-m", "pip", "install"]:
        raw_requirement = tokens[4]
    else:
        return None
    try:
        requirement = Requirement(raw_requirement)
    except InvalidRequirement:
        return None
    if requirement.url or requirement.marker:
        return None
    return requirement


def find_unsafe_installation_call(code: str) -> str:
    """Return a reason when generated solution code tries to mutate its environment."""
    if DIRECT_INSTALL_TEXT_RE.search(code or ""):
        return "notebook/shell package installation syntax is forbidden"
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            owner = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
            name = f"{owner}.{node.func.attr}" if owner else node.func.attr
        if name in {"os.system", "os.popen", "subprocess.run", "subprocess.call", "subprocess.Popen", "subprocess.check_call", "subprocess.check_output"}:
            literals = [
                value
                for value in (
                    constant.value
                    for constant in ast.walk(node)
                    if isinstance(constant, ast.Constant)
                )
                if isinstance(value, str)
            ]
            joined = " ".join(literals).lower()
            if re.search(r"(?:^|[\s/\\-])(?:pip|conda|mamba|uv)(?:[\s.]|$)", joined):
                return f"direct package installation through {name} is forbidden"
    return ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class DependencyInstaller:
    """Recover missing imports into an isolated task package directory."""

    def __init__(self, cfg: Any) -> None:
        exec_cfg = getattr(cfg, "exec", None)
        self.enabled = bool(
            getattr(exec_cfg, "auto_install_missing_dependencies", False)
        )
        self.policy = str(
            getattr(exec_cfg, "dependency_install_policy", "ai_declared")
            or "ai_declared"
        ).strip().lower()
        if self.policy not in {"ai_declared", "allowlist"}:
            raise ValueError(
                "exec.dependency_install_policy must be 'ai_declared' or 'allowlist'"
            )
        self.timeout_seconds = max(
            1,
            int(getattr(exec_cfg, "dependency_install_timeout_seconds", 600)),
        )
        self.lock_timeout_seconds = max(
            1,
            int(
                getattr(
                    exec_cfg,
                    "dependency_install_lock_timeout_seconds",
                    self.timeout_seconds + 60,
                )
            ),
        )
        self.max_packages_per_execution = max(
            0,
            int(
                getattr(
                    exec_cfg,
                    "dependency_install_max_packages_per_execution",
                    3,
                )
            ),
        )
        self.output_tail_chars = max(
            0,
            int(getattr(exec_cfg, "dependency_install_output_tail_chars", 8000)),
        )
        self.allowlist = {
            canonicalize_name(str(name))
            for name in (
                getattr(exec_cfg, "dependency_install_allowlist", []) or []
            )
            if str(name).strip()
        }
        self.import_map = {
            str(module).strip().lower(): canonicalize_name(str(distribution))
            for module, distribution in dict(
                getattr(exec_cfg, "dependency_import_map", {}) or {}
            ).items()
            if str(module).strip() and str(distribution).strip()
        }
        self.package_specs = {
            canonicalize_name(str(distribution)): str(spec).strip()
            for distribution, spec in dict(
                getattr(exec_cfg, "dependency_package_specs", {}) or {}
            ).items()
            if str(distribution).strip() and str(spec).strip()
        }
        log_dir = Path(getattr(cfg, "log_dir", Path.cwd())).resolve()
        target_value = str(
            getattr(exec_cfg, "dependency_install_target_path", "") or ""
        ).strip()
        if target_value:
            install_target = Path(target_value).expanduser()
            if not install_target.is_absolute():
                install_target = log_dir / install_target
        else:
            install_target = log_dir / "python_packages"
        self.install_target_path = install_target.resolve()
        detail_name = str(
            getattr(
                exec_cfg,
                "dependency_install_log_filename",
                "dependency_installations.jsonl",
            )
            or "dependency_installations.jsonl"
        )
        summary_name = str(
            getattr(
                exec_cfg,
                "dependency_install_summary_filename",
                "dependency_installations_summary.json",
            )
            or "dependency_installations_summary.json"
        )
        self.run_log_dir = log_dir
        self.local_detail_path = log_dir / detail_name
        self.detail_paths = [self.local_detail_path]
        self.central_detail_path: Path | None = None
        central_log = str(
            getattr(exec_cfg, "dependency_install_central_log_path", "") or ""
        ).strip()
        if central_log:
            central_path = Path(central_log).expanduser().resolve()
            if central_path not in self.detail_paths:
                self.detail_paths.append(central_path)
            self.central_detail_path = central_path
        self.local_summary_path = log_dir / summary_name
        self.central_summary_path: Path | None = None
        central_summary = str(
            getattr(exec_cfg, "dependency_install_central_summary_path", "") or ""
        ).strip()
        if central_summary:
            central_summary_path = Path(central_summary).expanduser().resolve()
            if central_summary_path != self.local_summary_path:
                self.central_summary_path = central_summary_path

        self._condition = threading.Condition(threading.RLock())
        self._records_lock = threading.Lock()
        self._pip_install_lock = threading.Lock()
        lock_identity = (
            f"{Path(sys.executable).resolve()}|{self.install_target_path}"
        ).casefold()
        interpreter_hash = hashlib.sha256(lock_identity.encode("utf-8")).hexdigest()[:20]
        self._interpreter_lock_path = (
            Path(tempfile.gettempdir())
            / "algoevolve-dependency-locks"
            / f"{interpreter_hash}.lock"
        )
        self._status_by_distribution: dict[str, dict[str, Any]] = {}
        self._records: list[dict[str, Any]] = []
        self._active_installers: set[subprocess.Popen[str]] = set()
        self._load_previous_run_attempts()

    def execution_environment(
        self,
        base_environment: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        environment = dict(base_environment or os.environ)
        target = str(self.install_target_path)
        existing = str(environment.get("PYTHONPATH") or "").strip()
        environment["PYTHONPATH"] = (
            target if not existing else os.pathsep.join((target, existing))
        )
        environment["ALGOEVOLVE_TASK_PACKAGE_DIR"] = target
        environment["MLEVOLVE_TASK_PACKAGE_DIR"] = target
        return environment

    def terminate_all(self) -> None:
        with self._condition:
            processes = list(self._active_installers)
        for process in processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            except Exception:
                continue

    def maybe_recover(
        self,
        *,
        code: str,
        stderr: str,
        node_id: str,
        execution_started_monotonic: float,
        installed_for_execution: set[str],
    ) -> DependencyRecoveryDecision:
        missing_module = extract_missing_module(stderr)
        if not missing_module:
            return DependencyRecoveryDecision(False, "not a ModuleNotFoundError")
        if not self.enabled:
            return DependencyRecoveryDecision(
                False,
                "automatic missing-dependency installation is disabled",
            )
        if len(installed_for_execution) >= self.max_packages_per_execution:
            return DependencyRecoveryDecision(
                False,
                "per-execution dependency installation limit reached",
            )

        selection = self._select_requirement(code, missing_module)
        if selection is None:
            self._record_rejection(
                node_id=node_id,
                missing_module=missing_module,
                reason=(
                    "no unambiguous safe AI package declaration or configured import mapping"
                ),
            )
            return DependencyRecoveryDecision(
                False,
                f"no unambiguous package selection for missing module {missing_module!r}",
            )
        distribution, requirement_text, source, declared_command = selection
        if distribution in installed_for_execution:
            return DependencyRecoveryDecision(
                False,
                f"dependency {distribution!r} is still missing after its one installation attempt",
                distribution,
                requirement_text,
            )

        with self._condition:
            existing = self._status_by_distribution.get(distribution)
            if existing and existing.get("status") == "installing":
                while (
                    self._status_by_distribution.get(distribution, {}).get("status")
                    == "installing"
                ):
                    self._condition.wait(timeout=0.25)
                existing = self._status_by_distribution.get(distribution, {})
                if (
                    existing.get("status") == "installed"
                    and execution_started_monotonic
                    <= float(existing.get("completed_monotonic") or 0.0)
                ):
                    installed_for_execution.add(distribution)
                    return DependencyRecoveryDecision(
                        True,
                        f"dependency {distribution!r} was installed by a parallel node; rerunning the same script",
                        distribution,
                        requirement_text,
                    )
                return DependencyRecoveryDecision(
                    False,
                    f"dependency {distribution!r} was already attempted in this run",
                    distribution,
                    requirement_text,
                )
            if existing:
                if (
                    existing.get("status") == "installed"
                    and not existing.get("loaded_from_disk")
                    and execution_started_monotonic
                    <= float(existing.get("completed_monotonic") or 0.0)
                ):
                    installed_for_execution.add(distribution)
                    return DependencyRecoveryDecision(
                        True,
                        f"dependency {distribution!r} was installed concurrently; rerunning the same script",
                        distribution,
                        requirement_text,
                    )
                return DependencyRecoveryDecision(
                    False,
                    f"dependency {distribution!r} was already attempted in this run",
                    distribution,
                    requirement_text,
                )
            self._status_by_distribution[distribution] = {
                "status": "installing",
                "completed_monotonic": 0.0,
            }

        try:
            with self._pip_install_lock, _exclusive_file_lock(
                self._interpreter_lock_path,
                self.lock_timeout_seconds,
            ):
                record = self._install(
                    node_id=node_id,
                    missing_module=missing_module,
                    distribution=distribution,
                    requirement=requirement_text,
                    source=source,
                    declared_command=declared_command,
                )
        except TimeoutError as exc:
            record = self._record_lock_timeout(
                node_id=node_id,
                missing_module=missing_module,
                distribution=distribution,
                requirement=requirement_text,
                source=source,
                declared_command=declared_command,
                reason=str(exc),
            )
        with self._condition:
            self._status_by_distribution[distribution] = {
                "status": "installed" if record["success"] else "failed",
                "completed_monotonic": time.monotonic(),
                "loaded_from_disk": False,
            }
            self._condition.notify_all()

        if record["success"]:
            installed_for_execution.add(distribution)
            return DependencyRecoveryDecision(
                True,
                f"installed {requirement_text!r} into task package directory "
                f"{self.install_target_path}; rerunning the same script",
                distribution,
                requirement_text,
            )
        return DependencyRecoveryDecision(
            False,
            f"pip installation failed for {requirement_text!r}; it will not be retried in this run",
            distribution,
            requirement_text,
        )

    def _record_lock_timeout(
        self,
        *,
        node_id: str,
        missing_module: str,
        distribution: str,
        requirement: str,
        source: str,
        declared_command: str,
        reason: str,
    ) -> dict[str, Any]:
        record = {
            "schema_version": "algoevolve.dependency_installation.v1",
            "timestamp": _utc_now(),
            "run_log_dir": str(self.run_log_dir),
            "node_id": node_id,
            "missing_module": missing_module,
            "distribution": distribution,
            "requirement": requirement,
            "selection_source": source,
            "llm_declared_command": declared_command,
            "executed_command": [],
            "python_executable": sys.executable,
            "status": "lock_timeout",
            "success": False,
            "exit_code": None,
            "duration_seconds": float(self.lock_timeout_seconds),
            "stdout_tail": "",
            "stderr_tail": reason,
        }
        self._append_record(record)
        return record

    def _select_requirement(
        self,
        code: str,
        missing_module: str,
    ) -> tuple[str, str, str, str] | None:
        module_key = missing_module.strip().lower()
        expected_distribution = self.import_map.get(module_key)
        declarations = extract_scoped_install_declarations(code)
        valid_declarations: list[tuple[str, str, str, str]] = []
        for declared_module, command in declarations:
            requirement = parse_declared_requirement(command)
            if requirement is None:
                continue
            distribution = canonicalize_name(requirement.name)
            if self.policy == "allowlist" and distribution not in self.allowlist:
                continue
            trusted_spec = self.package_specs.get(distribution)
            requirement_text = trusted_spec or str(requirement)
            valid_declarations.append(
                (declared_module, distribution, requirement_text, command)
            )

        scoped_declarations = {
            distribution: (requirement_text, command)
            for declared_module, distribution, requirement_text, command in valid_declarations
            if declared_module == module_key
        }
        if len(scoped_declarations) == 1:
            distribution, (requirement_text, command) = next(
                iter(scoped_declarations.items())
            )
            return (
                distribution,
                requirement_text,
                "llm_declaration_module",
                command,
            )

        applicable_declarations = [
            (distribution, requirement_text, command)
            for declared_module, distribution, requirement_text, command in valid_declarations
            if not declared_module or declared_module == module_key
        ]

        if expected_distribution:
            for distribution, requirement_text, command in applicable_declarations:
                if expected_distribution == distribution:
                    return (
                        distribution,
                        requirement_text,
                        "llm_declaration_mapped",
                        command,
                    )
        unique_declarations = {
            distribution: (requirement_text, command)
            for distribution, requirement_text, command in applicable_declarations
            if not any(
                declared_module
                for declared_module, candidate, _requirement, _command in valid_declarations
                if candidate == distribution and _command == command
            )
        }
        if len(unique_declarations) == 1:
            distribution, (requirement_text, command) = next(
                iter(unique_declarations.items())
            )
            return (
                distribution,
                requirement_text,
                "llm_declaration_selected",
                command,
            )

        if expected_distribution and (
            self.policy == "ai_declared" or expected_distribution in self.allowlist
        ):
            requirement_text = self.package_specs.get(
                expected_distribution,
                expected_distribution,
            )
            return (
                expected_distribution,
                requirement_text,
                "configured_import_map",
                "",
            )
        direct_distribution = canonicalize_name(module_key)
        if self.policy == "allowlist" and direct_distribution in self.allowlist:
            requirement_text = self.package_specs.get(
                direct_distribution,
                direct_distribution,
            )
            return (
                direct_distribution,
                requirement_text,
                "direct_name_match",
                "",
            )
        return None

    def _install(self, **metadata: Any) -> dict[str, Any]:
        requirement = str(metadata["requirement"])
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            str(self.install_target_path),
            requirement,
        ]
        started = time.monotonic()
        process: subprocess.Popen[str] | None = None
        stdout = ""
        stderr = ""
        exit_code: int | None = None
        status = "failed"
        try:
            self.install_target_path.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.execution_environment(
                    {**os.environ, "PYTHONUNBUFFERED": "1"}
                ),
            )
            with self._condition:
                self._active_installers.add(process)
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                exit_code = process.returncode
                status = "installed" if exit_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = process.returncode
                status = "timeout"
        except Exception as exc:
            stderr = str(exc)
            status = "installer_error"
        finally:
            if process is not None:
                with self._condition:
                    self._active_installers.discard(process)

        installed_version = (
            self._installed_distribution_version(str(metadata["distribution"]))
            if status == "installed"
            else ""
        )
        record = {
            "schema_version": "algoevolve.dependency_installation.v1",
            "timestamp": _utc_now(),
            "run_log_dir": str(self.run_log_dir),
            "node_id": str(metadata["node_id"]),
            "missing_module": str(metadata["missing_module"]),
            "distribution": str(metadata["distribution"]),
            "requirement": requirement,
            "selection_source": str(metadata["source"]),
            "llm_declared_command": str(metadata["declared_command"]),
            "executed_command": command,
            "python_executable": sys.executable,
            "install_target": str(self.install_target_path),
            "installed_version": installed_version,
            "resolved_requirement": (
                f"{metadata['distribution']}=={installed_version}"
                if installed_version
                else requirement
            ),
            "status": status,
            "success": status == "installed",
            "exit_code": exit_code,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout_tail": stdout[-self.output_tail_chars :],
            "stderr_tail": stderr[-self.output_tail_chars :],
        }
        self._append_record(record)
        logger.info(
            "[dependency-install] distribution=%s status=%s node=%s python=%s",
            record["distribution"],
            record["status"],
            record["node_id"],
            record["python_executable"],
        )
        return record

    def _installed_distribution_version(self, distribution: str) -> str:
        canonical_name = canonicalize_name(distribution)
        try:
            for metadata in importlib.metadata.distributions(
                path=[str(self.install_target_path)]
            ):
                name = str(metadata.metadata.get("Name") or "")
                if canonicalize_name(name) == canonical_name:
                    return str(metadata.version or "")
        except Exception:
            return ""
        return ""

    def _record_rejection(self, *, node_id: str, missing_module: str, reason: str) -> None:
        key = f"module:{missing_module.lower()}"
        with self._condition:
            if key in self._status_by_distribution:
                return
            self._status_by_distribution[key] = {
                "status": "rejected",
                "completed_monotonic": time.monotonic(),
            }
        self._append_record(
            {
                "schema_version": "algoevolve.dependency_installation.v1",
                "timestamp": _utc_now(),
                "run_log_dir": str(self.run_log_dir),
                "node_id": str(node_id),
                "missing_module": missing_module,
                "distribution": "",
                "requirement": "",
                "selection_source": "rejected",
                "llm_declared_command": "",
                "executed_command": [],
                "python_executable": sys.executable,
                "install_target": str(self.install_target_path),
                "status": "rejected",
                "success": False,
                "exit_code": None,
                "duration_seconds": 0.0,
                "stdout_tail": "",
                "stderr_tail": reason,
            }
        )

    def _append_record(self, record: dict[str, Any]) -> None:
        with self._records_lock:
            self._records.append(record)
            line = json.dumps(record, ensure_ascii=False) + "\n"
            for path in self.detail_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            self._write_summaries()

    def _write_summaries(self) -> None:
        _atomic_write_json(
            self.local_summary_path,
            self._build_summary_payload(self._records),
        )
        if self.central_summary_path is not None:
            central_records = self._read_records(
                self.central_detail_path or self.local_detail_path
            )
            _atomic_write_json(
                self.central_summary_path,
                self._build_summary_payload(central_records),
            )

    def _build_summary_payload(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        attempted = [
            record
            for record in records
            if record.get("status") != "rejected"
        ]
        installed = [record for record in attempted if record.get("success")]
        failed = [record for record in attempted if not record.get("success")]
        return {
            "schema_version": "algoevolve.dependency_installation_summary.v1",
            "updated_at": _utc_now(),
            "python_executable": sys.executable,
            "install_target": str(self.install_target_path),
            "attempt_count": len(attempted),
            "installed_count": len(installed),
            "failed_count": len(failed),
            "rejected_count": len(records) - len(attempted),
            "installed_requirements": sorted(
                {
                    str(record.get("resolved_requirement") or record.get("requirement"))
                    for record in installed
                }
            ),
            "requirements_candidates": sorted(
                {
                    str(record.get("resolved_requirement") or record.get("requirement"))
                    for record in attempted
                    if record.get("requirement")
                }
            ),
            "records": [
                {
                    "timestamp": record.get("timestamp"),
                    "run_log_dir": record.get("run_log_dir"),
                    "node_id": record.get("node_id"),
                    "missing_module": record.get("missing_module"),
                    "distribution": record.get("distribution"),
                    "requirement": record.get("requirement"),
                    "status": record.get("status"),
                }
                for record in records
            ],
        }

    @staticmethod
    def _read_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def _load_previous_run_attempts(self) -> None:
        for record in self._read_records(self.local_detail_path):
            self._records.append(record)
            distribution = canonicalize_name(str(record.get("distribution") or ""))
            if not distribution:
                continue
            self._status_by_distribution[distribution] = {
                "status": "installed" if record.get("success") else "failed",
                "completed_monotonic": 0.0,
                "loaded_from_disk": True,
            }
