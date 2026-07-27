from __future__ import annotations

import csv
import ctypes
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import psutil

if os.name == "nt":
    from ctypes import wintypes


ACCELERATOR_VISIBILITY_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "ZE_AFFINITY_MASK",
    "ASCEND_RT_VISIBLE_DEVICES",
)


@dataclass(frozen=True)
class MemoryPressureAction:
    action: str
    child_pid: int | None
    observed_bytes: int
    limit_bytes: int


class ProcessTreeMemoryLimiter:
    """OS-backed total-memory limit inherited by a task's descendants."""

    backend = "unsupported"
    hard_limit = False
    total_process_tree = False
    over_limit_behavior = "observe_only"

    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = max(0, int(limit_bytes))

    def attach(self, root_pid: int) -> None:
        raise NotImplementedError

    def peak_memory_bytes(self) -> int:
        return 0

    def close(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "hard_limit": self.hard_limit,
            "total_process_tree": self.total_process_tree,
            "over_limit_behavior": self.over_limit_behavior,
            "limit_bytes": self.limit_bytes,
        }


if os.name == "nt":
    _SIZE_T = ctypes.c_size_t
    _ULONG_PTR = wintypes.WPARAM
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    class _LargeInteger(ctypes.Structure):
        _fields_ = [("QuadPart", ctypes.c_longlong)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", _LargeInteger),
            ("PerJobUserTimeLimit", _LargeInteger),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", _SIZE_T),
            ("MaximumWorkingSetSize", _SIZE_T),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", _ULONG_PTR),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", _SIZE_T),
            ("JobMemoryLimit", _SIZE_T),
            ("PeakProcessMemoryUsed", _SIZE_T),
            ("PeakJobMemoryUsed", _SIZE_T),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL

    class WindowsJobMemoryLimiter(ProcessTreeMemoryLimiter):
        backend = "windows_job_object"
        hard_limit = True
        total_process_tree = True
        over_limit_behavior = "deny_allocation_in_requesting_process"

        def __init__(self, limit_bytes: int) -> None:
            super().__init__(limit_bytes)
            self._job_handle = _kernel32.CreateJobObjectW(None, None)
            if not self._job_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            info = _ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = (
                _JOB_OBJECT_LIMIT_JOB_MEMORY | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            info.JobMemoryLimit = self.limit_bytes
            if not _kernel32.SetInformationJobObject(
                self._job_handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                error = ctypes.WinError(ctypes.get_last_error())
                self.close()
                raise error

        def attach(self, root_pid: int) -> None:
            process_handle = _kernel32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE,
                False,
                int(root_pid),
            )
            if not process_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not _kernel32.AssignProcessToJobObject(self._job_handle, process_handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                _kernel32.CloseHandle(process_handle)

        def peak_memory_bytes(self) -> int:
            if not self._job_handle:
                return 0
            info = _ExtendedLimitInformation()
            returned = wintypes.DWORD(0)
            if not _kernel32.QueryInformationJobObject(
                self._job_handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(info),
                ctypes.sizeof(info),
                ctypes.byref(returned),
            ):
                return 0
            return int(info.PeakJobMemoryUsed)

        def close(self) -> None:
            if self._job_handle:
                _kernel32.CloseHandle(self._job_handle)
                self._job_handle = None


def _linux_current_cgroup_dir() -> Path | None:
    if not sys.platform.startswith("linux"):
        return None
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").exists():
        return None
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.split("::", 1)[1].strip().lstrip("/")
                return (root / relative).resolve()
    except Exception:
        return None
    return None


def _linux_cgroup_parent() -> Path | None:
    configured = str(
        os.environ.get("ALGOEVOLVE_CGROUP_ROOT")
        or os.environ.get("MLEVOLVE_CGROUP_ROOT")
        or ""
    ).strip()
    candidates = [Path(configured).expanduser()] if configured else []
    current = _linux_current_cgroup_dir()
    if current is not None:
        candidates.append(current)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            controllers = (resolved / "cgroup.controllers").read_text(encoding="utf-8").split()
            if "memory" in controllers and os.access(resolved, os.W_OK):
                return resolved
        except Exception:
            continue
    return None


def linux_cgroup_v2_memory_available() -> bool:
    parent = _linux_cgroup_parent()
    if parent is None:
        return False
    try:
        enabled = (parent / "cgroup.subtree_control").read_text(encoding="utf-8").split()
        return "memory" in enabled or os.access(parent / "cgroup.subtree_control", os.W_OK)
    except Exception:
        return False


class LinuxCgroupMemoryLimiter(ProcessTreeMemoryLimiter):
    backend = "linux_cgroup_v2"
    hard_limit = True
    total_process_tree = True
    over_limit_behavior = "kernel_oom_in_requesting_cgroup_process"

    def __init__(self, limit_bytes: int) -> None:
        super().__init__(limit_bytes)
        parent = _linux_cgroup_parent()
        if parent is None:
            raise RuntimeError(
                "no writable cgroup v2 memory delegation; set ALGOEVOLVE_CGROUP_ROOT to a delegated cgroup"
            )
        subtree_control = parent / "cgroup.subtree_control"
        enabled = subtree_control.read_text(encoding="utf-8").split()
        if "memory" not in enabled:
            subtree_control.write_text("+memory", encoding="utf-8")
        self._path = parent / f"algoevolve-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        try:
            self._path.mkdir()
            (self._path / "memory.max").write_text(str(self.limit_bytes), encoding="utf-8")
            swap_max = self._path / "memory.swap.max"
            if swap_max.exists():
                swap_max.write_text("0", encoding="utf-8")
            oom_group = self._path / "memory.oom.group"
            if oom_group.exists():
                oom_group.write_text("0", encoding="utf-8")
        except Exception:
            self.close()
            raise

    def attach(self, root_pid: int) -> None:
        if self._path is None:
            raise RuntimeError("cgroup memory limiter is closed")
        (self._path / "cgroup.procs").write_text(str(int(root_pid)), encoding="utf-8")

    def peak_memory_bytes(self) -> int:
        if self._path is None:
            return 0
        for name in ("memory.peak", "memory.current"):
            try:
                return int((self._path / name).read_text(encoding="utf-8").strip())
            except Exception:
                continue
        return 0

    def close(self) -> None:
        path = getattr(self, "_path", None)
        if path is None:
            return
        try:
            procs_path = path / "cgroup.procs"
            pids = [int(item) for item in procs_path.read_text(encoding="utf-8").split() if item.isdigit()]
        except Exception:
            pids = []
        if pids:
            kill_path = path / "cgroup.kill"
            if kill_path.exists():
                try:
                    kill_path.write_text("1", encoding="utf-8")
                except Exception:
                    pass
            else:
                for pid in pids:
                    try:
                        psutil.Process(pid).kill()
                    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                        pass
        for _ in range(10):
            try:
                path.rmdir()
                break
            except OSError:
                time.sleep(0.05)
            except Exception:
                break
        self._path = None


def cpu_enforcement_capabilities(
    *,
    platform_name: str | None = None,
    affinity_supported: bool | None = None,
) -> dict[str, Any]:
    current_platform = platform_name or sys.platform
    if affinity_supported is None:
        affinity_supported = bool(hasattr(psutil.Process(), "cpu_affinity"))
    if current_platform.startswith("win") or current_platform.startswith("linux"):
        if affinity_supported:
            return {
                "backend": "process_affinity",
                "hard_limit": True,
                "total_process_tree": True,
                "exact_core_set": True,
            }
    return {
        "backend": "worker_and_thread_budget",
        "hard_limit": False,
        "total_process_tree": True,
        "exact_core_set": False,
    }


def cpu_limit_environment(
    cpu_cores: int,
    parallel_workers: int,
    *,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, str]:
    capabilities = capabilities or cpu_enforcement_capabilities()
    cores = max(1, int(cpu_cores))
    workers = max(1, min(int(parallel_workers or 1), cores))
    env = {
        "ALGOEVOLVE_CPU_LIMIT_MODE": str(capabilities["backend"]),
        "ALGOEVOLVE_CPU_CORE_BUDGET": str(cores),
        "MLEVOLVE_CPU_LIMIT_MODE": str(capabilities["backend"]),
        "MLEVOLVE_CPU_CORE_BUDGET": str(cores),
    }
    if bool(capabilities.get("hard_limit")):
        return env
    threads_per_worker = max(1, cores // workers)
    env.update(
        {
            "ALGOEVOLVE_CPU_WORKER_CAP": str(workers),
            "MLEVOLVE_CPU_WORKER_CAP": str(workers),
            "OMP_NUM_THREADS": str(threads_per_worker),
            "MKL_NUM_THREADS": str(threads_per_worker),
            "OPENBLAS_NUM_THREADS": str(threads_per_worker),
            "NUMEXPR_NUM_THREADS": str(threads_per_worker),
            "VECLIB_MAXIMUM_THREADS": str(threads_per_worker),
            "BLIS_NUM_THREADS": str(threads_per_worker),
            "RAYON_NUM_THREADS": str(threads_per_worker),
            "POLARS_MAX_THREADS": str(threads_per_worker),
            "LOKY_MAX_CPU_COUNT": str(threads_per_worker),
        }
    )
    return env


def memory_enforcement_capabilities(*, platform_name: str | None = None) -> dict[str, Any]:
    current_platform = platform_name or sys.platform
    if current_platform.startswith("win"):
        return {
            "backend": "windows_job_object",
            "hard_limit_supported": True,
            "total_process_tree": True,
            "over_limit_behavior": "deny_allocation_in_requesting_process",
            "whole_task_termination": False,
        }
    if current_platform.startswith("linux") and linux_cgroup_v2_memory_available():
        return {
            "backend": "linux_cgroup_v2",
            "hard_limit_supported": True,
            "total_process_tree": True,
            "over_limit_behavior": "kernel_oom_in_requesting_cgroup_process",
            "whole_task_termination": False,
        }
    return {
        "backend": "posix_rlimit_as_plus_child_guard",
        "hard_limit_supported": False,
        "total_process_tree": True,
        "over_limit_behavior": "per_process_allocation_failure_then_child_guard",
        "whole_task_termination": False,
    }


def create_process_tree_memory_limiter(limit_bytes: int) -> ProcessTreeMemoryLimiter | None:
    normalized = max(0, int(limit_bytes))
    if normalized <= 0:
        return None
    if os.name == "nt":
        return WindowsJobMemoryLimiter(normalized)
    if sys.platform.startswith("linux"):
        return LinuxCgroupMemoryLimiter(normalized)
    return None


def _run_command(args: list[str], *, timeout: float = 8.0) -> str:
    executable = shutil.which(args[0])
    if not executable:
        return ""
    try:
        proc = subprocess.run(
            [executable, *args[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(0.1, timeout),
            check=False,
        )
    except Exception:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _local_torch_runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "version": "",
        "cuda_available": False,
        "cuda_count": 0,
        "hip_version": "",
        "xpu_available": False,
        "xpu_count": 0,
        "mps_available": False,
        "python_executable": str(Path(sys.executable).resolve()),
        "probe_source": "service_runtime",
    }
    try:
        import torch

        info["version"] = str(getattr(torch, "__version__", ""))
        info["hip_version"] = str(getattr(getattr(torch, "version", None), "hip", "") or "")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
        xpu = getattr(torch, "xpu", None)
        info["xpu_available"] = bool(xpu is not None and xpu.is_available())
        info["xpu_count"] = int(xpu.device_count()) if info["xpu_available"] else 0
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        info["mps_available"] = bool(mps is not None and mps.is_available())
    except Exception as exc:
        info["error"] = str(exc)
    return info


_TORCH_RUNTIME_PROBE = r"""
import json
import sys

info = {
    "version": "",
    "cuda_available": False,
    "cuda_count": 0,
    "hip_version": "",
    "xpu_available": False,
    "xpu_count": 0,
    "mps_available": False,
    "python_executable": sys.executable,
    "probe_source": "configured_python",
    "cuda_devices": [],
    "xpu_devices": [],
    "ascend_devices": [],
}
try:
    import torch

    info["version"] = str(getattr(torch, "__version__", ""))
    info["hip_version"] = str(getattr(getattr(torch, "version", None), "hip", "") or "")
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
    for index in range(info["cuda_count"]):
        props = torch.cuda.get_device_properties(index)
        info["cuda_devices"].append({
            "index": index,
            "name": str(getattr(props, "name", "") or torch.cuda.get_device_name(index)),
            "uuid": str(getattr(props, "uuid", "") or ""),
            "memory_mb": int(getattr(props, "total_memory", 0) / (1024 * 1024)),
        })

    xpu = getattr(torch, "xpu", None)
    info["xpu_available"] = bool(xpu is not None and xpu.is_available())
    info["xpu_count"] = int(xpu.device_count()) if info["xpu_available"] else 0
    for index in range(info["xpu_count"]):
        props = xpu.get_device_properties(index)
        info["xpu_devices"].append({
            "index": index,
            "name": str(getattr(props, "name", "") or f"Intel XPU {index}"),
            "uuid": "",
            "memory_mb": int(getattr(props, "total_memory", 0) / (1024 * 1024)),
        })

    try:
        import torch_npu  # noqa: F401
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            for index in range(int(npu.device_count())):
                try:
                    name = str(npu.get_device_name(index))
                except Exception:
                    name = f"Ascend NPU {index}"
                info["ascend_devices"].append({
                    "index": index,
                    "name": name,
                    "uuid": "",
                    "memory_mb": 0,
                })
    except Exception:
        pass

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    info["mps_available"] = bool(mps is not None and mps.is_available())
except Exception as exc:
    info["error"] = str(exc)
print(json.dumps(info, ensure_ascii=True))
"""


def _configured_torch_runtime_info(python_executable: str) -> dict[str, Any]:
    requested = str(python_executable or "").strip()
    executable = requested
    if not executable:
        return _local_torch_runtime_info()
    if not Path(executable).expanduser().is_file():
        executable = shutil.which(executable) or executable
    base = {
        "version": "",
        "cuda_available": False,
        "cuda_count": 0,
        "hip_version": "",
        "xpu_available": False,
        "xpu_count": 0,
        "mps_available": False,
        "python_executable": str(executable),
        "probe_source": "configured_python",
    }
    try:
        probe_env = os.environ.copy()
        for name in ACCELERATOR_VISIBILITY_ENV_VARS:
            probe_env.pop(name, None)
        proc = subprocess.run(
            [executable, "-c", _TORCH_RUNTIME_PROBE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20.0,
            check=False,
            env=probe_env,
        )
    except Exception as exc:
        return {**base, "error": f"configured Python probe failed: {exc}"}
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return {**base, "error": f"configured Python probe exited with {proc.returncode}: {detail[-1000:]}"}
    payload = None
    for line in reversed((proc.stdout or "").splitlines()):
        try:
            candidate = json.loads(line.strip())
        except Exception:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        exc = "no JSON object found"
        return {**base, "error": f"configured Python returned invalid probe output: {exc}"}
    return {**base, **payload}


def _torch_runtime_info(python_executable: str | None = None) -> dict[str, Any]:
    if python_executable is None or not str(python_executable).strip():
        return _local_torch_runtime_info()
    return _configured_torch_runtime_info(str(python_executable))


def _nvidia_devices(torch_info: dict[str, Any]) -> list[dict[str, Any]]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    runtime_available = bool(torch_info.get("cuda_available")) and not bool(torch_info.get("hip_version"))
    devices: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) < 4:
            continue
        try:
            index = int(row[0].strip())
            memory_mb = int(float(row[3].strip()))
        except (TypeError, ValueError):
            continue
        devices.append(
            {
                "id": f"cuda:{index}",
                "backend": "cuda",
                "index": index,
                "name": row[1].strip() or f"NVIDIA GPU {index}",
                "vendor": "NVIDIA",
                "uuid": row[2].strip(),
                "memory_mb": memory_mb,
                "visibility_env": "CUDA_VISIBLE_DEVICES",
                "visibility_supported": True,
                "runtime_available": runtime_available,
                "source": "nvidia-smi",
            }
        )
    return devices


def _torch_cuda_or_rocm_devices(torch_info: dict[str, Any]) -> list[dict[str, Any]]:
    count = int(torch_info.get("cuda_count") or 0)
    if count <= 0:
        return []
    is_rocm = bool(torch_info.get("hip_version"))
    backend = "rocm" if is_rocm else "cuda"
    vendor = "AMD" if is_rocm else "NVIDIA"
    visibility_env = "HIP_VISIBLE_DEVICES" if is_rocm else "CUDA_VISIBLE_DEVICES"
    runtime_rows = torch_info.get("cuda_devices")
    if isinstance(runtime_rows, list):
        return [
            {
                "id": f"{backend}:{int(row.get('index') or 0)}",
                "backend": backend,
                "index": int(row.get("index") or 0),
                "name": str(row.get("name") or f"{vendor} GPU"),
                "vendor": vendor,
                "uuid": str(row.get("uuid") or ""),
                "memory_mb": int(row.get("memory_mb") or 0),
                "visibility_env": visibility_env,
                "visibility_supported": True,
                "runtime_available": True,
                "source": "configured_python.torch",
            }
            for row in runtime_rows
            if isinstance(row, dict)
        ]
    devices: list[dict[str, Any]] = []
    try:
        import torch

        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "id": f"{backend}:{index}",
                    "backend": backend,
                    "index": index,
                    "name": str(getattr(props, "name", "") or torch.cuda.get_device_name(index)),
                    "vendor": vendor,
                    "uuid": str(getattr(props, "uuid", "") or ""),
                    "memory_mb": int(getattr(props, "total_memory", 0) / (1024 * 1024)),
                    "visibility_env": visibility_env,
                    "visibility_supported": True,
                    "runtime_available": True,
                    "source": "torch",
                }
            )
    except Exception:
        return []
    return devices


def _torch_xpu_devices(torch_info: dict[str, Any]) -> list[dict[str, Any]]:
    if not torch_info.get("xpu_available"):
        return []
    runtime_rows = torch_info.get("xpu_devices")
    if isinstance(runtime_rows, list):
        return [
            {
                "id": f"xpu:{int(row.get('index') or 0)}",
                "backend": "xpu",
                "index": int(row.get("index") or 0),
                "name": str(row.get("name") or "Intel XPU"),
                "vendor": "Intel",
                "uuid": str(row.get("uuid") or ""),
                "memory_mb": int(row.get("memory_mb") or 0),
                "visibility_env": "ZE_AFFINITY_MASK",
                "visibility_supported": True,
                "runtime_available": True,
                "source": "configured_python.torch.xpu",
            }
            for row in runtime_rows
            if isinstance(row, dict)
        ]
    devices: list[dict[str, Any]] = []
    try:
        import torch

        for index in range(int(torch_info.get("xpu_count") or 0)):
            props = torch.xpu.get_device_properties(index)
            devices.append(
                {
                    "id": f"xpu:{index}",
                    "backend": "xpu",
                    "index": index,
                    "name": str(getattr(props, "name", "") or f"Intel XPU {index}"),
                    "vendor": "Intel",
                    "uuid": "",
                    "memory_mb": int(getattr(props, "total_memory", 0) / (1024 * 1024)),
                    "visibility_env": "ZE_AFFINITY_MASK",
                    "visibility_supported": True,
                    "runtime_available": True,
                    "source": "torch.xpu",
                }
            )
    except Exception:
        return []
    return devices


def _torch_ascend_devices(torch_info: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    runtime_rows = (torch_info or {}).get("ascend_devices")
    if isinstance(runtime_rows, list):
        return [
            {
                "id": f"ascend:{int(row.get('index') or 0)}",
                "backend": "ascend",
                "index": int(row.get("index") or 0),
                "name": str(row.get("name") or "Ascend NPU"),
                "vendor": "Huawei",
                "uuid": str(row.get("uuid") or ""),
                "memory_mb": int(row.get("memory_mb") or 0),
                "visibility_env": "ASCEND_RT_VISIBLE_DEVICES",
                "visibility_supported": True,
                "runtime_available": True,
                "source": "configured_python.torch_npu",
            }
            for row in runtime_rows
            if isinstance(row, dict)
        ]
    try:
        import torch
        import torch_npu  # noqa: F401

        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            return []
        devices: list[dict[str, Any]] = []
        for index in range(int(npu.device_count())):
            try:
                name = str(npu.get_device_name(index))
            except Exception:
                name = f"Ascend NPU {index}"
            devices.append(
                {
                    "id": f"ascend:{index}",
                    "backend": "ascend",
                    "index": index,
                    "name": name,
                    "vendor": "Huawei",
                    "uuid": "",
                    "memory_mb": 0,
                    "visibility_env": "ASCEND_RT_VISIBLE_DEVICES",
                    "visibility_supported": True,
                    "runtime_available": True,
                    "source": "torch_npu",
                }
            )
        return devices
    except Exception:
        return []


def _macos_mps_devices(
    torch_info: dict[str, Any],
    *,
    platform_name: str | None = None,
) -> list[dict[str, Any]]:
    if (platform_name or sys.platform) != "darwin":
        return []
    output = _run_command(["system_profiler", "SPDisplaysDataType", "-json"], timeout=15.0)
    rows: list[dict[str, Any]] = []
    if output:
        try:
            payload = json.loads(output)
            raw_rows = payload.get("SPDisplaysDataType") if isinstance(payload, dict) else []
            rows = [item for item in raw_rows if isinstance(item, dict)]
        except Exception:
            rows = []
    name = "Apple Metal GPU"
    if rows:
        first = rows[0]
        name = str(
            first.get("sppci_model")
            or first.get("_name")
            or first.get("spdisplays_device-id")
            or name
        )
    if not rows and not torch_info.get("mps_available"):
        return []
    return [
        {
            "id": "mps:0",
            "backend": "mps",
            "index": 0,
            "name": name,
            "vendor": "Apple",
            "uuid": "",
            "memory_mb": 0,
            "visibility_env": "",
            "visibility_supported": False,
            "runtime_available": bool(torch_info.get("mps_available")),
            "source": "system_profiler" if rows else "torch.mps",
        }
    ]


def accelerator_visibility_capabilities(devices: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [item for item in devices if isinstance(item, dict)]
    isolatable = [str(item.get("id")) for item in rows if bool(item.get("visibility_supported"))]
    non_isolatable = [str(item.get("id")) for item in rows if not bool(item.get("visibility_supported"))]
    return {
        "backend": "backend_visibility_environment",
        "isolatable_device_ids": isolatable,
        "non_isolatable_device_ids": non_isolatable,
        "mode_none_fully_enforced": not non_isolatable,
        "exclusive_reservation": False,
        "vram_quota": False,
    }


def detect_resource_inventory(python_executable: str | None = None) -> dict[str, Any]:
    torch_info = _torch_runtime_info(python_executable)
    devices = _nvidia_devices(torch_info)
    known_ids = {str(item.get("id")) for item in devices}
    runtime_devices = (
        _torch_cuda_or_rocm_devices(torch_info)
        + _torch_xpu_devices(torch_info)
        + _torch_ascend_devices(torch_info)
        + _macos_mps_devices(torch_info)
    )
    for item in runtime_devices:
        if str(item.get("id")) not in known_ids:
            devices.append(item)
            known_ids.add(str(item.get("id")))

    cpu_ids = available_cpu_ids()
    virtual_memory = psutil.virtual_memory()
    cpu_capabilities = cpu_enforcement_capabilities()
    sorted_devices = sorted(devices, key=lambda item: (str(item.get("backend")), int(item.get("index") or 0)))
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_platform": sys.platform,
        },
        "cpu": {
            "logical_count": int(psutil.cpu_count(logical=True) or len(cpu_ids) or 1),
            "physical_count": int(psutil.cpu_count(logical=False) or 0),
            "available_ids": cpu_ids,
            "affinity_supported": bool(cpu_capabilities.get("hard_limit")),
            "enforcement": cpu_capabilities,
        },
        "memory": {
            "total_bytes": int(virtual_memory.total),
            "total_gb": round(float(virtual_memory.total) / (1024**3), 2),
            "enforcement": memory_enforcement_capabilities(),
        },
        "devices": sorted_devices,
        "accelerator": accelerator_visibility_capabilities(sorted_devices),
        "torch": torch_info,
    }


def accelerator_visibility_env(mode: str, device_ids: Iterable[str]) -> dict[str, str]:
    normalized_mode = str(mode or "all").strip().lower()
    selected = sorted({str(item).strip().lower() for item in device_ids if str(item).strip()})
    metadata = {
        "ALGOEVOLVE_ACCELERATOR_MODE": normalized_mode,
        "ALGOEVOLVE_VISIBLE_ACCELERATOR_IDS": json.dumps(selected, ensure_ascii=True),
        "MLEVOLVE_ACCELERATOR_MODE": normalized_mode,
        "MLEVOLVE_VISIBLE_ACCELERATOR_IDS": json.dumps(selected, ensure_ascii=True),
    }
    if normalized_mode == "all":
        return metadata

    env = {name: "" for name in ACCELERATOR_VISIBILITY_ENV_VARS}
    env.update(metadata)
    if normalized_mode == "none":
        return env

    groups: dict[str, list[str]] = {}
    for device_id in selected:
        backend, sep, raw_index = device_id.partition(":")
        if not sep or not raw_index.isdigit():
            continue
        groups.setdefault(backend, []).append(raw_index)
    if groups.get("cuda"):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(groups["cuda"])
    if groups.get("rocm"):
        indexes = ",".join(groups["rocm"])
        env["HIP_VISIBLE_DEVICES"] = indexes
        env["ROCR_VISIBLE_DEVICES"] = indexes
    if groups.get("xpu"):
        env["ZE_AFFINITY_MASK"] = ",".join(groups["xpu"])
    if groups.get("ascend"):
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(groups["ascend"])
    return env


def validate_accelerator_selection(mode: str, device_ids: Iterable[str], inventory: dict[str, Any]) -> list[str]:
    normalized_mode = str(mode or "all").strip().lower()
    if normalized_mode not in {"all", "selected", "none"}:
        return [f"unsupported accelerator_mode: {mode}"]
    if normalized_mode != "selected":
        return []
    known = {str(item.get("id")): item for item in inventory.get("devices", []) if isinstance(item, dict)}
    selected = [str(item).strip().lower() for item in device_ids if str(item).strip()]
    if not selected:
        return ["accelerator_mode=selected requires at least one device id"]
    errors: list[str] = []
    for device_id in selected:
        item = known.get(device_id)
        if item is None:
            errors.append(f"accelerator device is not present: {device_id}")
        elif not bool(item.get("visibility_supported", True)):
            errors.append(f"per-task visibility is not supported for device: {device_id}")
    return errors


def available_cpu_ids() -> list[int]:
    process = psutil.Process()
    if hasattr(process, "cpu_affinity"):
        try:
            ids = [int(item) for item in process.cpu_affinity()]
            if ids:
                return sorted(ids)
        except Exception:
            pass
    return list(range(max(1, int(psutil.cpu_count(logical=True) or os.cpu_count() or 1))))


def choose_cpu_ids(cpu_cores: int) -> list[int]:
    available = available_cpu_ids()
    requested = max(1, int(cpu_cores))
    if requested > len(available):
        raise ValueError(f"requested {requested} CPU cores, but only {len(available)} are available")
    return available[:requested]


def _process_tree(root_pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(root_pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    unique: dict[int, psutil.Process] = {proc.pid: proc for proc in processes}
    return list(unique.values())


def apply_process_tree_cpu_affinity(root_pid: int, cpu_ids: Iterable[int]) -> list[str]:
    target = sorted({int(item) for item in cpu_ids})
    if not target:
        return ["empty CPU affinity set"]
    if not bool(cpu_enforcement_capabilities().get("hard_limit")):
        return []
    errors: list[str] = []
    for process in _process_tree(root_pid):
        if not hasattr(process, "cpu_affinity"):
            errors.append(f"pid={process.pid}: CPU affinity is unsupported")
            continue
        try:
            process.cpu_affinity(target)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except Exception as exc:
            errors.append(f"pid={process.pid}: {exc}")
    return errors


def process_tree_memory_bytes(root_pid: int) -> int:
    total = 0
    for process in _process_tree(root_pid):
        try:
            full = process.memory_full_info()
            value = int(getattr(full, "uss", 0) or 0)
            if value <= 0:
                value = int(process.memory_info().rss)
            total += max(0, value)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            continue
    return total


def _same_process_invocation(left: psutil.Process, right: psutil.Process) -> bool:
    """Return whether a child transparently re-executes its parent command."""
    try:
        left_args = left.cmdline()
        right_args = right.cmdline()
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return False
    return bool(left_args and right_args and left_args[1:] == right_args[1:])


def _resolve_controller_process(root: psutil.Process) -> psutil.Process:
    """Skip Windows/venv launcher processes while preserving the real controller."""
    controller = root
    visited = {root.pid}
    while True:
        try:
            children = controller.children(recursive=False)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            break
        if len(children) != 1:
            break
        child = children[0]
        if child.pid in visited or not _same_process_invocation(controller, child):
            break
        visited.add(child.pid)
        controller = child
    return controller


def relieve_process_tree_memory_pressure(root_pid: int, limit_bytes: int) -> MemoryPressureAction:
    """Fallback guard that preserves the task controller and stops one heavy child."""
    observed = process_tree_memory_bytes(root_pid)
    normalized_limit = max(0, int(limit_bytes))
    if normalized_limit <= 0 or observed <= normalized_limit:
        return MemoryPressureAction("none", None, observed, normalized_limit)
    try:
        root = psutil.Process(root_pid)
        controller = _resolve_controller_process(root)
        direct_children = controller.children(recursive=False)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        direct_children = []
    candidates: list[tuple[int, psutil.Process]] = []
    for child in direct_children:
        memory = process_tree_memory_bytes(child.pid)
        if memory > 0:
            candidates.append((memory, child))
    if not candidates:
        return MemoryPressureAction("controller_over_limit", None, observed, normalized_limit)
    _memory, child = max(candidates, key=lambda item: item[0])
    child_pid = child.pid
    terminate_process_tree(child_pid, grace_seconds=1.0)
    return MemoryPressureAction("terminated_child", child_pid, observed, normalized_limit)


def terminate_process_tree(root_pid: int, *, grace_seconds: float = 3.0) -> None:
    processes = _process_tree(root_pid)
    if not processes:
        return
    root = next((proc for proc in processes if proc.pid == root_pid), None)
    children = [proc for proc in processes if proc.pid != root_pid]
    for process in reversed(children):
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            pass
    if root is not None:
        try:
            root.terminate()
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            pass
    _gone, alive = psutil.wait_procs(processes, timeout=max(0.0, grace_seconds))
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=1.0)


def format_bytes(value: int) -> str:
    amount = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} TiB"
