from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import yaml
from fastapi import HTTPException

import service_api
from engine.executor import Interpreter, memory_limited_subprocess_command
from service_api import (
    JobStore,
    StartAlgoEvolveRequest,
    TaskResourceLimits,
    _memory_child_guard_threshold,
    _monitor_task_resources,
    _resolve_resource_limits,
)
from utils.resource_limits import (
    _macos_mps_devices,
    _nvidia_devices,
    accelerator_visibility_capabilities,
    accelerator_visibility_env,
    apply_process_tree_cpu_affinity,
    choose_cpu_ids,
    cpu_enforcement_capabilities,
    cpu_limit_environment,
    create_process_tree_memory_limiter,
    detect_resource_inventory,
    memory_enforcement_capabilities,
    validate_accelerator_selection,
)


def test_accelerator_visibility_env_supports_all_selected_and_none() -> None:
    all_env = accelerator_visibility_env("all", [])
    assert "CUDA_VISIBLE_DEVICES" not in all_env

    selected = accelerator_visibility_env("selected", ["cuda:1", "cuda:0", "xpu:2"])
    assert selected["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert selected["ZE_AFFINITY_MASK"] == "2"
    assert selected["HIP_VISIBLE_DEVICES"] == ""

    hidden = accelerator_visibility_env("none", [])
    assert hidden["CUDA_VISIBLE_DEVICES"] == ""
    assert hidden["ASCEND_RT_VISIBLE_DEVICES"] == ""


def test_nvidia_smi_rows_are_exposed_even_when_torch_is_cpu_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "utils.resource_limits._run_command",
        lambda *_args, **_kwargs: "0, NVIDIA Demo, GPU-demo, 16384",
    )
    devices = _nvidia_devices({"cuda_available": False, "hip_version": ""})

    assert devices[0]["id"] == "cuda:0"
    assert devices[0]["memory_mb"] == 16384
    assert devices[0]["runtime_available"] is False


def test_macos_gpu_is_detected_without_mps_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "utils.resource_limits._run_command",
        lambda *_args, **_kwargs: json.dumps(
            {"SPDisplaysDataType": [{"sppci_model": "Apple M4 Max"}]}
        ),
    )

    devices = _macos_mps_devices({"mps_available": False}, platform_name="darwin")

    assert devices[0]["id"] == "mps:0"
    assert devices[0]["name"] == "Apple M4 Max"
    assert devices[0]["runtime_available"] is False
    assert devices[0]["visibility_supported"] is False


def test_selected_accelerator_ids_must_exist_and_support_visibility() -> None:
    inventory = {
        "devices": [
            {"id": "cuda:0", "visibility_supported": True},
            {"id": "mps:0", "visibility_supported": False},
        ]
    }
    assert validate_accelerator_selection("selected", ["cuda:0"], inventory) == []
    assert "not present" in validate_accelerator_selection("selected", ["cuda:9"], inventory)[0]
    assert "not supported" in validate_accelerator_selection("selected", ["mps:0"], inventory)[0]


def test_accelerator_capabilities_report_non_isolatable_mps() -> None:
    capabilities = accelerator_visibility_capabilities(
        [
            {"id": "cuda:0", "visibility_supported": True},
            {"id": "mps:0", "visibility_supported": False},
        ]
    )

    assert capabilities["isolatable_device_ids"] == ["cuda:0"]
    assert capabilities["non_isolatable_device_ids"] == ["mps:0"]
    assert capabilities["mode_none_fully_enforced"] is False


def test_cross_platform_cpu_capability_matrix_and_macos_budget() -> None:
    windows = cpu_enforcement_capabilities(platform_name="win32", affinity_supported=True)
    linux = cpu_enforcement_capabilities(platform_name="linux", affinity_supported=True)
    macos = cpu_enforcement_capabilities(platform_name="darwin", affinity_supported=False)
    macos_env = cpu_limit_environment(4, 8, capabilities=macos)

    assert windows["backend"] == "process_affinity"
    assert windows["hard_limit"] is True
    assert linux["backend"] == "process_affinity"
    assert macos["backend"] == "worker_and_thread_budget"
    assert macos["hard_limit"] is False
    assert macos_env["ALGOEVOLVE_CPU_WORKER_CAP"] == "4"
    assert macos_env["VECLIB_MAXIMUM_THREADS"] == "1"


def test_cross_platform_memory_capability_matrix(monkeypatch) -> None:
    windows = memory_enforcement_capabilities(platform_name="win32")
    macos = memory_enforcement_capabilities(platform_name="darwin")
    monkeypatch.setattr("utils.resource_limits.linux_cgroup_v2_memory_available", lambda: True)
    linux_hard = memory_enforcement_capabilities(platform_name="linux")
    monkeypatch.setattr("utils.resource_limits.linux_cgroup_v2_memory_available", lambda: False)
    linux_fallback = memory_enforcement_capabilities(platform_name="linux")

    assert windows["backend"] == "windows_job_object"
    assert windows["hard_limit_supported"] is True
    assert linux_hard["backend"] == "linux_cgroup_v2"
    assert linux_hard["hard_limit_supported"] is True
    assert linux_fallback["backend"] == "posix_rlimit_as_plus_child_guard"
    assert macos["backend"] == "posix_rlimit_as_plus_child_guard"
    assert macos["whole_task_termination"] is False


def test_soft_cpu_backend_does_not_report_affinity_setup_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "utils.resource_limits.cpu_enforcement_capabilities",
        lambda: {"backend": "worker_and_thread_budget", "hard_limit": False},
    )

    assert apply_process_tree_cpu_affinity(999999, [0, 1]) == []


def test_runtime_inventory_reports_platform_and_enforcement_backends() -> None:
    inventory = detect_resource_inventory()

    assert inventory["platform"]["system"]
    assert inventory["cpu"]["enforcement"]["backend"]
    assert inventory["memory"]["enforcement"]["backend"]
    assert "mode_none_fully_enforced" in inventory["accelerator"]


def test_resource_inventory_endpoint_returns_detected_inventory(monkeypatch) -> None:
    expected = {
        "cpu": {"logical_count": 4, "physical_count": 2, "available_ids": [0, 1, 2, 3]},
        "memory": {"total_bytes": 8 * 1024**3, "total_gb": 8.0},
        "devices": [],
        "torch": {"version": "test"},
    }
    captured: dict[str, object] = {}

    def fake_detect(python_executable=None):
        captured["python_executable"] = python_executable
        return expected

    monkeypatch.setattr(service_api, "detect_resource_inventory", fake_detect)

    assert service_api.resource_inventory("/configured/python") == expected
    assert captured["python_executable"] == "/configured/python"


def test_start_job_rejects_unknown_selected_accelerator(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        service_api,
        "detect_resource_inventory",
        lambda _python_executable=None: {
            "cpu": {"available_ids": [0, 1]},
            "memory": {"total_bytes": 8 * 1024**3, "total_gb": 8.0},
            "devices": [{"id": "cuda:0", "visibility_supported": True}],
            "torch": {},
        },
    )
    request = StartAlgoEvolveRequest(
        task_id="unknown-device",
        args=["exp_name=unknown-device"],
        log_dir=str(tmp_path / "logs"),
        workspace_dir=str(tmp_path / "workspace"),
        resources=TaskResourceLimits(
            cpu_cores=1,
            accelerator_mode="selected",
            accelerator_device_ids=["cuda:9"],
        ),
    )

    with pytest.raises(HTTPException, match="accelerator device is not present"):
        service_api.start_job(request)


def test_inventory_uses_configured_python_for_torch_probe(monkeypatch) -> None:
    expected = {
        "version": "configured-torch",
        "cuda_available": True,
        "cuda_count": 1,
        "hip_version": "",
        "xpu_available": False,
        "xpu_count": 0,
        "mps_available": False,
        "python_executable": "/configured/python",
        "probe_source": "configured_python",
        "cuda_devices": [],
    }
    monkeypatch.setattr("utils.resource_limits._configured_torch_runtime_info", lambda value: expected)
    monkeypatch.setattr("utils.resource_limits._run_command", lambda *_args, **_kwargs: "0, RTX Demo, GPU-demo, 16384")

    inventory = detect_resource_inventory("/configured/python")

    assert inventory["torch"]["version"] == "configured-torch"
    assert inventory["torch"]["python_executable"] == "/configured/python"
    assert inventory["devices"][0]["runtime_available"] is True


def test_service_resource_request_reads_task_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "resources": {
                    "cpu_cores": 3,
                    "memory_limit_gb": 6.5,
                    "accelerator_mode": "selected",
                    "accelerator_device_ids": ["cuda:0"],
                }
            }
        ),
        encoding="utf-8",
    )
    request = StartAlgoEvolveRequest(
        task_id="resource-yaml",
        log_dir="logs",
        workspace_dir="workspace",
        config_path=str(config_path),
    )
    limits = _resolve_resource_limits(request)

    assert limits.cpu_cores == 3
    assert limits.memory_limit_gb == 6.5
    assert limits.accelerator_device_ids == ["cuda:0"]


def test_parallel_search_can_share_a_smaller_cpu_set(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        start_cpu_id="0",
        cpu_number="2",
        agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=4)),
    )
    interpreter = Interpreter(tmp_path, timeout=10, cfg=cfg)

    assert interpreter.max_parallel_run == 4
    assert len(interpreter._available_cpus()) <= 2


def test_soft_cpu_worker_cap_limits_parallel_execution_slots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALGOEVOLVE_CPU_WORKER_CAP", "2")
    cfg = SimpleNamespace(
        start_cpu_id="0",
        cpu_number="4",
        agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=6)),
    )

    interpreter = Interpreter(tmp_path, timeout=10, cfg=cfg)

    assert interpreter.max_parallel_run == 2


def test_posix_memory_fallback_wraps_generated_node_only(tmp_path: Path) -> None:
    command = ["python", "runfile.py"]
    guard = tmp_path / "resource_guard.py"
    environment = {
        "ALGOEVOLVE_MEMORY_ENFORCEMENT_MODE": "posix_rlimit_as_plus_child_guard",
        "ALGOEVOLVE_MEMORY_LIMIT_BYTES": "1073741824",
    }

    wrapped = memory_limited_subprocess_command(
        command,
        platform_name="darwin",
        environment=environment,
        python_executable="python3",
        guard_path=guard,
    )
    windows = memory_limited_subprocess_command(
        command,
        platform_name="win32",
        environment=environment,
        python_executable="python3",
        guard_path=guard,
    )

    assert wrapped == [
        "python3",
        str(guard),
        "--memory-bytes",
        "1073741824",
        "--",
        *command,
    ]
    assert windows == command


def test_run_job_without_config_path_applies_task_environment(tmp_path: Path, monkeypatch) -> None:
    local_store = JobStore()
    monkeypatch.setattr(service_api, "store", local_store)
    monkeypatch.delenv("ALGOEVOLVE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("PYTHONFAULTHANDLER", raising=False)
    cpu_ids = choose_cpu_ids(1)
    workdir = tmp_path / "service-workdir"
    log_dir = tmp_path / "service-logs"
    workspace_dir = tmp_path / "service-workspace"
    workdir.mkdir()
    (workdir / "run.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import time",
                "import psutil",
                "time.sleep(0.5)",
                "print(json.dumps({",
                "    'affinity': psutil.Process().cpu_affinity(),",
                "    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),",
                "    'config_path_present': bool(os.environ.get('ALGOEVOLVE_CONFIG_PATH')),",
                "    'python_faulthandler': os.environ.get('PYTHONFAULTHANDLER'),",
                "}))",
            ]
        ),
        encoding="utf-8",
    )
    limits = TaskResourceLimits(cpu_cores=1, memory_limit_gb=0, accelerator_mode="none")
    request = StartAlgoEvolveRequest(
        task_id="service-environment",
        python_executable=sys.executable,
        working_dir=str(workdir),
        log_dir=str(log_dir),
        workspace_dir=str(workspace_dir),
        resources=limits,
    )
    job = local_store.create(
        request.task_id,
        str(log_dir),
        str(workspace_dir),
        resource_limits=limits.model_dump(),
        assigned_cpu_ids=cpu_ids,
    )

    service_api._run_job(job.job_id, request, log_dir, workspace_dir, "20260710_000000")

    status = local_store.status(job.job_id)
    payload = json.loads(status.stdout_tail.strip())
    assert status.status == "completed"
    assert payload["affinity"] == cpu_ids
    assert payload["cuda_visible_devices"] == ""
    assert payload["config_path_present"] is False
    assert payload["python_faulthandler"] == "1"
    assert (log_dir / "resource_usage.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object test")
def test_windows_job_memory_limit_denies_excess_allocation_without_killing_process() -> None:
    limiter = create_process_tree_memory_limiter(96 * 1024 * 1024)
    assert limiter is not None
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import time; time.sleep(0.6);\n"
                "try:\n"
                "    payload = bytearray(256 * 1024 * 1024)\n"
                "    print('allocated')\n"
                "except MemoryError:\n"
                "    print('memory_error')\n"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        limiter.attach(proc.pid)
        out, err = proc.communicate(timeout=10)
        peak_memory = limiter.peak_memory_bytes()
    finally:
        limiter.close()
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode == 0
    assert out.strip() == "memory_error"
    assert err.strip() == ""
    assert peak_memory > 0
    assert limiter.describe()["over_limit_behavior"] == "deny_allocation_in_requesting_process"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration test")
def test_service_memory_limit_keeps_controller_alive_after_child_oom(tmp_path: Path, monkeypatch) -> None:
    local_store = JobStore()
    monkeypatch.setattr(service_api, "store", local_store)
    cpu_ids = choose_cpu_ids(1)
    workdir = tmp_path / "memory-service-workdir"
    log_dir = tmp_path / "memory-service-logs"
    workspace_dir = tmp_path / "memory-service-workspace"
    workdir.mkdir()
    child_code = (
        "try:\n"
        "    payload = bytearray(256 * 1024 * 1024)\n"
        "    print('allocated')\n"
        "except MemoryError:\n"
        "    print('memory_error')\n"
    )
    (workdir / "run.py").write_text(
        "\n".join(
            [
                "import subprocess",
                "import sys",
                "import time",
                "time.sleep(0.8)",
                f"child = subprocess.run([sys.executable, '-c', {child_code!r}], capture_output=True, text=True)",
                "print(child.stdout.strip())",
                "print('controller_continued')",
            ]
        ),
        encoding="utf-8",
    )
    limits = TaskResourceLimits(cpu_cores=1, memory_limit_gb=0.12, monitor_interval_seconds=0.1)
    request = StartAlgoEvolveRequest(
        task_id="service-memory-limit",
        python_executable=sys.executable,
        working_dir=str(workdir),
        log_dir=str(log_dir),
        workspace_dir=str(workspace_dir),
        resources=limits,
    )
    job = local_store.create(
        request.task_id,
        str(log_dir),
        str(workspace_dir),
        resource_limits=limits.model_dump(),
        assigned_cpu_ids=cpu_ids,
    )

    service_api._run_job(job.job_id, request, log_dir, workspace_dir, "20260710_000001")

    status = local_store.status(job.job_id)
    usage = json.loads((log_dir / "resource_usage.json").read_text(encoding="utf-8"))
    assert status.status == "completed"
    assert "memory_error" in status.stdout_tail
    assert "controller_continued" in status.stdout_tail
    assert status.resource_violation is None
    assert usage["memory_enforcement"]["backend"] == "windows_job_object"
    assert usage["memory_enforcement"]["hard_limit"] is True


def test_memory_monitor_observes_limit_without_terminating_task(tmp_path: Path, monkeypatch) -> None:
    local_store = JobStore()
    monkeypatch.setattr(service_api, "store", local_store)
    cpu_ids = choose_cpu_ids(1)
    limits = TaskResourceLimits(cpu_cores=1, memory_limit_gb=0.02, monitor_interval_seconds=0.1)
    job = local_store.create(
        "memory-test",
        str(tmp_path / "logs"),
        str(tmp_path / "workspace"),
        resource_limits=limits.model_dump(),
        assigned_cpu_ids=cpu_ids,
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; payload = bytearray(80 * 1024 * 1024); print(len(payload), flush=True); time.sleep(1)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    local_store.set_process(job.job_id, proc)
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_task_resources,
        args=(job.job_id, proc, limits, cpu_ids, stop_event, True),
        daemon=True,
    )
    monitor.start()
    try:
        proc.wait(timeout=5)
        monitor.join(timeout=5)
    finally:
        if proc.poll() is None:
            psutil.Process(proc.pid).kill()

    status = local_store.status(job.job_id)
    assert not monitor.is_alive()
    assert proc.returncode == 0
    assert status.resource_violation is None
    assert status.peak_memory_bytes > int(0.02 * (1024**3))


def test_hard_memory_limit_reserves_controller_headroom(monkeypatch, tmp_path: Path) -> None:
    memory_limit = 8 * 1024**3
    guard_threshold = _memory_child_guard_threshold(memory_limit, True)
    assert guard_threshold == memory_limit - memory_limit // 10

    local_store = JobStore()
    monkeypatch.setattr(service_api, "store", local_store)
    job = local_store.create(
        "hard-limit-guard",
        str(tmp_path / "logs"),
        str(tmp_path / "workspace"),
    )
    stop_event = threading.Event()
    observed: dict[str, int] = {}

    class FakeProcess:
        pid = 43210

        @staticmethod
        def poll():
            return None

    def fake_relieve(_root_pid: int, threshold: int):
        observed["threshold"] = threshold
        stop_event.set()
        return SimpleNamespace(
            action="terminated_child",
            child_pid=43211,
            observed_bytes=threshold + 1,
            limit_bytes=threshold,
        )

    monkeypatch.setattr(service_api, "apply_process_tree_cpu_affinity", lambda *_args: [])
    monkeypatch.setattr(service_api, "process_tree_memory_bytes", lambda _pid: guard_threshold + 1)
    monkeypatch.setattr(service_api, "relieve_process_tree_memory_pressure", fake_relieve)
    limits = TaskResourceLimits(memory_limit_gb=8, monitor_interval_seconds=0.1)

    _monitor_task_resources(job.job_id, FakeProcess(), limits, [], stop_event, True)

    status = local_store.status(job.job_id)
    assert observed["threshold"] == guard_threshold
    assert status.resource_warning is not None
    assert "AlgoEvolve controller continues" in status.resource_warning
    assert "configured limit=8.00 GiB" in status.resource_warning


def test_memory_fallback_stops_child_but_preserves_controller(tmp_path: Path, monkeypatch) -> None:
    local_store = JobStore()
    monkeypatch.setattr(service_api, "store", local_store)
    cpu_ids = choose_cpu_ids(1)
    limits = TaskResourceLimits(cpu_cores=1, memory_limit_gb=0.03, monitor_interval_seconds=0.1)
    job = local_store.create(
        "child-memory-test",
        str(tmp_path / "logs"),
        str(tmp_path / "workspace"),
        resource_limits=limits.model_dump(),
        assigned_cpu_ids=cpu_ids,
    )
    child_pid_path = tmp_path / "child.pid"
    child_code = "import time; payload = bytearray(80 * 1024 * 1024); time.sleep(30)"
    parent_code = (
        "import pathlib, subprocess, sys; "
        f"proc = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(proc.pid)); "
        "print(proc.wait(), flush=True)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    local_store.set_process(job.job_id, proc)
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_task_resources,
        args=(job.job_id, proc, limits, cpu_ids, stop_event),
        daemon=True,
    )
    monitor.start()
    try:
        out, _err = proc.communicate(timeout=12)
        monitor.join(timeout=5)
    finally:
        if proc.poll() is None:
            psutil.Process(proc.pid).kill()
        if child_pid_path.exists():
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            if psutil.pid_exists(child_pid):
                psutil.Process(child_pid).kill()

    status = local_store.status(job.job_id)
    assert not monitor.is_alive()
    assert proc.returncode == 0
    assert int(out.strip()) != 0
    assert status.resource_violation is None
    assert status.resource_warning is not None
    assert "AlgoEvolve controller continues" in status.resource_warning
    assert status.peak_memory_bytes > int(0.03 * (1024**3))
