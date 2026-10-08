import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import accelerator_runtime as runtime


def test_memory_telemetry_uses_csv_parser(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout='0, "NVIDIA, Example", 16384, 12288\n'))
    assert runtime.nvidia_memory() == [{"index": 0, "name": "NVIDIA, Example", "total_gib": 16, "free_gib": 12}]


def test_disabled_guard_leaves_cpu_runtime_unchanged(monkeypatch):
    monkeypatch.delenv("ALGOEVOLVE_GPU_EXECUTION_SLOTS", raising=False)
    monkeypatch.setattr(runtime, "nvidia_memory", lambda: pytest.fail("CPU execution queried GPU"))
    guard = runtime.AcceleratorExecutionGuard()
    with guard.admission(), guard.monitor(SimpleNamespace()) as pressure:
        assert not pressure
        assert guard.runtime_prelude() == ""


def test_admission_refuses_pressure_and_releases_slot(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_GPU_EXECUTION_SLOTS", "1")
    monkeypatch.setattr(runtime, "nvidia_memory", lambda: [{"free_gib": 2.5}])
    guard = runtime.AcceleratorExecutionGuard()
    with pytest.raises(RuntimeError, match="admission reserve"):
        with guard.admission():
            pytest.fail("Low-memory admission succeeded")
    assert guard.slots.acquire(blocking=False)
    guard.slots.release()


@pytest.mark.parametrize("telemetry_lost", [False, True])
def test_monitor_stops_only_own_candidate(monkeypatch, telemetry_lost):
    monkeypatch.setenv("ALGOEVOLVE_GPU_EXECUTION_SLOTS", "1")
    def memory():
        if telemetry_lost:
            raise RuntimeError("no telemetry")
        return [{"free_gib": 1}]
    monkeypatch.setattr(runtime, "nvidia_memory", memory)
    stopped = threading.Event()
    calls = []
    def terminate(pid, **kwargs):
        calls.append(pid)
        stopped.set()
    monkeypatch.setattr(runtime, "terminate_process_tree", terminate)
    guard = runtime.AcceleratorExecutionGuard()
    with guard.monitor(SimpleNamespace(pid=123, poll=lambda: None)) as pressure:
        assert stopped.wait(3)
    assert calls == [123]
    assert len(pressure) == 1


def test_only_one_execution_slot_and_shutdown_releases_waiter(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_GPU_EXECUTION_SLOTS", "1")
    monkeypatch.setattr(runtime, "nvidia_memory", lambda: [{"free_gib": 12}])
    guard = runtime.AcceleratorExecutionGuard()
    ended = threading.Event()
    failures = []
    def wait_for_admission():
        try:
            with guard.admission():
                failures.append("Second execution admitted")
        except RuntimeError:
            ended.set()
    with guard.admission():
        worker = threading.Thread(target=wait_for_admission)
        worker.start()
        assert not ended.wait(.1)
        guard.close()
        assert ended.wait(2)
    worker.join(2)
    assert not failures


def test_torch_allocator_prelude_is_bounded(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_GPU_EXECUTION_SLOTS", "1")
    monkeypatch.setenv("ALGOEVOLVE_TORCH_MEMORY_FRACTION", "0.5")
    guard = runtime.AcceleratorExecutionGuard()
    assert "set_per_process_memory_fraction(0.5, 0)" in guard.runtime_prelude()
    monkeypatch.setenv("ALGOEVOLVE_TORCH_MEMORY_FRACTION", "nan")
    with pytest.raises(ValueError):
        guard.runtime_prelude()


def test_execution_slots_are_shared_across_processes(tmp_path):
    pool = runtime.ExecutionSlotPool(tmp_path, 2)
    closed = threading.Event()
    first, second = pool.acquire(closed), pool.acquire(closed)
    code = "from utils.accelerator_runtime import ExecutionSlotPool; import threading, sys; p=ExecutionSlotPool(sys.argv[1], 2); s=p.acquire(threading.Event()); print('admitted', flush=True); p.release(s)"
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)], cwd=Path(runtime.__file__).resolve().parents[1],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            child.communicate(timeout=.5)
        pool.release(first)
        first = None
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 0, stderr
        assert stdout.strip() == "admitted"
    finally:
        if first is not None:
            pool.release(first)
        pool.release(second)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_host_pressure_stops_own_candidate(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_GPU_EXECUTION_SLOTS", "2")
    monkeypatch.setenv("ALGOEVOLVE_HOST_FREE_RESERVE_GIB", "2")
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=1024**3))
    stopped = threading.Event()
    monkeypatch.setattr(runtime, "terminate_process_tree", lambda *a, **k: stopped.set())
    guard = runtime.AcceleratorExecutionGuard()
    with guard.monitor(SimpleNamespace(pid=123, poll=lambda: None)) as pressure:
        assert stopped.wait(3)
    assert pressure == ["Host free-memory reserve reached; candidate stopped"]


def test_reduced_pool_waits_for_all_existing_leases_and_stays_serial(tmp_path):
    first_pool = runtime.ExecutionSlotPool(tmp_path, 2)
    second_pool = runtime.ExecutionSlotPool(tmp_path, 2)
    closed = threading.Event()
    first = first_pool.acquire(closed)
    second = first_pool.acquire(closed)
    first_pool.reduce_to_one()
    first_pool.release(first)
    admitted = threading.Event()
    finish = threading.Event()

    def pending():
        lease = second_pool.acquire(closed)
        admitted.set()
        finish.wait(3)
        second_pool.release(lease)

    worker = threading.Thread(target=pending)
    worker.start()
    try:
        assert not admitted.wait(.4)
        first_pool.release(second)
        second = None
        assert admitted.wait(3)
        assert first_pool._try_acquire(0) is None
        assert first_pool._try_acquire(1) is None
    finally:
        finish.set()
        closed.set()
        if second:
            first_pool.release(second)
        worker.join(4)


@pytest.mark.parametrize("failure", ["CUDA out of memory", "MemoryError", "Unable to allocate 5 GiB", "CUBLAS_STATUS_ALLOC_FAILED", "OSError: [WinError 1455]", "The paging file is too small"])
def test_oom_reduces_future_executions_across_guard_instances(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("ALGOEVOLVE_GPU_EXECUTION_SLOTS", "2")
    monkeypatch.setenv("ALGOEVOLVE_EXECUTION_POOL_PATH", str(tmp_path))
    guard = runtime.AcceleratorExecutionGuard()
    assert not guard.observe_failure(0, failure)
    assert not guard.observe_failure(1, "ValueError: unknown category")
    assert not (tmp_path / "reduced-to-one").exists()
    assert guard.observe_failure(1, failure)
    another = runtime.AcceleratorExecutionGuard()
    lease = another.pool.acquire(threading.Event())
    try:
        assert isinstance(lease, list) and len(lease) == 2
    finally:
        another.pool.release(lease)


def test_non_torch_probe_retains_shared_admission_without_loading_torch(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_GPU_EXECUTION_SLOTS", "2")
    monkeypatch.setenv("ALGOEVOLVE_EXECUTION_POOL_PATH", str(tmp_path))
    monkeypatch.setenv("ALGOEVOLVE_TORCH_ALLOCATOR_GUARD", "0")
    guard = runtime.AcceleratorExecutionGuard()
    assert guard.enabled and guard.pool is not None
    assert guard.runtime_prelude() == ""


def test_commit_pressure_refuses_admission_despite_available_physical_ram(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_GPU_EXECUTION_SLOTS", "2")
    monkeypatch.setenv("ALGOEVOLVE_EXECUTION_POOL_PATH", str(tmp_path))
    monkeypatch.setenv("ALGOEVOLVE_HOST_FREE_RESERVE_GIB", "2")
    monkeypatch.setattr(runtime, "nvidia_memory", lambda: [{"free_gib": 13}])
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=15 * 1024**3))
    monkeypatch.setattr(runtime, "host_commit_available_bytes", lambda: 3 * 1024**3)
    guard = runtime.AcceleratorExecutionGuard()
    with pytest.raises(RuntimeError, match="commit memory"):
        with guard.admission():
            pytest.fail("Low-commit candidate admitted")
    assert (tmp_path / "reduced-to-one").is_dir()
    assert guard.slots.acquire(blocking=False)
    guard.slots.release()


def test_commit_pressure_monitor_stops_own_candidate(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_GPU_EXECUTION_SLOTS", "2")
    monkeypatch.setenv("ALGOEVOLVE_HOST_FREE_RESERVE_GIB", "2")
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=15 * 1024**3))
    monkeypatch.setattr(runtime, "host_commit_available_bytes", lambda: 1024**3)
    stopped = threading.Event()
    monkeypatch.setattr(runtime, "terminate_process_tree", lambda *a, **k: stopped.set())
    guard = runtime.AcceleratorExecutionGuard()
    with guard.monitor(SimpleNamespace(pid=123, poll=lambda: None)) as pressure:
        assert stopped.wait(3)
    assert pressure == ["Host commit-memory reserve reached; candidate stopped"]
