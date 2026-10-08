"""Opt-in GPU execution admission and monitored free-memory reserve."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import io
import math
import os
from pathlib import Path
import re
import subprocess
import threading

import psutil

from utils.resource_limits import host_commit_available_bytes, terminate_process_tree


class ExecutionSlotPool:
    """OS file locks bound executions across tasks and release on process exit."""
    def __init__(self, directory, count):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.count = count

    def reduce_to_one(self):
        # The durable marker is shared by every search process and survives restarts.
        (self.directory / "reduced-to-one").mkdir(exist_ok=True)

    def _try_acquire(self, index):
        stream = (self.directory / f"execution-{index}.lock").open("a+b")
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return stream
        except OSError:
            stream.close()
            return None

    def acquire(self, closed):
        while not closed.is_set():
            reduced = (self.directory / "reduced-to-one").exists()
            leases = []
            for index in range(self.count):
                stream = self._try_acquire(index)
                if stream is None:
                    if reduced:
                        break
                    continue
                leases.append(stream)
                if not reduced:
                    if not (self.directory / "reduced-to-one").exists():
                        return stream
                    break
            # Reserving ALL original slots drains existing leases before serial execution.
            if reduced and len(leases) == self.count:
                return leases
            for stream in leases:
                self.release(stream)
            closed.wait(.25)
        raise RuntimeError("GPU execution admission closed")

    @staticmethod
    def release(stream):
        if isinstance(stream, list):
            for item in reversed(stream):
                ExecutionSlotPool.release(item)
            return
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def nvidia_memory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError("NVIDIA memory telemetry unavailable")
    return [{"index": int(row[0]), "name": row[1].strip(), "total_gib": float(row[2]) / 1024,
             "free_gib": float(row[3]) / 1024} for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True) if row]


class AcceleratorExecutionGuard:
    def __init__(self):
        count = int(os.environ.get("ALGOEVOLVE_GPU_EXECUTION_SLOTS", "0"))
        if count < 0:
            raise ValueError("GPU execution slot count must be nonnegative")
        self.enabled = count > 0
        self.slots = threading.BoundedSemaphore(max(1, count))
        pool_path = os.environ.get("ALGOEVOLVE_EXECUTION_POOL_PATH")
        self.pool = ExecutionSlotPool(pool_path, count) if self.enabled and pool_path else None
        self.closed = threading.Event()
        self.reserve_gib = float(os.environ.get("ALGOEVOLVE_GPU_FREE_RESERVE_GIB", "2"))
        self.host_reserve_gib = float(os.environ.get("ALGOEVOLVE_HOST_FREE_RESERVE_GIB", "0"))
        if not math.isfinite(self.reserve_gib) or self.reserve_gib < 0:
            raise ValueError("GPU free-memory reserve must be finite and nonnegative")
        if not math.isfinite(self.host_reserve_gib) or self.host_reserve_gib < 0:
            raise ValueError("Host free-memory reserve must be finite and nonnegative")

    def runtime_prelude(self):
        if not self.enabled or os.environ.get("ALGOEVOLVE_TORCH_ALLOCATOR_GUARD", "1") == "0":
            return ""
        fraction = float(os.environ.get("ALGOEVOLVE_TORCH_MEMORY_FRACTION", ".5"))
        if not 0 < fraction <= 1:
            raise ValueError("PyTorch memory fraction must be in (0, 1]")
        return ("import importlib.util as _gpu_import_util\n"
                "if _gpu_import_util.find_spec('torch') is not None:\n"
                "    import torch as _gpu_runtime_torch\n"
                "    if _gpu_runtime_torch.cuda.is_available():\n"
                f"        _gpu_runtime_torch.cuda.set_per_process_memory_fraction({fraction}, 0)\n")

    def close(self):
        self.closed.set()

    def reduce_concurrency(self):
        if self.pool:
            self.pool.reduce_to_one()

    def observe_failure(self, returncode, stderr):
        if returncode and re.search(
            r"out[ -]of[ -]memory|\bMemoryError\b|\bOutOfMemoryError\b|"
            r"unable to allocate|cannot allocate memory|CUBLAS_STATUS_ALLOC_FAILED|"
            r"cudaErrorMemoryAllocation|DefaultCPUAllocator.*allocate|WinError 1455|paging file is too small",
            stderr or "", re.IGNORECASE,
        ):
            self.reduce_concurrency()
            return True
        return False

    @contextmanager
    def admission(self):
        if not self.enabled:
            yield
            return
        while not self.slots.acquire(timeout=.25):
            if self.closed.is_set():
                raise RuntimeError("GPU execution admission closed")
        lease = None
        try:
            if self.closed.is_set():
                raise RuntimeError("GPU execution admission closed")
            if self.pool:
                lease = self.pool.acquire(self.closed)
            if nvidia_memory()[0]["free_gib"] < self.reserve_gib + 1:
                self.reduce_concurrency()
                raise RuntimeError("GPU free memory below admission reserve; do not start another fit")
            if self.host_reserve_gib and psutil.virtual_memory().available / 1024**3 < self.host_reserve_gib + 2:
                self.reduce_concurrency()
                raise RuntimeError("Host free memory below execution admission reserve")
            commit_available = host_commit_available_bytes() if self.host_reserve_gib else None
            if commit_available is not None and commit_available / 1024**3 < self.host_reserve_gib + 2:
                self.reduce_concurrency()
                raise RuntimeError("Host commit memory below execution admission reserve")
            yield
        finally:
            if lease is not None:
                self.pool.release(lease)
            self.slots.release()

    @contextmanager
    def monitor(self, process):
        finished = threading.Event()
        pressure = []
        def watch():
            while not finished.wait(.5):
                if process.poll() is not None:
                    return
                if self.closed.is_set():
                    pressure.append("GPU execution shutdown requested; candidate stopped")
                    terminate_process_tree(process.pid, grace_seconds=.5)
                    return
                try:
                    if self.host_reserve_gib and psutil.virtual_memory().available / 1024**3 < self.host_reserve_gib:
                        self.reduce_concurrency()
                        pressure.append("Host free-memory reserve reached; candidate stopped")
                        terminate_process_tree(process.pid, grace_seconds=.5)
                        return
                    commit_available = host_commit_available_bytes() if self.host_reserve_gib else None
                    if commit_available is not None and commit_available / 1024**3 < self.host_reserve_gib:
                        self.reduce_concurrency()
                        pressure.append("Host commit-memory reserve reached; candidate stopped")
                        terminate_process_tree(process.pid, grace_seconds=.5)
                        return
                    if nvidia_memory()[0]["free_gib"] < self.reserve_gib:
                        self.reduce_concurrency()
                        pressure.append("GPU free-memory reserve reached; candidate stopped")
                        terminate_process_tree(process.pid, grace_seconds=.5)
                        return
                except (OSError, RuntimeError, subprocess.TimeoutExpired, IndexError, ValueError):
                    pressure.append("GPU memory telemetry lost; candidate stopped")
                    terminate_process_tree(process.pid, grace_seconds=.5)
                    return
        worker = threading.Thread(target=watch, daemon=True)
        if self.enabled:
            worker.start()
        try:
            yield pressure
        finally:
            finished.set()
            if self.enabled:
                worker.join(timeout=6)
