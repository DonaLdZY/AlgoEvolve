"""
Python interpreter for executing code snippets via subprocess.
- Executes code in a separate Python process (avoids CUDA/fork issues).
- Captures stdout/stderr, exceptions and stack traces, execution time limit.
- Supports multiple parallel slots (max_parallel_run) with CPU pinning.
"""

import logging
import os
import signal
import sys
import threading
import time
import traceback
import subprocess
from dataclasses import dataclass
from multiprocessing import Lock
from pathlib import Path

import humanize
import psutil
from dataclasses_json import DataClassJsonMixin

from engine.dependency_installer import DependencyInstaller, find_unsafe_installation_call


def memory_limited_subprocess_command(
    command: list[str],
    *,
    platform_name: str | None = None,
    environment: dict[str, str] | None = None,
    python_executable: str | None = None,
    guard_path: Path | None = None,
) -> list[str]:
    current_platform = platform_name or sys.platform
    env = environment if environment is not None else os.environ
    mode = str(
        env.get("ALGOEVOLVE_MEMORY_ENFORCEMENT_MODE")
        or env.get("MLEVOLVE_MEMORY_ENFORCEMENT_MODE")
        or ""
    ).strip()
    try:
        limit_bytes = max(
            0,
            int(
                env.get("ALGOEVOLVE_MEMORY_LIMIT_BYTES")
                or env.get("MLEVOLVE_MEMORY_LIMIT_BYTES")
                or 0
            ),
        )
    except (TypeError, ValueError):
        limit_bytes = 0
    if not (current_platform.startswith("linux") or current_platform == "darwin"):
        return command
    if mode != "posix_rlimit_as_plus_child_guard" or limit_bytes <= 0:
        return command
    resolved_guard = guard_path or (Path(__file__).resolve().parents[1] / "utils" / "resource_guard.py")
    return [
        python_executable or sys.executable,
        str(resolved_guard),
        "--memory-bytes",
        str(limit_bytes),
        "--",
        *command,
    ]

logger = logging.getLogger("AlgoEvolve")

@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None



class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        agent_file_name: str = "runfile.py",
        max_parallel_run: int = 3,
        cfg=None,
        **kwargs,
    ):
        """
        Executes Python code in subprocess(es). No fork/multiprocessing.Process to avoid CUDA issues.

        Args:
            working_dir: working directory of the agent
            timeout: timeout per code execution (seconds)
            agent_file_name: base name for runfile (actual names are runfile_0.py, ...)
            max_parallel_run: max concurrent execution slots
            cfg: config (start_cpu_id, cpu_number, agent.search.parallel_search_num)
        """
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        configured_parallel_run = (
            cfg.agent.search.parallel_search_num if (cfg and getattr(cfg.agent.search, "parallel_search_num", None)) else max_parallel_run
        )
        worker_cap_raw = str(
            os.environ.get("ALGOEVOLVE_CPU_WORKER_CAP")
            or os.environ.get("MLEVOLVE_CPU_WORKER_CAP")
            or ""
        ).strip()
        try:
            worker_cap = max(1, int(worker_cap_raw)) if worker_cap_raw else configured_parallel_run
        except ValueError:
            worker_cap = configured_parallel_run
        self.max_parallel_run = min(configured_parallel_run, worker_cap)
        self.agent_file_name = [f"runfile_{i}.py" for i in range(self.max_parallel_run)]
        self.current_parallel_run = 0
        self.status_map = [0] * self.max_parallel_run
        self.start_cpu_id = int(cfg.start_cpu_id) if cfg else 0
        self.cpu_number = int(cfg.cpu_number) if cfg else 1
        self.lock = Lock()
        self._procs_lock = threading.Lock()
        self._active_procs: dict[int, subprocess.Popen] = {}
        self.dependency_installer = DependencyInstaller(cfg)

    def _available_cpus(self) -> list[int]:
        """Return the CPU ids available to this process on the current platform."""
        process = psutil.Process(os.getpid())
        cpus: list[int] = []
        if hasattr(process, "cpu_affinity"):
            try:
                cpus = sorted(int(item) for item in process.cpu_affinity())
            except Exception:
                cpus = []
        if not cpus and hasattr(os, "sched_getaffinity"):
            cpus = sorted(os.sched_getaffinity(0))
        if not cpus:
            cpus = list(range(max(1, int(os.cpu_count() or self.cpu_number or 1))))

        start_index = cpus.index(self.start_cpu_id) if self.start_cpu_id in cpus else 0
        selected = cpus[start_index : start_index + max(1, self.cpu_number)]
        return selected or cpus[: max(1, self.cpu_number)]

    def _build_affinity_pre_code(self, cpu_set: set[int]) -> str:
        """Build cross-platform best-effort CPU affinity setup code for the child process."""
        if not cpu_set:
            return ""

        cpu_list = sorted(cpu_set)
        if hasattr(os, "sched_setaffinity"):
            return (
                "import os\n"
                "try:\n"
                f"    os.sched_setaffinity(0, {cpu_list})\n"
                "except Exception:\n"
                "    pass\n"
            )

        if sys.platform == "win32":
            return (
                "import os\n"
                "try:\n"
                "    import psutil\n"
                f"    psutil.Process(os.getpid()).cpu_affinity({cpu_list})\n"
                "except Exception:\n"
                "    pass\n"
            )

        return ""

    def terminate_all_subprocesses(self) -> None:
        """Terminate all active subprocesses (for graceful Ctrl+C exit)."""
        with self._procs_lock:
            procs = list(self._active_procs.items())
            self._active_procs.clear()
        for slot_id, proc in procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception as e:
                logger.warning(f"Error terminating subprocess slot {slot_id}: {e}")
        self.dependency_installer.terminate_all()

    def check_current_status(self):
        """Check current parallel run number."""
        return self.current_parallel_run < self.max_parallel_run

    def isolate_submission_path(self, code: str, _id) -> str:
        """Per-process submission filename to avoid write conflicts."""
        target = f"submission_{_id}.csv"

        code = code.replace("submission/submission.csv", f"submission/{target}")
        code = code.replace("/submission.csv", f"/{target}")

        for quote in ("'", '"'):
            code = code.replace(
                f"to_csv({quote}submission.csv",
                f"to_csv({quote}submission/{target}",
            )

        for quote in ("'", '"'):
            code = code.replace(f"{quote}submission.csv{quote}", f"{quote}{target}{quote}")

        return code
    
    def isolate_model_path(self, code, _id):
        """Replace generic model filenames in code to avoid multi-process conflicts."""
        model_exts = (
            ".pth",
            ".bin",
            ".pt",
            ".pkl",
            ".pickle",
            ".joblib",
            ".onnx",
            ".safetensors",
            ".model",
            ".cbm",
            ".bst",
            ".ubj",
            ".npz",
            ".txt",
        )
        if not any(ext in code for ext in model_exts):
            return code

        modified_code = code

        generic_model_names = [
            "model_artifact.pkl",
            "model_artifact.joblib",
            "model_artifact.pt",
            "model_artifact.pth",
            "best_model.pth",
            "best_model.bin",
            "best_model.pt",
            "best_model.pkl",
            "best_model.joblib",
            "best_model.onnx",
            "best_model.safetensors",
            "model_best.pth",
            "model_best.bin",
            "model_best.pt",
            "model_best.pkl",
            "model_best.joblib",
            "model.pth",
            "model.pt",
            "model.bin",
            "model.pkl",
            "model.joblib",
            "model.onnx",
            "model.safetensors",
            "checkpoint.pth",
            "checkpoint.pt",
            "checkpoint.bin",
            "checkpoint.pkl",
            "checkpoint.joblib",
            "checkpoint.safetensors",
            "xgb_model.json",
            "xgb_model.ubj",
            "xgb_model.model",
            "lgb_model.txt",
            "lgb_model.pkl",
            "catboost_model.cbm",
        ]

        generic_model_names.sort(key=len, reverse=True)

        for filename in generic_model_names:
            if filename not in modified_code:
                continue

            name, ext = filename.rsplit('.', 1)
            new_filename = f"{name}_{_id}.{ext}"

            modified_code = modified_code.replace(f"/{filename}", f"/{new_filename}")
            modified_code = modified_code.replace(f"\\{filename}", f"\\{new_filename}")
            modified_code = modified_code.replace(f'"{filename}"', f'"{new_filename}"')
            modified_code = modified_code.replace(f"'{filename}'", f"'{new_filename}'")

        return modified_code
    
    def cleanup_session(self, process_id: int = -1) -> None:
        """Clean up resources for the given process slot."""
        pass

    def run(self, code: str, id, reset_session=True, working_dir: str | None = None):
        """
        Execute the provided Python command in a subprocess and return its output.

        Parameters:
            code: Python code to execute.
            reset_session: Reserved for future use.
            working_dir: Optional per-run working directory.

        Returns:
            ExecutionResult: output, exec_time, exc_type, exc_info, exc_stack.
        """
        unsafe_install = find_unsafe_installation_call(code)
        if unsafe_install:
            return ExecutionResult(
                term_out=[
                    "Generated solution code was not executed: "
                    f"{unsafe_install}. Use only `# ALGOEVOLVE_PIP_INSTALL: pip install <distribution>`; "
                    "the runtime controls dependency installation.\n"
                ],
                exec_time=0.0,
                exc_type="PermissionError",
                exc_info={"message": unsafe_install},
                exc_stack=[],
            )
        return self._run_subprocess(code=code, id=id, working_dir=working_dir)

    def _run_subprocess(self, code: str, id, working_dir: str | None = None):
        """
        Execute code via subprocess (avoids CUDA fork issues).
        Aligned with multiprocessing mode for consistency.
        """
        logger.info("REPL is executing code via subprocess")
        logger.info(f"Current running process: {self.current_parallel_run}")
        process_id = None

        with self.lock:
            self.current_parallel_run += 1
            for idx in range(self.max_parallel_run):
                if self.status_map[idx] == 0:
                    self.status_map[idx] = 1
                    process_id = idx
                    logger.info(f"Assigned process_id: {process_id}")
                    break
                elif idx == self.max_parallel_run - 1:
                    logger.info("reach max process parallel number")
                    raise ValueError("reach max process parallel number")

        start_time = time.time()
        runfile_path = None
        proc = None
        
        try:
            avail_cpus = self._available_cpus()
            cpu_set = set(avail_cpus[process_id::self.max_parallel_run])
            if not cpu_set and avail_cpus:
                cpu_set = {avail_cpus[process_id % len(avail_cpus)]}
            pre_code = self._build_affinity_pre_code(cpu_set)
            if pre_code:
                logger.info(f"Set process_id:{process_id} to prefer cpu: {cpu_set}")
            else:
                logger.info(
                    f"CPU affinity is unavailable on this platform, continuing without pinning for process_id:{process_id}"
                )

            code = self.isolate_submission_path(code=code, _id=id)
            code = self.isolate_model_path(code=code, _id=id)
            code = pre_code + code

            # decide runfile location and cwd
            run_wd = Path(working_dir).resolve() if working_dir is not None else self.working_dir
            runfile_path = run_wd / self.agent_file_name[process_id]
            run_wd.mkdir(parents=True, exist_ok=True)
            with open(runfile_path, "w") as f:
                f.write(code)

            child_in_overtime = False
            exc_type = None
            exc_info = {}
            exc_stack = []

            installed_for_execution: set[str] = set()
            while True:
                execution_started_monotonic = time.monotonic()
                cmd = memory_limited_subprocess_command(
                    [sys.executable, str(runfile_path)]
                )
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(run_wd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=self.dependency_installer.execution_environment(
                        {**os.environ, "PYTHONUNBUFFERED": "1"}
                    ),
                )
                with self._procs_lock:
                    self._active_procs[process_id] = proc
                try:
                    stdout, stderr = proc.communicate(timeout=self.timeout)
                    exec_time = time.time() - start_time
                except subprocess.TimeoutExpired:
                    logger.warning("Subprocess timeout, sending SIGINT...")
                    try:
                        proc.send_signal(signal.SIGINT)
                        stdout, stderr = proc.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            "Subprocess failed to terminate after SIGINT, killing..."
                        )
                        proc.kill()
                        stdout, stderr = proc.communicate()
                    exec_time = time.time() - start_time
                    exc_type = "TimeoutError"
                    exc_info = {}
                    exc_stack = []
                    break
                finally:
                    with self._procs_lock:
                        self._active_procs.pop(process_id, None)

                if proc.returncode != 0:
                    recovery = self.dependency_installer.maybe_recover(
                        code=code,
                        stderr=stderr or "",
                        node_id=str(id),
                        execution_started_monotonic=execution_started_monotonic,
                        installed_for_execution=installed_for_execution,
                    )
                    if recovery.retry:
                        logger.info("[dependency-install] %s", recovery.message)
                        proc = None
                        continue
                break

            if proc is not None and proc.returncode != 0 and exc_type != "TimeoutError":
                    exc_type = "RuntimeError"
                    exc_info = {}
                    exc_stack = []
                    
                    if stderr:
                        stderr_text = stderr
                        exc_patterns = [
                            ("KeyboardInterrupt", "KeyboardInterrupt"),
                            ("TimeoutError", "TimeoutError"),
                            ("CUDA", "RuntimeError"),
                            ("cuda", "RuntimeError"),
                            ("ValueError", "ValueError"),
                            ("TypeError", "TypeError"),
                            ("AttributeError", "AttributeError"),
                            ("KeyError", "KeyError"),
                            ("IndexError", "IndexError"),
                            ("FileNotFoundError", "FileNotFoundError"),
                            ("PermissionError", "PermissionError"),
                            ("ImportError", "ImportError"),
                            ("AssertionError", "AssertionError"),
                            ("NameError", "NameError"),
                            ("RuntimeError", "RuntimeError"),
                        ]
                        
                        for pattern, exc_name in exc_patterns:
                            if pattern in stderr_text:
                                exc_type = exc_name
                                break
                        
                        stderr_lines = stderr_text.splitlines()
                        for line in stderr_lines:
                            if 'File "' in line and 'line' in line:
                                try:
                                    file_start = line.find('File "') + 6
                                    file_end = line.find('"', file_start)
                                    if file_end > file_start:
                                        filename = line[file_start:file_end]
                                        line_start = line.find('line ') + 5
                                        line_end = line.find(',', line_start)
                                        if line_end == -1:
                                            line_end = line.find('\n', line_start)
                                        if line_end == -1:
                                            line_end = len(line)
                                        line_num_str = line[line_start:line_end].strip()
                                        if line_num_str.isdigit():
                                            line_num = int(line_num_str)
                                            func_name = ""
                                            if 'in ' in line:
                                                func_start = line.find('in ') + 3
                                                func_end = line.find('\n', func_start)
                                                if func_end == -1:
                                                    func_end = len(line)
                                                func_name = line[func_start:func_end].strip()
                                            
                                            filename_short = filename.replace(str(self.working_dir / self.agent_file_name[process_id]), self.agent_file_name[process_id])
                                            filename_short = os.path.basename(filename_short)
                                            exc_stack.append((filename_short, line_num, func_name, ""))
                                except Exception:
                                    pass
                        
                        for line in reversed(stderr_lines):
                            line = line.strip()
                            if line and not line.startswith("File") and not line.startswith("Traceback"):
                                if ":" in line:
                                    parts = line.split(":", 1)
                                    if len(parts) == 2:
                                        exc_info["message"] = parts[1].strip()
                                    break
            output: list[str] = []
            if stdout:
                output.extend(stdout.splitlines(keepends=True))
            if stderr:
                output.extend(stderr.splitlines(keepends=True))
            if not output:
                output = [""]
            if output and output[-1] and not output[-1].endswith("\n"):
                output.append("\n")

            if exc_type == "TimeoutError":
                output.append(
                    f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
                )
            else:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
                )
            
            return ExecutionResult(output, exec_time, exc_type, exc_info, exc_stack)
            
        except Exception as e:
            logger.error(f"Error in _run_subprocess: {e}")
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            
            exec_time = time.time() - start_time if start_time else 0
            return ExecutionResult(
                term_out=[f"Subprocess execution error: {str(e)}", error_trace],
                exec_time=exec_time,
                exc_type="RuntimeError",
                exc_info={"error": str(e)},
                exc_stack=[],
            )
        finally:
            if process_id is not None:
                with self._procs_lock:
                    self._active_procs.pop(process_id, None)
            if proc is not None:
                try:
                    if proc.poll() is None:
                        logger.warning(f"Subprocess {process_id} still running, terminating...")
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Subprocess {process_id} failed to terminate, killing...")
                            proc.kill()
                            proc.wait()
                except Exception as e:
                    logger.warning(f"Error cleaning up subprocess {process_id}: {e}")
            
            try:
                if runfile_path and runfile_path.exists():
                    os.remove(runfile_path)
            except Exception as e:
                logger.warning(f"Failed to remove runfile after subprocess execution: {e}")
            
            with self.lock:
                if process_id is not None:
                    self.status_map[process_id] = 0
                    self.current_parallel_run -= 1
