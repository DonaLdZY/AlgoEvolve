import json
import os
import time

import psutil
import pytest

from engine.executor import Interpreter, insert_runtime_prelude


def test_affinity_injection_preserves_docstring_and_future_imports(tmp_path):
    code = '\"\"\"module docs\"\"\"\nfrom __future__ import annotations\n\ndef f(x: MissingType):\n    return x\nprint(__doc__, f.__annotations__["x"])\n'
    prepared = insert_runtime_prelude(code, "import os\n")
    compile(prepared, "candidate.py", "exec")
    result = Interpreter(tmp_path, timeout=10, max_parallel_run=1).run(code, "future")
    assert result.exc_type is None, result.term_out
    assert "module docs MissingType" in "".join(result.term_out)


def test_timeout_terminates_child_tree_and_reuses_worker(tmp_path):
    code = '''import json, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path("child.json").write_text(json.dumps({"pid": child.pid}))
print("before timeout", flush=True)
time.sleep(60)
'''
    runner = Interpreter(tmp_path, timeout=1, max_parallel_run=1)
    started = time.monotonic()
    result = runner.run(code, "timeout")
    assert result.exc_type == "TimeoutError", result.term_out
    assert time.monotonic() - started < 10
    assert "before timeout" in "".join(result.term_out)
    child_pid = json.loads((tmp_path / "child.json").read_text())["pid"]
    assert not psutil.pid_exists(child_pid)
    assert runner.current_parallel_run == 0
    assert runner.run("print('reused')", "next").exc_type is None


@pytest.mark.skipif(os.name != "nt", reason="Windows CreateProcess rejects long cwd paths")
def test_long_windows_working_directory_uses_short_subprocess_alias(tmp_path):
    working_dir = tmp_path
    for index in range(3):
        working_dir = working_dir / (f"segment-{index}-" + "x" * 75)
    working_dir.mkdir(parents=True)
    assert len(str(working_dir)) >= 260

    result = Interpreter(working_dir, timeout=10, max_parallel_run=1).run(
        "from pathlib import Path\n"
        "Path('child-output.txt').write_text('ok', encoding='utf-8')\n"
        "print('cwd-ok')",
        "long-cwd",
    )

    assert result.exc_type is None, result.term_out
    assert "cwd-ok" in "".join(result.term_out)
    assert (working_dir / "child-output.txt").read_text(encoding="utf-8") == "ok"
    assert not list(working_dir.glob("runfile_*.py"))
