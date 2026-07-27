from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_run_help_exits_before_runtime_or_llm_initialization() -> None:
    env = os.environ.copy()
    for name in (
        "DEEPSEEK_API_KEY",
        "ALGOEVOLVE_CODE_API_KEY",
        "ALGOEVOLVE_FEEDBACK_API_KEY",
        "ALGOEVOLVE_EMBEDDING_API_KEY",
        "MLEVOLVE_CODE_API_KEY",
        "MLEVOLVE_FEEDBACK_API_KEY",
        "MLEVOLVE_EMBEDDING_API_KEY",
    ):
        env.pop(name, None)

    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: python run.py [key=value ...]" in result.stdout
    assert "data_dir=PATH" in result.stdout
    assert "Starting run" not in result.stdout
    assert "Querying OpenAI-compatible API" not in result.stdout
    assert result.stderr == ""
