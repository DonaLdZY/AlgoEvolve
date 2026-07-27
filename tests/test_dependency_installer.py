from __future__ import annotations

import json
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

from engine.dependency_installer import (
    DependencyInstaller,
    extract_install_declarations,
    extract_scoped_install_declarations,
    extract_missing_module,
    parse_declared_requirement,
    find_unsafe_installation_call,
)
from engine.executor import Interpreter


def _cfg(
    tmp_path: Path,
    *,
    enabled: bool = True,
    policy: str = "ai_declared",
    install_target: Path | None = None,
):
    exec_cfg = SimpleNamespace(
        auto_install_missing_dependencies=enabled,
        dependency_install_policy=policy,
        dependency_install_target_path=str(
            install_target or (tmp_path / "python_packages")
        ),
        dependency_install_timeout_seconds=10,
        dependency_install_lock_timeout_seconds=10,
        dependency_install_max_packages_per_execution=2,
        dependency_install_output_tail_chars=1000,
        dependency_install_allowlist=["demo-package", "ortools", "scikit-learn"],
        dependency_import_map={
            "demo_missing": "demo-package",
            "ortools": "ortools",
            "sklearn": "scikit-learn",
        },
        dependency_package_specs={
            "demo-package": "demo-package>=1,<2",
            "ortools": "ortools>=9.9,<10",
            "scikit-learn": "scikit-learn>=1.4,<2",
        },
        dependency_install_log_filename="dependency_installations.jsonl",
        dependency_install_summary_filename="dependency_installations_summary.json",
        dependency_install_central_log_path=str(tmp_path / "central.jsonl"),
        dependency_install_central_summary_path=str(tmp_path / "central-summary.json"),
    )
    return SimpleNamespace(
        exec=exec_cfg,
        log_dir=tmp_path / "logs",
        agent=SimpleNamespace(
            search=SimpleNamespace(parallel_search_num=1),
        ),
        start_cpu_id="0",
        cpu_number="1",
    )


def test_extracts_exact_missing_module_and_safe_single_package_declaration() -> None:
    stderr = "Traceback\nModuleNotFoundError: No module named 'ortools.sat'\n"
    code = "# ALGOEVOLVE_PIP_INSTALL: pip install ortools\nimport ortools\n"

    assert extract_missing_module(stderr) == "ortools"
    assert extract_install_declarations(code) == ["pip install ortools"]
    assert extract_scoped_install_declarations(
        "# ALGOEVOLVE_PIP_INSTALL[sklearn]: pip install scikit-learn\n"
    ) == [("sklearn", "pip install scikit-learn")]
    assert parse_declared_requirement("pip install ortools>=9.9,<10").name == "ortools"
    assert parse_declared_requirement(f"{sys.executable} -m pip install ortools") is not None


def test_rejects_flags_urls_shell_and_multiple_packages() -> None:
    unsafe = [
        "pip install --upgrade ortools",
        "pip install ortools pandas",
        "pip install https://example.invalid/package.whl",
        "pip install ortools; echo unsafe",
        "conda install ortools",
    ]

    assert all(parse_declared_requirement(command) is None for command in unsafe)
    assert find_unsafe_installation_call("import subprocess\nsubprocess.run(['pip', 'install', 'x'])")
    assert find_unsafe_installation_call("!pip install x")
    assert find_unsafe_installation_call("# MLEVOLVE_PIP_INSTALL: pip install ortools\nimport ortools") == ""


def test_disabled_installer_never_attempts_recovery(tmp_path: Path, monkeypatch) -> None:
    installer = DependencyInstaller(_cfg(tmp_path, enabled=False))

    def fail_install(**_metadata):
        raise AssertionError("disabled dependency installer must not invoke pip")

    monkeypatch.setattr(installer, "_install", fail_install)
    decision = installer.maybe_recover(
        code="# MLEVOLVE_PIP_INSTALL: pip install ortools\nimport ortools",
        stderr="ModuleNotFoundError: No module named 'ortools'",
        node_id="disabled-node",
        execution_started_monotonic=time.monotonic(),
        installed_for_execution=set(),
    )

    assert decision.retry is False
    assert "disabled" in decision.message


def test_llm_may_select_distribution_with_different_import_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    installer = DependencyInstaller(_cfg(tmp_path))
    captured: list[dict] = []

    def fake_install(**metadata):
        captured.append(metadata)
        return {"success": True}

    monkeypatch.setattr(installer, "_install", fake_install)
    decision = installer.maybe_recover(
        code=(
            "# MLEVOLVE_PIP_INSTALL: pip install scikit-learn\n"
            "from sklearn.ensemble import RandomForestRegressor\n"
        ),
        stderr="ModuleNotFoundError: No module named 'sklearn'",
        node_id="sklearn-node",
        execution_started_monotonic=time.monotonic(),
        installed_for_execution=set(),
    )

    assert decision.retry is True
    assert captured[0]["distribution"] == "scikit-learn"
    assert captured[0]["requirement"] == "scikit-learn>=1.4,<2"
    assert captured[0]["source"] == "llm_declaration_mapped"


def test_llm_selection_can_override_missing_import_mapping(
    tmp_path: Path,
    monkeypatch,
) -> None:
    installer = DependencyInstaller(_cfg(tmp_path))
    captured: list[dict] = []

    def fake_install(**metadata):
        captured.append(metadata)
        return {"success": True}

    monkeypatch.setattr(installer, "_install", fake_install)
    decision = installer.maybe_recover(
        code=(
            "# MLEVOLVE_PIP_INSTALL: pip install scikit-learn\n"
            "import custom_sklearn_compat\n"
        ),
        stderr="ModuleNotFoundError: No module named 'custom_sklearn_compat'",
        node_id="ai-selected-node",
        execution_started_monotonic=time.monotonic(),
        installed_for_execution=set(),
    )

    assert decision.retry is True
    assert captured[0]["distribution"] == "scikit-learn"
    assert captured[0]["source"] == "llm_declaration_selected"


def test_ai_declared_policy_accepts_unlisted_distribution(tmp_path: Path) -> None:
    installer = DependencyInstaller(_cfg(tmp_path))

    selection = installer._select_requirement(
        "# MLEVOLVE_PIP_INSTALL: pip install never-preconfigured-package\n"
        "import novel_import_name\n",
        "novel_import_name",
    )

    assert selection is not None
    assert selection[0] == "never-preconfigured-package"
    assert selection[1] == "never-preconfigured-package"
    assert selection[2] == "llm_declaration_selected"


def test_allowlist_policy_rejects_unlisted_distribution(tmp_path: Path) -> None:
    installer = DependencyInstaller(_cfg(tmp_path, policy="allowlist"))

    selection = installer._select_requirement(
        "# MLEVOLVE_PIP_INSTALL: pip install never-preconfigured-package\n"
        "import novel_import_name\n",
        "novel_import_name",
    )

    assert selection is None


def test_known_mapping_wins_when_code_declares_multiple_packages(tmp_path: Path) -> None:
    installer = DependencyInstaller(_cfg(tmp_path))

    selection = installer._select_requirement(
        "# MLEVOLVE_PIP_INSTALL: pip install ortools\n"
        "# MLEVOLVE_PIP_INSTALL: pip install scikit-learn\n"
        "import sklearn\n",
        "sklearn",
    )

    assert selection is not None
    assert selection[0] == "scikit-learn"
    assert selection[2] == "llm_declaration_mapped"


def test_unknown_import_with_multiple_declarations_is_not_guessed(tmp_path: Path) -> None:
    installer = DependencyInstaller(_cfg(tmp_path))

    selection = installer._select_requirement(
        "# MLEVOLVE_PIP_INSTALL: pip install ortools\n"
        "# MLEVOLVE_PIP_INSTALL: pip install scikit-learn\n"
        "import unknown_module\n",
        "unknown_module",
    )

    assert selection is None


def test_scoped_ai_declarations_select_correct_package_for_each_import(
    tmp_path: Path,
) -> None:
    installer = DependencyInstaller(_cfg(tmp_path))
    code = (
        "# MLEVOLVE_PIP_INSTALL[novel_a]: pip install package-for-a\n"
        "# MLEVOLVE_PIP_INSTALL[novel_b]: pip install package-for-b\n"
        "import novel_a\n"
        "import novel_b\n"
    )

    first = installer._select_requirement(code, "novel_a")
    second = installer._select_requirement(code, "novel_b")

    assert first is not None
    assert first[:3] == (
        "package-for-a",
        "package-for-a",
        "llm_declaration_module",
    )
    assert second is not None
    assert second[:3] == (
        "package-for-b",
        "package-for-b",
        "llm_declaration_module",
    )


def test_installs_once_writes_logs_and_refuses_same_package_loop(tmp_path: Path, monkeypatch) -> None:
    installer = DependencyInstaller(_cfg(tmp_path))
    calls: list[dict] = []

    def fake_install(**metadata):
        calls.append(metadata)
        record = {
            "schema_version": "algoevolve.dependency_installation.v1",
            "timestamp": "now",
            "node_id": metadata["node_id"],
            "missing_module": metadata["missing_module"],
            "distribution": metadata["distribution"],
            "requirement": metadata["requirement"],
            "selection_source": metadata["source"],
            "llm_declared_command": metadata["declared_command"],
            "executed_command": [sys.executable, "-m", "pip", "install", metadata["requirement"]],
            "python_executable": sys.executable,
            "install_target": str(installer.install_target_path),
            "status": "installed",
            "success": True,
            "exit_code": 0,
            "duration_seconds": 0.01,
            "stdout_tail": "installed",
            "stderr_tail": "",
        }
        installer._append_record(record)
        return record

    monkeypatch.setattr(installer, "_install", fake_install)
    installed: set[str] = set()
    first = installer.maybe_recover(
        code="# MLEVOLVE_PIP_INSTALL: pip install ortools\nimport ortools",
        stderr="ModuleNotFoundError: No module named 'ortools'",
        node_id="node-1",
        execution_started_monotonic=time.monotonic(),
        installed_for_execution=installed,
    )
    second = installer.maybe_recover(
        code="# MLEVOLVE_PIP_INSTALL: pip install ortools\nimport ortools",
        stderr="ModuleNotFoundError: No module named 'ortools'",
        node_id="node-1",
        execution_started_monotonic=time.monotonic(),
        installed_for_execution=installed,
    )

    assert first.retry is True
    assert second.retry is False
    assert len(calls) == 1
    assert calls[0]["requirement"] == "ortools>=9.9,<10"
    detail = tmp_path / "logs" / "dependency_installations.jsonl"
    summary = json.loads(
        (tmp_path / "logs" / "dependency_installations_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert detail.exists()
    assert summary["installed_requirements"] == ["ortools>=9.9,<10"]
    assert (tmp_path / "central.jsonl").exists()


def test_parallel_nodes_share_one_install_and_both_retry(tmp_path: Path, monkeypatch) -> None:
    installer = DependencyInstaller(_cfg(tmp_path))
    install_started = threading.Event()
    release_install = threading.Event()
    call_count = 0

    def slow_install(**metadata):
        nonlocal call_count
        call_count += 1
        install_started.set()
        release_install.wait(timeout=2)
        return {"success": True}

    monkeypatch.setattr(installer, "_install", slow_install)
    results = []

    def recover(node_id: str, started: float) -> None:
        results.append(
            installer.maybe_recover(
                code="import demo_missing",
                stderr="ModuleNotFoundError: No module named 'demo_missing'",
                node_id=node_id,
                execution_started_monotonic=started,
                installed_for_execution=set(),
            )
        )

    started = time.monotonic()
    first = threading.Thread(target=recover, args=("node-1", started))
    second = threading.Thread(target=recover, args=("node-2", started))
    first.start()
    assert install_started.wait(timeout=1)
    second.start()
    release_install.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert call_count == 1
    assert len(results) == 2
    assert all(result.retry for result in results)


def test_separate_installers_serialize_mutations_for_same_task_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    shared_target = tmp_path / "shared-python-packages"
    first_installer = DependencyInstaller(
        _cfg(tmp_path / "first", install_target=shared_target)
    )
    second_installer = DependencyInstaller(
        _cfg(tmp_path / "second", install_target=shared_target)
    )
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def first_install(**_metadata):
        first_started.set()
        release_first.wait(timeout=2)
        return {"success": True}

    def second_install(**_metadata):
        second_started.set()
        return {"success": True}

    monkeypatch.setattr(first_installer, "_install", first_install)
    monkeypatch.setattr(second_installer, "_install", second_install)

    threads = [
        threading.Thread(
            target=installer.maybe_recover,
            kwargs={
                "code": "import demo_missing",
                "stderr": "ModuleNotFoundError: No module named 'demo_missing'",
                "node_id": node_id,
                "execution_started_monotonic": time.monotonic(),
                "installed_for_execution": set(),
            },
        )
        for installer, node_id in (
            (first_installer, "node-1"),
            (second_installer, "node-2"),
        )
    ]
    threads[0].start()
    assert first_started.wait(timeout=1)
    threads[1].start()
    time.sleep(0.2)
    assert not second_started.is_set()
    release_first.set()
    for thread in threads:
        thread.join(timeout=2)

    assert second_started.is_set()
    assert all(not thread.is_alive() for thread in threads)


def test_interpreter_reruns_same_script_after_environment_recovery(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    working = tmp_path / "workspace"
    working.mkdir()
    interpreter = Interpreter(working, timeout=10, max_parallel_run=1, cfg=cfg)
    attempts = 0

    def create_module(**_metadata):
        nonlocal attempts
        attempts += 1
        target = interpreter.dependency_installer.install_target_path
        target.mkdir(parents=True, exist_ok=True)
        (target / "demo_missing.py").write_text("VALUE = 42\n", encoding="utf-8")
        return {"success": True}

    monkeypatch.setattr(interpreter.dependency_installer, "_install", create_module)
    result = interpreter.run(
        "# MLEVOLVE_PIP_INSTALL: pip install demo-package\n"
        "import demo_missing\n"
        "print(demo_missing.VALUE)\n",
        "same-node",
    )

    assert attempts == 1
    assert result.exc_type is None
    assert "42" in "".join(result.term_out)
    assert not (working / "demo_missing.py").exists()


def test_real_pip_installs_unlisted_ai_package_into_task_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_name = "algoevolve-local-dependency-fixture"
    import_name = "algoevolve_local_dependency_fixture"
    version = "0.1.0"
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel_path = wheel_dir / f"algoevolve_local_dependency_fixture-{version}-py3-none-any.whl"
    dist_info = f"algoevolve_local_dependency_fixture-{version}.dist-info"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(f"{import_name}.py", "VALUE = 314\n")
        wheel.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\n"
            f"Name: {package_name}\n"
            f"Version: {version}\n",
        )
        wheel.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: algoevolve-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
        wheel.writestr(f"{dist_info}/RECORD", "")

    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(wheel_dir))
    working = tmp_path / "workspace"
    working.mkdir()
    interpreter = Interpreter(
        working,
        timeout=30,
        max_parallel_run=1,
        cfg=_cfg(tmp_path),
    )
    result = interpreter.run(
        f"# MLEVOLVE_PIP_INSTALL[{import_name}]: pip install {package_name}\n"
        f"import {import_name}\n"
        f"print({import_name}.VALUE)\n",
        "real-local-install",
    )

    assert result.exc_type is None
    assert "314" in "".join(result.term_out)
    target = interpreter.dependency_installer.install_target_path
    assert (target / f"{import_name}.py").exists()
    assert not (working / f"{import_name}.py").exists()
    summary = json.loads(
        (tmp_path / "logs" / "dependency_installations_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["install_target"] == str(target)
    assert summary["requirements_candidates"] == [f"{package_name}=={version}"]
