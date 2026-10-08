"""Keep the import identity required by models containing generated Python classes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


def package_final_solution(workspace, state) -> None:
    if state.get("status") != "completed":
        return
    workspace = Path(workspace)
    exported = workspace / "best_solution"
    source = workspace / "final_evaluation/selected_solution.py"
    module_name = state.get("serialization_module", "selected_final_solution")
    if not module_name.isidentifier():
        raise ValueError("Final model serialization module must be a Python identifier")
    original = exported / "generated_solution.py"
    if not original.exists():
        shutil.copy2(exported / "solution.py", original)
    implementation = exported / (module_name + ".py")
    shutil.copy2(source, implementation)
    checksum = hashlib.sha256(implementation.read_bytes()).hexdigest()
    manifest_path = exported / "solution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    wrapper = f'''"""Portable entrypoints for the unchanged frozen generated implementation."""
from pathlib import Path
import hashlib
import importlib.util
import sys

_ROOT = Path(__file__).resolve().parent
_MODULE_NAME = {module_name!r}
_SOURCE = _ROOT / (_MODULE_NAME + ".py")
if hashlib.sha256(_SOURCE.read_bytes()).hexdigest() != {checksum!r}:
    raise RuntimeError("Packaged implementation changed after model export")
_implementation = sys.modules.get(_MODULE_NAME)
if _implementation is None:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    _implementation = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _implementation
    try:
        _spec.loader.exec_module(_implementation)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise

def __getattr__(name):
    return getattr(_implementation, name)

def default_model_path():
    return _ROOT / (_ROOT / "model_path.txt").read_text(encoding="utf-8").strip()
'''
    signatures = {"predict": "model_path, data", "train": "data, artifact_dir",
                  "solve": "model_path, data", "rollout": "model_path, data", "train_policy": "data, artifact_dir"}
    for name in (manifest.get("entrypoint", "predict"), manifest.get("train_entrypoint", "train")):
        if name in signatures:
            args = signatures[name]
            wrapper += f"\ndef {name}({args}):\n    return _implementation.{name}({args})\n"
    (exported / "solution.py").write_text(wrapper, encoding="utf-8")
    manifest.update(implementation_path=implementation.name, generated_source_path=original.name,
                    serialization_module=module_name, portable_entrypoint_path="solution.py",
                    implementation_file_sha256=checksum)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
