"""Evaluate the frozen winning recipe once, outside search and its feedback loop."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
from engine.node_workspace import node_workspace
from engine.solution_package import package_final_solution


def require_unexposed_search(workspace_dir) -> None:
    if (Path(workspace_dir) / "search_evaluation_exposure.json").exists():
        raise RuntimeError("Search-time final evaluation exposure was recorded. This run cannot resume or evaluate a final winner; preserve the audit and start a separately reviewed run.")
    path = Path(workspace_dir) / "final_evaluation" / "status.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("holdout_access_possible", True):
        raise RuntimeError("Final evaluation has already been locked. Further search requires a new, independently defined evaluation population and a new run.")


def finalize_selected_candidate(cfg, node, interpreter, *, input_path=None, prior_attempt_review=None) -> dict:
    if (Path(cfg.workspace_dir) / "search_evaluation_exposure.json").exists():
        raise RuntimeError("Cannot finalize a run with recorded search-time holdout exposure")
    input_root = Path(cfg.workspace_dir).resolve() / "input"
    final_input = Path(input_path).absolute() if input_path is not None else input_root
    # Input entries may be symlinks to the original dataset; validate the frozen entry path.
    if not final_input.is_relative_to(input_root) or ".." in final_input.relative_to(input_root).parts:
        raise ValueError("Final evaluation input must stay within the frozen workspace input")
    directory = Path(cfg.workspace_dir) / "final_evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "status.json"
    identity = {
        "node_id": node.id if node else None,
        "code_sha256": hashlib.sha256((node.code if node else "").encode("utf-8")).hexdigest(),
        "selection_score": node.metric.value if node and node.metric else None,
        "selection_protocol": node.evaluation_protocol if node else {},
    }
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("selection") == identity:
            # A crash after lock acquisition cannot prove the holdout was unread.
            return previous
        if previous.get("holdout_access_possible", True):
            raise RuntimeError("A final evaluation is already locked to another selection; do not reuse its holdout for a new search winner")
        status_path.rename(directory / f"unexecuted-{time.time_ns()}.json")
    has_entrypoint = node is not None and any(isinstance(item, ast.FunctionDef) and item.name == "final_evaluate"
                                             for item in ast.parse(node.code).body)
    state = {"schema_version": "algoevolve.final_evaluation.v1", "selection": identity, "started_at": time.time(),
             "status": "started", "holdout_access_possible": has_entrypoint,
             "serialization_module": "algoevolve_solution_" + str(node.id).replace("-", "_") if node else None}
    if input_path is not None:
        state["input_path"] = str(final_input)
    if prior_attempt_review is not None:
        state["prior_attempt_review"] = prior_attempt_review
    with status_path.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
    if node is None:
        state["status"] = "no_eligible_candidate"
    elif not has_entrypoint:
        state["status"] = "not_provided"
        state["reason"] = "The selected solution does not expose final_evaluate(data, artifact_dir); no final evaluation or refit is claimed."
    else:
        source = directory / "selected_solution.py"
        executed_code = interpreter.isolate_model_path(interpreter.isolate_submission_path(node.code, node.id), node.id)
        source.write_text(executed_code, encoding="utf-8")
        state["executed_code_sha256"] = hashlib.sha256(executed_code.encode("utf-8")).hexdigest()
        result_path = directory / "result.json"
        artifact_dir = directory / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        isolated = node_workspace(cfg.workspace_dir, node.id)
        selected_working_dir = isolated if isolated.is_dir() else Path(cfg.workspace_dir)
        working_outputs = selected_working_dir / "working"
        if working_outputs.is_dir():
            shutil.copytree(working_outputs, artifact_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        wrapper = (
            "import importlib.util, json, os, sys\n"
            "from pathlib import Path\n"
            "os.environ['ALGOEVOLVE_EVALUATION_PHASE'] = 'final'\n"
            f"os.chdir({str(selected_working_dir.resolve())!r})\n"
            f"spec = importlib.util.spec_from_file_location({state['serialization_module']!r}, {str(source.resolve())!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name] = module\n"
            "spec.loader.exec_module(module)\n"
            f"result = module.final_evaluate(Path({str(final_input)!r}), Path({str(artifact_dir.resolve())!r}))\n"
            "if not isinstance(result, dict): raise TypeError('final_evaluate must return a JSON object')\n"
            f"Path({str(result_path.resolve())!r}).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')\n"
            "print('Final evaluation finished for the frozen selection; search scores remain unchanged.')\n"
        )
        try:
            result = interpreter.run(wrapper, "final_" + node.id, working_dir=str(directory), allow_dependency_retry=False)
            (directory / "execution.log").write_text("".join(result.term_out), encoding="utf-8")
            state.update(execution_seconds=result.exec_time, exception_type=result.exc_type)
            if result.exc_type is not None or not result_path.is_file():
                state["status"] = "failed"
            else:
                final_result = json.loads(result_path.read_text(encoding="utf-8"))
                state["status"] = final_result.get("status") if final_result.get("status") in {"completed", "not_applicable"} else "invalid_result_status"
                if final_result.get("status") == "completed":
                    model_spec = Path(str(final_result.get("model_path") or ""))
                    model = (artifact_dir / model_spec).resolve()
                    if not isinstance(final_result.get("metrics"), dict) or not final_result["metrics"] or not isinstance(final_result.get("inference_checks"), dict):
                        state["status"] = "invalid_result_fields"
                    elif model_spec.drive or model_spec.is_absolute() or not model.is_relative_to(artifact_dir.resolve()) or not model.is_file():
                        state["status"] = "invalid_model_path"
                elif final_result.get("status") == "not_applicable" and not final_result.get("reason"):
                    state["status"] = "invalid_result_fields"
                state["result_path"] = str(result_path)
                state["artifact_dir"] = str(artifact_dir)
                state["independent_verification"] = "pending"
        except Exception as exc:
            state.update(status="failed", error=str(exc))
    state["finished_at"] = time.time()
    status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    if state["status"] in {"completed", "not_applicable"}:
        export_dir = Path(cfg.workspace_dir) / "best_solution" / "final_evaluation"
        shutil.copytree(directory, export_dir, dirs_exist_ok=True)
        if state["status"] == "completed":
            exported_model = export_dir / "artifacts" / final_result["model_path"]
            relative_model = exported_model.relative_to(export_dir.parent).as_posix()
            (export_dir.parent / "model_path.txt").write_text(relative_model, encoding="utf-8")
            manifest_path = export_dir.parent / "solution_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(artifact_path=relative_model, artifact_evaluation_phase="final",
                                final_evaluation_status="completed")
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if (export_dir.parent / "solution.py").is_file():
                package_final_solution(cfg.workspace_dir, state)
    return state
