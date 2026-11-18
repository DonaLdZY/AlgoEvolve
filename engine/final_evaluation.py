"""Evaluate the frozen winning recipe once, outside search and its feedback loop."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from engine.node_workspace import node_workspace
from engine.solution_package import package_final_solution


_EVALUATION_EVIDENCE_DIRS = {"evidence", "evaluation_evidence", "__pycache__"}
_EVALUATION_EVIDENCE_FILES = {
    "dev_payloads.json",
    "dev_row_evidence.json",
    "development_evidence.csv",
    "development_evidence.json",
    "development_rows.csv",
    "holdout_index.json",
    "holdout_payloads.json",
    "holdout_rows.json",
    "holdout_rows.csv",
    "holdout_evidence.json",
    "holdout_evidence.csv",
    "holdout_predictions.csv",
    "holdout_predictions.json",
    "holdout_metrics.json",
    "validation_payloads.json",
    "validation_evidence.csv",
    "validation_evidence.json",
    "predict_sample_input.json",
    "predict_sample_output.json",
    "search_audit.json",
    "final_evaluation.json",
    "final_summary.json",
}
_EVALUATION_EVIDENCE_PREFIXES = (
    "holdout_", "development_evidence", "development_rows", "dev_row_evidence",
)


def _is_evaluation_evidence_path(path: Path) -> bool:
    return any(
        part.lower() in _EVALUATION_EVIDENCE_DIRS
        or part.lower() in _EVALUATION_EVIDENCE_FILES
        or part.lower().startswith(_EVALUATION_EVIDENCE_PREFIXES)
        for part in Path(path).parts
    )


def _materialize_final_model_path(
    *,
    workspace: Path,
    artifact_dir: Path,
    selected_working_dir: Path,
    model_spec: Path,
) -> tuple[str, Path]:
    """Resolve a candidate model and freeze it inside final_evaluation/artifacts.

    Search candidates are allowed to return a relative path to a model written
    under their isolated ``node_runs/<id>/working`` directory.  The old
    implementation only accepted files already copied into ``artifact_dir``;
    a relative path such as ``../../node_runs/<id>/working/model.pkl`` was
    therefore rejected even though it stayed inside the frozen workspace.

    Absolute paths remain forbidden.  Relative paths are accepted only when
    they resolve to the final artifact directory or the selected candidate's
    working directory.  The returned path is always relative to
    ``artifact_dir`` and is the path persisted in the final result/package.
    """

    if not str(model_spec).strip() or model_spec.drive or model_spec.is_absolute():
        raise ValueError("Final model_path must be a non-empty relative path")

    workspace_root = Path(workspace).resolve()
    artifact_root = Path(artifact_dir).resolve()
    selected_root = Path(selected_working_dir).resolve()
    working_root = (selected_root / "working").resolve()
    if not artifact_root.is_relative_to(workspace_root):
        raise ValueError("Final artifact directory must stay inside the workspace")
    if not working_root.is_relative_to(workspace_root):
        raise ValueError("Selected candidate working directory must stay inside the workspace")

    candidates: list[Path] = []
    # A path returned relative to artifact_dir is the normal delivery form.
    # The second base also supports candidate code returning a path relative
    # to the isolated node directory.
    for base in (artifact_root, selected_root):
        resolved = (base / model_spec).resolve()
        if not resolved.is_relative_to(workspace_root):
            continue
        if not (
            resolved.is_relative_to(artifact_root)
            or resolved.is_relative_to(working_root)
        ):
            continue
        if resolved.is_file() and resolved not in candidates:
            candidates.append(resolved)

    if not candidates:
        raise ValueError(
            "Final model_path must resolve to an existing file in final artifacts "
            "or the selected candidate working directory"
        )

    source = candidates[0]
    if source.is_relative_to(artifact_root):
        relative = source.relative_to(artifact_root)
        target = source
    else:
        # Preserve the candidate's working-relative layout while removing the
        # node-specific absolute workspace prefix from the delivery package.
        relative = source.relative_to(working_root)
        if _is_evaluation_evidence_path(relative):
            raise ValueError("Final model_path cannot point into search or evaluation evidence")
        target = artifact_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != source:
            shutil.copy2(source, target)

    if _is_evaluation_evidence_path(relative):
        raise ValueError("Final model_path cannot point into search or evaluation evidence")
    if not target.is_file() or not target.resolve().is_relative_to(artifact_root):
        raise ValueError("Materialized final model escaped the artifact directory")
    return relative.as_posix(), source


def _final_artifact_source(working_outputs: Path, workspace: Path) -> tuple[Path, str]:
    """Resolve the selected candidate's frozen artifact root.

    Candidates conventionally train into ``./working/artifact``. Its contents
    become the root supplied to ``final_evaluate``; legacy candidates that
    write directly into ``./working`` retain that layout.
    """

    workspace_root = Path(workspace).resolve()
    working_root = Path(working_outputs).resolve()
    if not working_root.is_relative_to(workspace_root):
        raise ValueError("Selected candidate working directory must stay inside the workspace")

    canonical = Path(working_outputs) / "artifact"
    if canonical.is_symlink():
        raise ValueError("Selected candidate artifact directory cannot be a symlink")
    if canonical.is_dir() and any(
        item.name != "__pycache__" for item in canonical.iterdir()
    ):
        canonical_root = canonical.resolve()
        if not canonical_root.is_relative_to(working_root):
            raise ValueError("Selected candidate artifact directory escaped working")
        return canonical, "working/artifact"
    return Path(working_outputs), "working"


def _validate_artifact_tree(root: Path, workspace: Path) -> None:
    """Reject links and resolved paths outside the workspace before copying artifacts."""

    root = Path(root)
    workspace_root = Path(workspace).resolve()
    if not root.exists():
        return

    def is_link_or_junction(path: Path) -> bool:
        is_junction = getattr(path, "is_junction", None)
        return path.is_symlink() or (callable(is_junction) and is_junction())

    lexical_root = Path(os.path.abspath(root))
    if not lexical_root.is_relative_to(workspace_root):
        raise ValueError(f"Artifact tree path is outside the workspace: {root}")
    cursor = lexical_root
    while cursor != workspace_root:
        if is_link_or_junction(cursor):
            raise ValueError(f"Artifact tree path cannot pass through a link: {cursor}")
        cursor = cursor.parent
    resolved_root = root.resolve()
    if not resolved_root.is_relative_to(workspace_root):
        raise ValueError(f"Artifact tree escaped the workspace: {root}")

    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if is_link_or_junction(path):
                    raise ValueError(f"Artifact tree cannot contain links: {path}")
                resolved = path.resolve()
                if not resolved.is_relative_to(resolved_root):
                    raise ValueError(f"Artifact path escaped its source tree: {path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)


def _static_required_artifact_files(code: str) -> tuple[str, ...]:
    """Read literal required-file lists from candidate finalization code."""

    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return ()
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reachable: set[str] = set()
    pending = ["final_evaluate"] if "final_evaluate" in functions else []
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        for call in (
            item for item in ast.walk(functions[name]) if isinstance(item, ast.Call)
        ):
            if isinstance(call.func, ast.Name) and call.func.id in functions:
                pending.append(call.func.id)
            elif isinstance(call.func, ast.Attribute) and call.func.attr in functions:
                pending.append(call.func.attr)

    requirements: set[str] = set()
    requirement_names = {"required", "required_files", "required_artifacts", "required_paths"}
    for name in reachable:
        assignments = (
            item for item in ast.walk(functions[name])
            if isinstance(item, (ast.Assign, ast.AnnAssign))
        )
        for assignment in assignments:
            value = assignment.value
            targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
            if not isinstance(value, (ast.List, ast.Tuple)):
                continue
            if not any(
                isinstance(target, ast.Name)
                and target.id.casefold() in requirement_names
                for target in targets
            ):
                continue
            for item in value.elts:
                if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                    continue
                relative = Path(item.value)
                if (
                    item.value.strip()
                    and not relative.is_absolute()
                    and not relative.drive
                    and ".." not in relative.parts
                ):
                    requirements.add(relative.as_posix())
    return tuple(sorted(requirements))


def _ignore_evaluation_evidence(directory: str, names: list[str]) -> set[str]:
    """Keep search and validation payloads out of final inference artifacts."""
    return {name for name in names if _is_evaluation_evidence_path(Path(name))}


def require_unexposed_search(workspace_dir) -> None:
    if (Path(workspace_dir) / 'final_evaluation_lock.json').exists():
        raise RuntimeError('Final selection is locked; further search requires a new run.')
    if (Path(workspace_dir) / "search_evaluation_exposure.json").exists():
        raise RuntimeError("Search-time final evaluation exposure was recorded. This run cannot resume or evaluate a final winner; preserve the audit and start a separately reviewed run.")
    path = Path(workspace_dir) / "final_evaluation" / "status.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("holdout_access_possible", True):
        raise RuntimeError("Final evaluation has already been locked. Further search requires a new, independently defined evaluation population and a new run.")


def _completed_result_field_errors(result: dict) -> list[str]:
    errors: list[str] = []
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("metrics: expected a non-empty object")
    elif any(
        not isinstance(name, str)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for name, value in metrics.items()
    ):
        errors.append("metrics: expected finite numeric scalar values")

    checks = result.get("inference_checks")
    if isinstance(checks, dict):
        return errors
    if isinstance(checks, list) and all(isinstance(item, dict) for item in checks):
        return errors
    errors.append("inference_checks: expected an object or an array of objects")
    return errors


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
    lock = Path(cfg.workspace_dir) / 'final_evaluation_lock.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))['selection'] != identity:
        raise RuntimeError('Reviewed final evaluation retry cannot change the frozen selection')
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("selection") == identity:
            if previous.get("status") == "preflight_failed" and not previous.get(
                "holdout_access_possible", True
            ):
                attempt_stamp = time.time_ns()
                status_path.rename(directory / f"preflight-failed-{attempt_stamp}.json")
                stale_artifacts = directory / "artifacts"
                if stale_artifacts.exists() or stale_artifacts.is_symlink():
                    stale_artifacts.rename(
                        directory / f"preflight-failed-{attempt_stamp}-artifacts"
                    )
            else:
                # A crash after lock acquisition cannot prove the holdout was unread.
                return previous
        elif previous.get("holdout_access_possible", True):
            raise RuntimeError("A final evaluation is already locked to another selection; do not reuse its holdout for a new search winner")
        if status_path.exists() and previous.get("selection") != identity:
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
        # Preserve the selected source for static delivery packaging, without executing it.
        source = directory / 'selected_solution.py'
        source.write_text(node.code, encoding='utf-8', newline='\n')
        state['selected_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        source = directory / "selected_solution.py"
        executed_code = interpreter.isolate_model_path(interpreter.isolate_submission_path(node.code, node.id), node.id)
        source.write_text(executed_code, encoding="utf-8", newline='\n')
        state["executed_code_sha256"] = hashlib.sha256(executed_code.encode("utf-8")).hexdigest()
        result_path = directory / "result.json"
        artifact_dir = directory / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        isolated = node_workspace(cfg.workspace_dir, node.id)
        selected_working_dir = isolated if isolated.is_dir() else Path(cfg.workspace_dir)
        working_outputs = selected_working_dir / "working"
        if working_outputs.is_dir():
            try:
                artifact_source, artifact_source_label = _final_artifact_source(
                    working_outputs, Path(cfg.workspace_dir)
                )
                _validate_artifact_tree(artifact_source, Path(cfg.workspace_dir))
                _validate_artifact_tree(artifact_dir, Path(cfg.workspace_dir))
                shutil.copytree(
                    artifact_source,
                    artifact_dir,
                    dirs_exist_ok=True,
                    ignore=_ignore_evaluation_evidence,
                )
                state["artifact_source"] = artifact_source_label
            except (OSError, ValueError) as exc:
                state.update(
                    status="invalid_artifacts",
                    artifact_error=str(exc),
                    finished_at=time.time(),
                )
                status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                return state
        required_artifacts = _static_required_artifact_files(node.code)
        missing_artifacts = []
        for relative in required_artifacts:
            required_path = artifact_dir / relative
            if (
                not required_path.is_file()
                or not required_path.resolve().is_relative_to(artifact_dir.resolve())
            ):
                missing_artifacts.append(relative)
        if missing_artifacts:
            state.update(
                status="preflight_failed",
                holdout_access_possible=False,
                required_artifact_errors=[
                    f"required artifact is missing from final artifact_dir: {name}"
                    for name in missing_artifacts
                ],
                finished_at=time.time(),
            )
            status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            return state
        wrapper = (
            "import importlib.util, json, os, sys\n"
            "from pathlib import Path\n"
            "os.environ['ALGOEVOLVE_EVALUATION_PHASE'] = 'final'\n"
            f"os.chdir({str(selected_working_dir.resolve())!r})\n"
            f"spec = importlib.util.spec_from_file_location({state['serialization_module']!r}, {str(source.resolve())!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name] = module\n"
            "spec.loader.exec_module(module)\n"
            # Search scripts run as __main__; their frozen pickle/joblib objects
            # can reference generated classes under that name. Keep the stable
            # identity for new artifacts while resolving the existing objects.
            "previous_main = sys.modules.get('__main__')\n"
            "sys.modules['__main__'] = module\n"
            "try:\n"
            f"    result = module.final_evaluate(Path({str(final_input)!r}), Path({str(artifact_dir.resolve())!r}))\n"
            "finally:\n"
            "    if previous_main is None: sys.modules.pop('__main__', None)\n"
            "    else: sys.modules['__main__'] = previous_main\n"
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
                    try:
                        _validate_artifact_tree(artifact_dir, Path(cfg.workspace_dir))
                    except (OSError, ValueError) as exc:
                        state["status"] = "invalid_artifact_tree"
                        state["artifact_error"] = str(exc)
                    else:
                        model_spec = Path(str(final_result.get("model_path") or ""))
                        field_errors = _completed_result_field_errors(final_result)
                        if field_errors:
                            state["status"] = "invalid_result_fields"
                            state["result_field_errors"] = field_errors
                        else:
                            try:
                                materialized_path, source_model = _materialize_final_model_path(
                                    workspace=Path(cfg.workspace_dir),
                                    artifact_dir=artifact_dir,
                                    selected_working_dir=selected_working_dir,
                                    model_spec=model_spec,
                                )
                            except (OSError, ValueError) as exc:
                                state["status"] = "invalid_model_path"
                                state["model_path_error"] = str(exc)
                            else:
                                original_model_path = str(final_result.get("model_path") or "")
                                final_result["model_path_original"] = original_model_path
                                final_result["model_path"] = materialized_path
                                result_path.write_text(
                                    json.dumps(final_result, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                                state["model_path_original"] = original_model_path
                                state["model_path"] = materialized_path
                                state["model_source"] = str(source_model)
                elif final_result.get("status") == "not_applicable" and not final_result.get("reason"):
                    state["status"] = "invalid_result_fields"
                state["result_path"] = str(result_path)
                state["artifact_dir"] = str(artifact_dir)
                state["independent_verification"] = "pending"
        except Exception as exc:
            state.update(status="failed", error=str(exc))
    state["finished_at"] = time.time()
    status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    if state["status"] in {"completed", "not_applicable", "not_provided"}:
        export_dir = Path(cfg.workspace_dir) / "best_solution" / "final_evaluation"
        shutil.copytree(directory, export_dir, dirs_exist_ok=True)
        if state["status"] == "completed":
            release_artifacts = export_dir.parent / "artifacts"
            release_artifacts.mkdir(parents=True, exist_ok=True)
            _validate_artifact_tree(release_artifacts, Path(cfg.workspace_dir))
            source_artifacts = (export_dir / "artifacts").resolve()
            if not source_artifacts.is_relative_to(Path(cfg.workspace_dir).resolve()):
                raise ValueError("Final inference artifacts escaped the workspace")
            for source in sorted(source_artifacts.rglob("*")):
                relative = source.relative_to(source_artifacts)
                if _is_evaluation_evidence_path(relative):
                    continue
                if source.is_symlink() or (hasattr(source, "is_junction") and source.is_junction()):
                    raise ValueError("Final inference artifacts cannot contain links")
                if not source.resolve().is_relative_to(source_artifacts):
                    raise ValueError("Final inference artifact escaped its source directory")
                target = release_artifacts / relative
                if source.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif source.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            model_relative = Path(final_result["model_path"])
            exported_model = (release_artifacts / model_relative).resolve()
            if not exported_model.is_relative_to(release_artifacts.resolve()) or not exported_model.is_file():
                raise ValueError("Final model is missing from the inference-only artifact directory")
            relative_model = exported_model.relative_to(export_dir.parent.resolve()).as_posix()
            (export_dir.parent / "model_path.txt").write_text(relative_model, encoding="utf-8")
            manifest_path = export_dir.parent / "solution_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(artifact_path=relative_model, artifact_required=True, stateful=True,
                                artifact_evaluation_phase="final", final_evaluation_status="completed")
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if (export_dir.parent / "solution.py").is_file():
            try:
                package_final_solution(cfg.workspace_dir, state)
                state['delivery'] = {'status': 'prepared', 'contract': 'best_solution/delivery_contract.json',
                    'inference_verification': 'not_run', 'business_acceptance': 'not_claimed'}
            except Exception as exc:
                state['delivery'] = {'status': 'blocked', 'reason': str(exc), 'inference_verification': 'not_run'}
            status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
            (export_dir / 'status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    return state


def retry_legacy_main_finalization(cfg, node, interpreter, *, expected_status_sha256: str) -> dict:
    """Recover the known script-serialization failure; never reopen model search.

    A user-triggered retry keeps the exact selected source/score, archives all
    prior evidence, and uses the corrected runtime loader with unchanged models.
    It is deliberately unavailable for generic failures or completed evaluations.
    """
    workspace = Path(cfg.workspace_dir).resolve()
    directory = workspace / 'final_evaluation'
    path = directory / 'status.json'
    raw = path.read_bytes()
    state = json.loads(raw)
    log = (directory / 'execution.log').read_text(encoding='utf-8')
    identity = {'node_id': node.id, 'code_sha256': hashlib.sha256(node.code.encode('utf-8')).hexdigest(),
                'selection_score': node.metric.value, 'selection_protocol': node.evaluation_protocol}
    if hashlib.sha256(raw).hexdigest() != expected_status_sha256 or state.get('selection') != identity:
        raise ValueError('Final status or frozen selection changed; refresh before retrying')
    if (state.get('status') != 'failed' or (directory/'result.json').exists()
            or "AttributeError: Can't get attribute" not in log or "on <module '__main__'" not in log
            or ('pickle.py' not in log and 'numpy_pickle.py' not in log)):
        raise ValueError('Only the identified __main__ deserialization failure supports this recovery')
    source = directory/'selected_solution.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != state.get('executed_code_sha256'):
        raise ValueError('Frozen executed source changed; recovery refused')
    lock = workspace/'final_evaluation_lock.json'
    if lock.exists():
        if json.loads(lock.read_text(encoding='utf-8'))['selection'] != identity:
            raise ValueError('Final selection lock mismatch')
    else:
        with lock.open('x', encoding='utf-8') as stream:
            json.dump({'selection': identity, 'reason': 'reviewed runtime recovery; search remains closed'}, stream)
    archive = workspace/'final_evaluation_attempts'/f'legacy-main-{time.time_ns()}'
    archive.parent.mkdir(exist_ok=True)
    if not directory.resolve().is_relative_to(workspace) or not archive.resolve().is_relative_to(workspace):
        raise ValueError('Attempt archive must remain inside the workspace')
    directory.rename(archive)
    review = {'repair': 'resolve_search_script_main_classes_from_frozen_module',
              'previous_status_sha256': expected_status_sha256, 'previous_attempt': str(archive),
              'previous_execution_sha256': hashlib.sha256((archive/'execution.log').read_bytes()).hexdigest(),
              'same_selection': True, 'search_reopened': False, 'holdout_access_possible_in_prior_attempt': True,
              'scope': 'Runtime deserialization repair only; no candidate, score, feature, split, or model edits.'}
    return finalize_selected_candidate(cfg, node, interpreter, prior_attempt_review=review)
