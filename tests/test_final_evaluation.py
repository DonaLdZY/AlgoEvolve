import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.final_evaluation import finalize_selected_candidate, require_unexposed_search, retry_legacy_main_finalization
from engine.executor import Interpreter


class LocalInterpreter:
    calls = 0
    isolate_model_path = Interpreter.isolate_model_path
    isolate_submission_path = Interpreter.isolate_submission_path

    def run(self, code, node_id, working_dir, *, allow_dependency_retry):
        assert allow_dependency_retry is False
        self.calls += 1
        process = subprocess.run([sys.executable, "-c", code], cwd=working_dir, capture_output=True, text=True, timeout=10)
        return SimpleNamespace(term_out=[process.stdout, process.stderr], exec_time=0.1,
                               exc_type=None if process.returncode == 0 else "RuntimeError")


def candidate(code):
    return SimpleNamespace(id="node-a", code=code, metric=SimpleNamespace(value=2.0),
                           evaluation_protocol={"evaluation_phase": "search", "scoring_split": "development"})


def test_final_evaluation_executes_once_after_frozen_selection_without_changing_search_score(tmp_path):
    cfg = SimpleNamespace(workspace_dir=tmp_path)
    (tmp_path / "input").mkdir()
    node = candidate("""def final_evaluate(data, artifact_dir):
    import os
    assert os.environ['ALGOEVOLVE_EVALUATION_PHASE'] == 'final'
    assert data.is_dir()
    (artifact_dir / 'model.bin').write_bytes(b'test artifact')
    return {'status': 'completed', 'metrics': {'MAE': 5.0}, 'model_path': 'model.bin', 'inference_checks': {}}
""")
    runner = LocalInterpreter()
    result = finalize_selected_candidate(cfg, node, runner)
    assert result["status"] == "completed"
    assert node.metric.value == 2.0
    assert finalize_selected_candidate(cfg, node, runner) == result
    assert runner.calls == 1
    assert (tmp_path / "best_solution/final_evaluation/artifacts/model_node-a.bin").is_file()
    with pytest.raises(RuntimeError, match="new run"):
        require_unexposed_search(tmp_path)
    node.code += "\n# different selection\n"
    with pytest.raises(RuntimeError, match="already locked"):
        finalize_selected_candidate(cfg, node, runner)
    assert runner.calls == 1


def test_failed_final_evaluation_is_not_automatically_repeated(tmp_path):
    cfg = SimpleNamespace(workspace_dir=tmp_path)
    node = candidate("def final_evaluate(data, artifact_dir):\n    raise RuntimeError('evaluation failed')\n")
    runner = LocalInterpreter()
    assert finalize_selected_candidate(cfg, node, runner)["status"] == "failed"
    assert finalize_selected_candidate(cfg, node, runner)["status"] == "failed"
    assert runner.calls == 1


def test_final_evaluation_accepts_per_sample_inference_check_objects(tmp_path):
    checks = [
        {"sample_idx": 3, "predicted": 0.42},
        {"sample_idx": 7, "predicted": 0.51},
    ]
    node = candidate("""from pathlib import Path
def final_evaluate(data, artifact_dir):
    (artifact_dir / 'model.pkl').write_bytes(b'frozen')
    return {
        'status': 'completed',
        'metrics': {'holdout_mae': 0.4},
        'model_path': 'model.pkl',
        'inference_checks': %s,
    }
""" % repr(checks))
    result = finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter())

    assert result["status"] == "completed"
    assert json.loads((tmp_path / "final_evaluation/result.json").read_text(encoding="utf-8"))["inference_checks"] == checks
    assert (tmp_path / "best_solution/final_evaluation/artifacts/model_node-a.pkl").is_file()


def test_final_evaluation_does_not_promote_flat_candidate_claims_to_metrics(tmp_path):
    node = candidate("""def final_evaluate(data, artifact_dir):
    (artifact_dir / 'model.pkl').write_bytes(b'frozen')
    return {
        'status': 'completed',
        'holdout_mae': 0.4,
        'holdout_rmse': 0.6,
        'train_median_baseline_mae_on_holdout': 0.3,
        'n_holdout': 12,
        'prediction_ok': True,
        'inference_checks': [{'check': 'predict', 'ok': True}],
        'model_path': 'model.pkl',
    }
""")
    result = finalize_selected_candidate(
        SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter()
    )

    assert result["status"] == "invalid_result_fields"
    assert result["result_field_errors"] == ["metrics: expected a non-empty object"]
    assert "metrics_normalized_from_flat_fields" not in result
    saved = json.loads(
        (tmp_path / "final_evaluation/result.json").read_text(encoding="utf-8")
    )
    assert "metrics" not in saved
    assert saved["holdout_mae"] == 0.4
    assert saved["n_holdout"] == 12


def test_final_evaluation_loads_frozen_recipe_from_supplied_artifact_dir(tmp_path):
    working = tmp_path / "node_runs/node-a/working"
    working.mkdir(parents=True)
    (working / "selected_config.json").write_text(
        json.dumps({"tier": "C", "objective": "l1", "half_life": 60}),
        encoding="utf-8",
    )
    node = candidate("""import json
def final_evaluate(data, artifact_dir):
    recipe = json.loads((artifact_dir / 'selected_config.json').read_text())
    (artifact_dir / 'model.pkl').write_bytes(b'frozen')
    return {
        'status': 'completed',
        'metrics': {'mae': 0.4},
        'model_path': 'model.pkl',
        'inference_checks': {'recipe_loaded': recipe['half_life'] == 60},
    }
""")

    result = finalize_selected_candidate(
        SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter()
    )

    assert result["status"] == "completed"
    assert json.loads(
        (tmp_path / "final_evaluation/artifacts/selected_config.json").read_text(
            encoding="utf-8"
        )
    )["half_life"] == 60
    saved = json.loads(
        (tmp_path / "final_evaluation/result.json").read_text(encoding="utf-8")
    )
    assert saved["inference_checks"]["recipe_loaded"] is True


def test_final_evaluation_promotes_canonical_working_artifact_to_argument_root(tmp_path):
    working = tmp_path / "node_runs/node-a/working"
    artifact = working / "artifact"
    evidence = working / "evidence"
    artifact.mkdir(parents=True)
    evidence.mkdir()
    (artifact / "selected_config.json").write_text(
        json.dumps({"model": "frozen", "feature_count": 12}),
        encoding="utf-8",
    )
    (artifact / "frozen_model.bin").write_bytes(b"frozen model")
    (evidence / "dev_payloads.json").write_text("[large search-only payload]")
    node = candidate("""import json
def final_evaluate(data, artifact_dir):
    recipe = json.loads((artifact_dir / 'selected_config.json').read_text())
    assert (artifact_dir / 'frozen_model.bin').read_bytes() == b'frozen model'
    assert not (artifact_dir / 'artifact').exists()
    return {
        'status': 'completed',
        'metrics': {'holdout_mae': 0.4},
        'model_path': 'frozen_model.bin',
        'inference_checks': {'recipe_loaded': recipe['model'] == 'frozen'},
    }
""")

    result = finalize_selected_candidate(
        SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter()
    )

    final_artifacts = tmp_path / "final_evaluation/artifacts"
    assert result["status"] == "completed"
    assert result["artifact_source"] == "working/artifact"
    assert (final_artifacts / "selected_config.json").is_file()
    assert (final_artifacts / "frozen_model.bin").is_file()
    assert not (final_artifacts / "artifact").exists()
    assert not (final_artifacts / "dev_payloads.json").exists()


def test_final_evaluation_rejects_link_in_selected_artifact_tree_before_copy(tmp_path):
    artifact_dir = tmp_path / "node_runs/node-a/working/artifact"
    artifact_dir.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (artifact_dir / "linked.bin").symlink_to(outside)
    node = candidate("""def final_evaluate(data, artifact_dir):
    (artifact_dir / 'model.pkl').write_bytes(b'frozen')
    return {'status': 'completed', 'metrics': {'mae': 0.4}, 'model_path': 'model.pkl', 'inference_checks': {}}
""")
    runner = LocalInterpreter()

    result = finalize_selected_candidate(
        SimpleNamespace(workspace_dir=tmp_path), node, runner
    )

    assert result["status"] == "invalid_artifacts"
    assert "cannot contain links" in result["artifact_error"]
    assert runner.calls == 0
    assert not (tmp_path / "final_evaluation/artifacts/linked.bin").exists()


def test_final_evaluation_rejects_link_created_in_final_artifact_tree(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    node = candidate("""from pathlib import Path
def final_evaluate(data, artifact_dir):
    (artifact_dir / 'model.pkl').write_bytes(b'frozen')
    (artifact_dir / 'linked.bin').symlink_to(Path(%r))
    return {'status': 'completed', 'metrics': {'mae': 0.4}, 'model_path': 'model.pkl', 'inference_checks': {}}
""" % str(outside))

    result = finalize_selected_candidate(
        SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter()
    )

    assert result["status"] == "invalid_artifact_tree"
    assert "cannot contain links" in result["artifact_error"]
    assert not (tmp_path / "best_solution/artifacts/linked.bin").exists()


def test_missing_static_required_artifact_blocks_before_holdout_and_is_recoverable(tmp_path):
    node = candidate("""from pathlib import Path
def final_evaluate(data, artifact_dir):
    required = ['selected_config.json', 'split_manifest.json']
    for name in required:
        if not (artifact_dir / name).is_file():
            raise FileNotFoundError(name)
    (artifact_dir / 'model.pkl').write_bytes(b'frozen')
    return {
        'status': 'completed',
        'metrics': {'mae': 0.4},
        'model_path': 'model.pkl',
        'inference_checks': {},
    }
""")
    cfg, runner = SimpleNamespace(workspace_dir=tmp_path), LocalInterpreter()

    first = finalize_selected_candidate(cfg, node, runner)

    assert first["status"] == "preflight_failed"
    assert first["holdout_access_possible"] is False
    assert first["required_artifact_errors"] == [
        "required artifact is missing from final artifact_dir: selected_config.json",
        "required artifact is missing from final artifact_dir: split_manifest.json",
    ]
    assert runner.calls == 0
    require_unexposed_search(tmp_path)

    artifact_dir = tmp_path / "node_runs/node-a/working/artifact"
    artifact_dir.mkdir(parents=True)
    for name in ("selected_config.json", "split_manifest.json", "model.pkl"):
        (artifact_dir / name).write_text("{}")
    second = finalize_selected_candidate(cfg, node, runner)

    assert second["status"] == "completed"
    assert runner.calls == 1
    assert len(list((tmp_path / "final_evaluation").glob("preflight-failed-*.json"))) == 1


def test_preflight_retry_does_not_reuse_stale_artifacts_from_prior_attempt(tmp_path):
    node = candidate("""from pathlib import Path
def final_evaluate(data, artifact_dir):
    required = ['selected_config.json', 'split_manifest.json']
    for name in required:
        if not (artifact_dir / name).is_file():
            raise FileNotFoundError(name)
    model = artifact_dir / 'model.pkl'
    model.write_bytes(b'frozen')
    return {'status':'completed','metrics':{'mae':1},'model_path':model.name,'inference_checks':{}}
""")
    cfg, runner = SimpleNamespace(workspace_dir=tmp_path), LocalInterpreter()
    artifact_source = tmp_path / "node_runs/node-a/working/artifact"
    artifact_source.mkdir(parents=True)
    (artifact_source / "selected_config.json").write_text('{"generation":1}')

    first = finalize_selected_candidate(cfg, node, runner)

    assert first["status"] == "preflight_failed"
    assert runner.calls == 0
    assert (tmp_path / "final_evaluation/artifacts/selected_config.json").is_file()

    for path in artifact_source.iterdir():
        path.unlink()
    (artifact_source / "split_manifest.json").write_text('{"generation":2}')

    second = finalize_selected_candidate(cfg, node, runner)

    assert second["status"] == "preflight_failed"
    assert runner.calls == 0
    assert second["required_artifact_errors"] == [
        "required artifact is missing from final artifact_dir: selected_config.json"
    ]
    assert not (tmp_path / "final_evaluation/artifacts/selected_config.json").exists()
    archived = list((tmp_path / "final_evaluation").glob("preflight-failed-*-artifacts"))
    assert len(archived) == 1
    assert (archived[0] / "selected_config.json").read_text() == '{"generation":1}'


def test_final_evaluation_keeps_holdout_evidence_out_of_inference_package(tmp_path):
    best_solution = tmp_path / "best_solution"
    best_solution.mkdir()
    code = """from pathlib import Path
def predict(model_path, data):
    return {'prediction': 1.0}
def final_evaluate(data, artifact_dir):
    (artifact_dir / 'model.bin').write_bytes(b'frozen model')
    (artifact_dir / 'holdout_evidence.csv').write_text(
        'sample_id,y_true,y_pred\\nsecret-row,0.2,0.3\\n', encoding='utf-8')
    (artifact_dir / 'final_evaluation.json').write_text(
        '{"holdout_rows":[{"y_true":0.2,"y_pred":0.3}]}', encoding='utf-8')
    (artifact_dir / 'nested').mkdir()
    (artifact_dir / 'nested' / 'holdout_rows.json').write_text(
        '[{"y_true":0.2,"y_pred":0.3}]', encoding='utf-8')
    return {
        'status': 'completed',
        'metrics': {'holdout_mae': 0.1},
        'model_path': 'model.bin',
        'inference_checks': {'predict_smoke': True},
    }
"""
    (best_solution / "solution.py").write_text(code, encoding="utf-8")
    (best_solution / "solution_manifest.json").write_text(
        json.dumps({"node_id": "node-a", "entrypoint": "predict"}),
        encoding="utf-8",
    )
    node = candidate(code)
    result = finalize_selected_candidate(
        SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter()
    )

    assert result["status"] == "completed"
    final_model = result["model_path"]
    assert (best_solution / "artifacts" / final_model).read_bytes() == b"frozen model"
    assert (best_solution / "model_path.txt").read_text(encoding="utf-8") == f"artifacts/{final_model}"
    manifest = json.loads((best_solution / "solution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_path"] == f"artifacts/{final_model}"
    assert not (best_solution / "artifacts/final_evaluation.json").exists()
    assert not (best_solution / "artifacts/nested/holdout_rows.json").exists()
    assert not (best_solution / "artifacts/holdout_evidence.csv").exists()
    assert (best_solution / "final_evaluation/result.json").is_file()


def test_final_evaluation_rejects_model_path_inside_evaluation_evidence(tmp_path):
    node = candidate("""def final_evaluate(data, artifact_dir):
    evidence = artifact_dir / 'evidence'
    evidence.mkdir()
    (evidence / 'model.bin').write_bytes(b'not a deliverable model')
    return {
        'status': 'completed',
        'metrics': {'holdout_mae': 0.1},
        'model_path': 'evidence/model.bin',
        'inference_checks': {'predict_smoke': True},
    }
""")

    result = finalize_selected_candidate(
        SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter()
    )

    assert result["status"] == "invalid_model_path"
    assert "cannot point into search or evaluation evidence" in result["model_path_error"]
    assert not (tmp_path / "final_evaluation/artifacts/evidence/model.bin").exists()
    assert not (tmp_path / "best_solution/delivery_contract.json").exists()


def test_final_evaluation_does_not_infer_metrics_from_non_numeric_flat_fields(tmp_path):
    node = candidate("""def final_evaluate(data, artifact_dir):
    return {
        'status': 'completed',
        'holdout_mae': '0.4',
        'n_holdout': 12,
        'inference_checks': {},
    }
""")
    result = finalize_selected_candidate(
        SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter()
    )

    assert result["status"] == "invalid_result_fields"
    assert result["result_field_errors"] == ["metrics: expected a non-empty object"]


def test_final_evaluation_rejects_non_numeric_values_inside_metrics_object(tmp_path):
    node = candidate("""def final_evaluate(data, artifact_dir):
    return {
        'status': 'completed',
        'metrics': {'holdout_mae': 0.4, 'note': 'not a metric'},
        'inference_checks': {},
        'model_path': 'model.pkl',
    }
""")
    result = finalize_selected_candidate(
        SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter()
    )

    assert result["status"] == "invalid_result_fields"
    assert result["result_field_errors"] == [
        "metrics: expected finite numeric scalar values"
    ]


def test_final_evaluation_reports_malformed_inference_check_items(tmp_path):
    node = candidate("""def final_evaluate(data, artifact_dir):
    return {
        'status': 'completed',
        'metrics': {'holdout_mae': 0.4},
        'model_path': 'model.pkl',
        'inference_checks': [{'sample_idx': 3}, 'not-an-object'],
    }
""")
    result = finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter())

    assert result["status"] == "invalid_result_fields"
    assert result["result_field_errors"] == [
        "inference_checks: expected an object or an array of objects"
    ]
    assert not (tmp_path / "best_solution/final_evaluation").exists()


def test_loading_recovery_keeps_exact_selection_and_archives_failure(tmp_path):
    import hashlib
    node = candidate("""def final_evaluate(data, artifact_dir):
    model = artifact_dir/'model.bin'
    model.write_bytes(b'frozen')
    return {'status':'completed','metrics':{'mae':1},'model_path':model.name,'inference_checks':{}}
""")
    class FailedLoader(LocalInterpreter):
        def run(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(term_out=["pickle.py\nAttributeError: Can't get attribute 'Model' on <module '__main__' from 'runfile.py'>"], exec_time=0.1, exc_type='AttributeError')
    cfg = SimpleNamespace(workspace_dir=tmp_path)
    finalize_selected_candidate(cfg, node, FailedLoader())
    path = tmp_path/'final_evaluation/status.json'
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match='changed'):
        retry_legacy_main_finalization(cfg, node, LocalInterpreter(), expected_status_sha256='0'*64)
    runner = LocalInterpreter()
    result = retry_legacy_main_finalization(cfg, node, runner, expected_status_sha256=digest)
    assert result['status'] == 'completed' and runner.calls == 1
    assert Path(result['prior_attempt_review']['previous_attempt'], 'status.json').read_bytes() == raw
    with pytest.raises(RuntimeError, match='locked'):
        require_unexposed_search(tmp_path)
    changed = candidate(node.code+'\n# changed')
    with pytest.raises(RuntimeError, match='selection'):
        finalize_selected_candidate(cfg, changed, runner)
    with pytest.raises(ValueError):
        retry_legacy_main_finalization(cfg, node, runner, expected_status_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert runner.calls == 1


def test_explicit_reviewed_final_input_path_does_not_bypass_once_only_lock(tmp_path):
    data = tmp_path / "input/data.csv"
    data.parent.mkdir()
    source = tmp_path / "original.csv"
    source.write_text("frozen input")
    data.symlink_to(source)
    node = candidate("""def final_evaluate(data, artifact_dir):
    assert data.read_text() == 'frozen input'
    (artifact_dir / 'model.pkl').write_bytes(b'frozen recipe')
    return {'status': 'completed', 'metrics': {'mae': 3}, 'model_path': 'model.pkl', 'inference_checks': {}}
""")
    cfg, runner = SimpleNamespace(workspace_dir=tmp_path), LocalInterpreter()
    result = finalize_selected_candidate(cfg, node, runner, input_path=data,
                                         prior_attempt_review={"no_data_read_verified": True})
    assert result["status"] == "completed"
    assert result["prior_attempt_review"]["no_data_read_verified"]
    assert node.metric.value == 2
    assert finalize_selected_candidate(cfg, node, runner, input_path=data) == result
    assert runner.calls == 1
    with pytest.raises(ValueError, match="frozen workspace"):
        finalize_selected_candidate(cfg, node, runner, input_path=tmp_path / "other.csv")


def test_absolute_final_model_path_is_rejected_before_export(tmp_path):
    node = candidate("""def final_evaluate(data, artifact_dir):
    model = artifact_dir / 'model.pkl'
    model.write_bytes(b'model')
    return {'status': 'completed', 'metrics': {'mae': 3}, 'model_path': str(model.resolve()), 'inference_checks': {}}
""")
    result = finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter())
    assert result["status"] == "invalid_model_path"


def test_workspace_relative_final_model_path_is_materialized_for_delivery(tmp_path):
    (tmp_path / "node_runs/node-a/working").mkdir(parents=True)
    node = candidate("""from pathlib import Path
def final_evaluate(data, artifact_dir):
    model = Path('working') / 'model.pkl'
    model.write_bytes(b'workspace model')
    return {
        'status': 'completed',
        'metrics': {'mae': 3},
        'model_path': '../../node_runs/node-a/working/model.pkl',
        'inference_checks': {},
    }
""")
    result = finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter())
    assert result["status"] == "completed"
    assert result["model_path"].endswith(".pkl")
    assert result["model_path_original"].startswith("../../node_runs/node-a/working/")
    assert result["model_path_original"].endswith(".pkl")
    artifact = tmp_path / "final_evaluation/artifacts" / result["model_path"]
    assert artifact.read_bytes() == b"workspace model"
    persisted = json.loads((tmp_path / "final_evaluation/result.json").read_text(encoding="utf-8"))
    assert persisted["model_path"] == result["model_path"]
    assert persisted["model_path_original"] == result["model_path_original"]


def test_missing_finalization_entrypoint_is_explicit_and_does_not_execute_search_again(tmp_path):
    runner = LocalInterpreter()
    result = finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), candidate("raise RuntimeError('must not run')"), runner)
    assert result["status"] == "not_provided"
    assert runner.calls == 0
    require_unexposed_search(tmp_path)
    different = candidate("pass")
    different.id = "node-b"
    assert finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), different, runner)["status"] == "not_provided"
    assert len(list((tmp_path / "final_evaluation").glob("unexecuted-*.json"))) == 1


def test_started_final_evaluation_requires_manual_review_after_interruption(tmp_path):
    runner = LocalInterpreter()
    cfg = SimpleNamespace(workspace_dir=tmp_path)
    node = candidate("pass")
    state = finalize_selected_candidate(cfg, node, runner)
    state["status"] = "started"
    (tmp_path / "final_evaluation/status.json").write_text(json.dumps(state), encoding="utf-8")
    assert finalize_selected_candidate(cfg, node, runner)["status"] == "started"
    assert runner.calls == 0


def test_production_interpreter_finalizes_using_frozen_search_artifact_without_reexecuting_main(tmp_path):
    (tmp_path / "input").mkdir()
    node = candidate("""from pathlib import Path
if __name__ == '__main__':
    Path('model.pkl').write_bytes(b'frozen recipe')

def final_evaluate(data, artifact_dir):
    frozen = Path('model.pkl').read_bytes()
    assert frozen == b'frozen recipe'
    (artifact_dir / 'model.pkl').write_bytes(frozen)
    return {'status': 'completed', 'metrics': {'mae': 7}, 'model_path': 'model.pkl', 'inference_checks': {'reload': True}}
""")
    runner = Interpreter(tmp_path, timeout=10, max_parallel_run=1)
    assert runner.run(node.code, node.id).exc_type is None
    result = finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), node, runner)
    assert result["status"] == "completed"
    assert node.metric.value == 2
    exported = tmp_path / "best_solution/final_evaluation/artifacts/model_node-a.pkl"
    assert exported.read_bytes() == b"frozen recipe"


@pytest.mark.parametrize("response,status", [
    ({"status": "not_applicable", "reason": "No separate final population is provided"}, "not_applicable"),
    ({"status": "not_applicable"}, "invalid_result_fields"),
    ({"status": "completed", "model_path": "../outside.pkl"}, "invalid_result_fields"),
])
def test_final_result_does_not_claim_completion_without_required_evidence(tmp_path, response, status):
    node = candidate("def final_evaluate(data, artifact_dir):\n    return " + repr(response))
    result = finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), node, LocalInterpreter())
    assert result["status"] == status
