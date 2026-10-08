import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from engine.final_evaluation import finalize_selected_candidate, require_unexposed_search
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
