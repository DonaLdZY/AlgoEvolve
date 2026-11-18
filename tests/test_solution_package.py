import json
import shutil
import subprocess
import sys
import pytest
from types import SimpleNamespace

from engine.executor import Interpreter
from engine.final_evaluation import finalize_selected_candidate
from engine.solution_package import draft_delivery_contract, package_inference_source


def test_exported_custom_model_loads_after_moving_without_original_workspace(tmp_path):
    workspace = tmp_path / "search"
    (workspace / "input").mkdir(parents=True)
    (workspace / "best_solution").mkdir()
    code = '''import pickle
class WeightedModel:
    def predict(self, value):
        return value * 2.5
def predict(model_path, data):
    with open(model_path, 'rb') as stream:
        model = pickle.load(stream)
    assert isinstance(model, WeightedModel)
    return model.predict(data)
def final_evaluate(data, artifact_dir):
    model = artifact_dir / 'model.pkl'
    with model.open('wb') as stream:
        pickle.dump(WeightedModel(), stream)
    return {'status': 'completed', 'metrics': {'mae': 1}, 'model_path': model.name, 'inference_checks': {}}
'''
    (workspace / "best_solution/solution.py").write_text(code)
    node = SimpleNamespace(id="node-a", code=code, metric=SimpleNamespace(value=3), evaluation_protocol={})
    state = finalize_selected_candidate(SimpleNamespace(workspace_dir=workspace), node,
                                        Interpreter(workspace, timeout=10, max_parallel_run=1))
    assert state["status"] == "completed"
    moved = tmp_path / "moved"
    shutil.copytree(workspace / "best_solution", moved)
    shutil.rmtree(workspace)
    execution = subprocess.run([sys.executable, "-c", "import solution; assert solution.predict(solution.default_model_path(), 4) == 10"],
                               cwd=moved, capture_output=True, text=True, timeout=10)
    assert execution.returncode == 0, execution.stderr
    assert (moved / "generated_solution.py").read_text() == code
    manifest = json.loads((moved / "solution_manifest.json").read_text())
    assert manifest["serialization_module"] == "algoevolve_solution_node_a"


def test_search_main_pickle_survives_final_evaluation_and_portable_inference(tmp_path):
    workspace = tmp_path / 'search'
    (workspace/'input').mkdir(parents=True)
    exported = workspace/'best_solution'; exported.mkdir()
    code = '''import pickle
from pathlib import Path
class WeightedModel:
    def predict(self, value): return value * 2.5
def predict(model_path, data):
    with open(model_path, 'rb') as stream: model = pickle.load(stream)
    assert isinstance(model, WeightedModel)
    return model.predict(data)
def final_evaluate(data, artifact_dir):
    path = artifact_dir / 'model.pkl'
    assert predict(path, 4) == 10
    return {'status':'completed', 'metrics':{'mae':1}, 'model_path':path.name, 'inference_checks':{'reload':True}}
if __name__ == '__main__':
    Path('working').mkdir(exist_ok=True)
    with open('working/model.pkl', 'wb') as stream: pickle.dump(WeightedModel(), stream)
'''
    (exported/'solution.py').write_text(code)
    node = SimpleNamespace(id='node-a', code=code, metric=SimpleNamespace(value=3), evaluation_protocol={})
    runner = Interpreter(workspace, timeout=10, max_parallel_run=1)
    assert runner.run(code, node.id).exc_type is None
    original = (workspace/'working/model_node-a.pkl').read_bytes()
    state = finalize_selected_candidate(SimpleNamespace(workspace_dir=workspace), node, runner)
    assert state['status'] == 'completed'
    assert (exported/'artifacts/model_node-a.pkl').read_bytes() == original
    moved = tmp_path/'moved'; shutil.copytree(exported, moved); shutil.rmtree(workspace)
    check = "import sys,solution; previous=sys.modules['__main__']; assert solution.predict(solution.default_model_path(),4)==10; assert sys.modules['__main__'] is previous"
    result = subprocess.run([sys.executable, '-c', check], cwd=moved, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_final_model_changes_decision_solver_manifest_to_artifact_required(tmp_path):
    workspace = tmp_path / "search"
    (workspace / "input").mkdir(parents=True)
    exported = workspace / "best_solution"
    exported.mkdir()
    code = '''import pickle
class WeightedModel:
    def predict(self, value): return value * 3
def solve(model_path, data):
    with open(model_path, "rb") as stream: model = pickle.load(stream)
    return {"answer": model.predict(data["value"])}
def final_evaluate(data, artifact_dir):
    path = artifact_dir / "model.pkl"
    with path.open("wb") as stream: pickle.dump(WeightedModel(), stream)
    return {"status":"completed", "metrics":{"score":1}, "model_path":path.name, "inference_checks":{}}
'''
    (exported / "solution.py").write_text(code)
    (exported / "solution_manifest.json").write_text(json.dumps({
        "node_id": "solver", "entrypoint": "solve", "artifact_required": False,
        "stateful": False,
    }))
    node = SimpleNamespace(id="solver", code=code, metric=SimpleNamespace(value=3), evaluation_protocol={})
    state = finalize_selected_candidate(
        SimpleNamespace(workspace_dir=workspace),
        node,
        Interpreter(workspace, timeout=10, max_parallel_run=1),
    )
    assert state["status"] == "completed"
    manifest = json.loads((exported / "solution_manifest.json").read_text())
    assert manifest["artifact_required"] is True and manifest["stateful"] is True
    assert manifest["artifact_path"] == "artifacts/model_solver.pkl"

    moved = tmp_path / "moved-solver"
    shutil.copytree(exported, moved)
    shutil.rmtree(workspace)
    result = subprocess.run(
        [sys.executable, "-c", "import solution; assert solution.solve(solution.default_model_path(), {'value': 7}) == {'answer': 21}"],
        cwd=moved, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('final_status', ['not_applicable', 'not_provided'])
def test_solver_automatically_packages_without_claiming_final_evaluation(tmp_path, final_status):
    exported = tmp_path / 'best_solution'; exported.mkdir()
    code = "def solve(model_path, data):\n    return {'answer': data['value'] * 2}\n"
    if final_status == 'not_applicable':
        code += "def final_evaluate(data, artifact_dir):\n    return {'status': 'not_applicable', 'reason': 'Pure solver, no independent population'}\n"
    (exported / 'solution.py').write_text(code)
    (exported / 'solution_manifest.json').write_text(json.dumps({'node_id': 'solver', 'entrypoint': 'solve', 'artifact_required': False}))
    node = SimpleNamespace(id='solver', code=code, metric=SimpleNamespace(value=3), evaluation_protocol={})
    state = finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), node, Interpreter(tmp_path, timeout=10, max_parallel_run=1))
    assert state['status'] == final_status
    assert state['delivery']['status'] == 'prepared'
    assert state['delivery']['inference_verification'] == 'not_run'
    draft = json.loads((exported / 'delivery_contract.json').read_text(encoding='utf-8'))
    assert draft['input_schema'] == {} and draft['observed_input_fields'] == ['value']
    assert draft['observed_output_fields'] == ['answer']
    moved = tmp_path / 'moved'; shutil.copytree(exported, moved)
    shutil.rmtree(exported)
    result = subprocess.run([sys.executable, '-c', "import solution; assert solution.solve(solution.default_model_path(), {'value':7}) == {'answer':14}"],
                            cwd=moved, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_static_draft_does_not_execute_source_or_invent_types(tmp_path):
    marker = tmp_path / 'imported'
    code = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n" + '''PREDICT_COLUMNS = ['value']
def predict(model, data):
    """Use the saved model for incoming observations."""
    def helper(data):
        return {'unrelated': data['inner']}
    data[PREDICT_COLUMNS]
    if data['optional']:
        return {'answer': 1}
'''
    (tmp_path / 'solution.py').write_text(code)
    (tmp_path / 'solution_manifest.json').write_text(json.dumps({'entrypoint': 'predict'}))
    draft = draft_delivery_contract(tmp_path)
    assert not marker.exists()
    assert draft['input_schema'] == {}
    assert draft['observed_input_fields'] == ['optional']
    assert draft['observed_output_fields'] == ['answer']
    assert draft['dataframe_columns'] == ['value']
    assert draft['status'] == 'draft'


@pytest.mark.parametrize('args', ['model, data, required', 'model, data, *, required', 'data', '*args'])
def test_incompatible_signature_rejected_before_mutation(tmp_path, args):
    original = f'def predict({args}): return 1\n'
    (tmp_path / 'solution.py').write_text(original)
    (tmp_path / 'solution_manifest.json').write_text('{"entrypoint": "predict"}')
    with pytest.raises(ValueError):
        package_inference_source(tmp_path, tmp_path / 'solution.py', 'frozen')
    assert (tmp_path / 'solution.py').read_text() == original
    assert not (tmp_path / 'generated_solution.py').exists()


def test_reversed_inference_arguments_are_rejected_before_mutation(tmp_path):
    original = 'def predict(data, model_path): return {"answer": data["value"]}\n'
    (tmp_path / 'solution.py').write_text(original)
    (tmp_path / 'solution_manifest.json').write_text('{"entrypoint": "predict"}')

    with pytest.raises(ValueError, match=r"\(model_path, data\)"):
        package_inference_source(tmp_path, tmp_path / 'solution.py', 'frozen')

    assert (tmp_path / 'solution.py').read_text() == original
    assert not (tmp_path / 'generated_solution.py').exists()


def test_delivery_failure_does_not_reclassify_evaluation_or_execute_training(tmp_path):
    exported = tmp_path / 'best_solution'; exported.mkdir()
    code = "raise RuntimeError('must not execute')\ndef solve(model, data, missing): return data\n"
    (exported / 'solution.py').write_text(code)
    (exported / 'solution_manifest.json').write_text(json.dumps({'node_id': 'solver', 'entrypoint': 'solve', 'artifact_required': False}))
    node = SimpleNamespace(id='solver', code=code, metric=SimpleNamespace(value=3), evaluation_protocol={})
    state = finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), node, Interpreter(tmp_path, timeout=10, max_parallel_run=1))
    assert state['status'] == 'not_provided' and state['delivery']['status'] == 'blocked'
    assert (exported / 'solution.py').read_text() == code
    assert node.metric.value == 3
