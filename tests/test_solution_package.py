import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

from engine.executor import Interpreter
from engine.final_evaluation import finalize_selected_candidate


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
