import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from engine.executor import Interpreter
from engine.final_evaluation import finalize_selected_candidate
from engine.model_artifacts import find_model_artifacts
from engine.node_workspace import node_workspace, prepare_node_workspace
from engine.solution_manager import copy_model_artifacts


def test_parallel_candidates_keep_same_named_recipes_and_models_separate_and_export_paths(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input/data.csv").write_text("value\n1\n")
    runner = Interpreter(tmp_path, timeout=10, max_parallel_run=2, isolate_node_workspaces=True)
    code = """import json
from pathlib import Path
assert Path('input/data.csv').read_text() == 'value\\n1\\n'
Path('working/selected_recipe.json').write_text(json.dumps({'choice': CHOICE}))
for name in ('first', 'second'):
    directory = Path('working') / name
    directory.mkdir()
    (directory / 'model.pkl').write_bytes((CHOICE + name).encode())
Path('submission.csv').write_text('value\\n1\\n')
"""
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: runner.run("CHOICE = " + repr(name) + "\n" + code, name), ("node-a", "node-b")))
    assert all(result.exc_type is None for result in results)
    for name in ("node-a", "node-b"):
        root = node_workspace(tmp_path, name)
        assert json.loads((root / "working/selected_recipe.json").read_text()) == {"choice": name}
        assert (tmp_path / f"submission/submission_{name}.csv").is_file()
        artifacts = find_model_artifacts(tmp_path, name)
        assert all(root in path.parents for path in artifacts)
        target = tmp_path / "exports" / name
        copy_model_artifacts(SimpleNamespace(cfg=SimpleNamespace(workspace_dir=tmp_path)), SimpleNamespace(id=name), target)
        exported = list((target / "model_artifacts").rglob("*.pkl"))
        assert len(exported) == 2
        assert {path.read_bytes() for path in exported} == {(name + "first").encode(), (name + "second").encode()}
        assert (target / "runtime/working/selected_recipe.json").is_file()
        assert (target / (target / "model_path.txt").read_text()).is_file()
        assert not (target / "runtime/input").exists()


def test_final_evaluation_receives_only_selected_candidates_saved_recipe(tmp_path):
    (tmp_path / "best_solution").mkdir()
    manifest_path = tmp_path / "best_solution/solution_manifest.json"
    manifest_path.write_text(json.dumps({"artifact_path": "old-search-model.pkl", "evaluation_protocol": {"scoring_split": "development"}}))
    for name in ("selected", "other"):
        directory = prepare_node_workspace(tmp_path, name)
        (directory / "working/selected_recipe.json").write_text(json.dumps({"choice": name}))
    node = SimpleNamespace(id="selected", metric=SimpleNamespace(value=1), evaluation_protocol={}, code="""def final_evaluate(data, artifact_dir):
    import json
    recipe = json.loads((artifact_dir / 'selected_recipe.json').read_text())
    assert recipe['choice'] == 'selected'
    (artifact_dir / 'model.pkl').write_bytes(b'selected')
    return {'status': 'completed', 'metrics': {'mae': 2}, 'model_path': 'model.pkl', 'inference_checks': {'recipe': True}}
""")
    runner = Interpreter(tmp_path, timeout=10, max_parallel_run=1, isolate_node_workspaces=True)
    state = finalize_selected_candidate(SimpleNamespace(workspace_dir=tmp_path), node, runner)
    assert state["status"] == "completed"
    assert node.metric.value == 1
    best = tmp_path / "best_solution"
    assert (best / (best / "model_path.txt").read_text()).read_bytes() == b"selected"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifact_path"] == (best / "model_path.txt").read_text()
    assert manifest["artifact_evaluation_phase"] == "final"
    assert manifest["evaluation_protocol"]["scoring_split"] == "development"
