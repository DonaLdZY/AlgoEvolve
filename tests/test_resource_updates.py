import hashlib
import json

import pytest

from utils.resource_updates import snapshot_resource_update


def test_resume_accepts_explicit_update_and_archives_previous_resource_snapshot(tmp_path):
    source = tmp_path / "authorized.json"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = {"authorization": "explicit_user_update", "scope": "Runtime resources only", "workers": 2}
    source.write_text(json.dumps(first, indent=2), encoding="utf-8")
    snapshot = snapshot_resource_update(workspace, source)
    snapshot.write_bytes(snapshot.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    source.write_text(json.dumps({**first, "workers": 4, "supersedes_sha256": digest}), encoding="utf-8")
    snapshot_resource_update(workspace, source)
    assert json.loads(snapshot.read_text())["workers"] == 4
    assert json.loads((workspace / "runtime_resource_history" / f"{digest}.json").read_text()) == first
    snapshot_resource_update(workspace, source)
    assert len(list((workspace / "runtime_resource_history").iterdir())) == 1


def test_unlinked_or_non_resource_changes_do_not_modify_snapshot(tmp_path):
    source = tmp_path / "authorized.json"
    first = {"authorization": "explicit_user_update", "scope": "Runtime resources only", "workers": 2}
    source.write_text(json.dumps(first), encoding="utf-8")
    target = snapshot_resource_update(tmp_path, source)
    for update in ({**first, "workers": 4}, {**first, "scope": "Change evaluation"}):
        source.write_text(json.dumps(update), encoding="utf-8")
        with pytest.raises(ValueError):
            snapshot_resource_update(tmp_path, source)
        assert json.loads(target.read_text()) == first
