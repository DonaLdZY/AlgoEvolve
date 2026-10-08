"""Preserve the provenance of explicit runtime-only changes across resumes."""
import hashlib
import json
from pathlib import Path


def snapshot_resource_update(workspace, source):
    target = Path(workspace) / "runtime_resource_override.json"
    if source is None or not Path(source).is_file():
        return target
    text = Path(source).read_text(encoding="utf-8")
    update = json.loads(text)
    if update.get("authorization") != "explicit_user_update" or update.get("scope") != "Runtime resources only":
        raise ValueError("Resource override lacks explicit runtime-only user provenance")
    if target.is_file():
        previous_bytes = target.read_bytes()
        previous = previous_bytes.decode("utf-8")
        if json.loads(previous) != update:
            digest = hashlib.sha256(previous_bytes).hexdigest()
            if update.get("supersedes_sha256") != digest:
                raise ValueError("Resource authorization changed without a matching superseded snapshot")
            archive = target.parent / "runtime_resource_history"
            archive.mkdir(exist_ok=True)
            (archive / f"{digest}.json").write_bytes(previous_bytes)
    target.write_text(text, encoding="utf-8")
    return target
