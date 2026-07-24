"""Helpers for locating model artifacts produced by a search node."""

from __future__ import annotations

from pathlib import Path


MODEL_ARTIFACT_EXTENSIONS = {
    ".pt",
    ".pth",
    ".bin",
    ".pkl",
    ".pickle",
    ".joblib",
    ".onnx",
    ".safetensors",
    ".model",
    ".cbm",
    ".bst",
    ".ubj",
    ".npz",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
}

IGNORED_ARTIFACT_DIRS = {
    "input",
    "submission",
    "best_solution",
    "best_submission",
    "top_solution",
    "top_solution_llm",
    "checkpoint_candidates",
    "global_memory",
}


def _candidate_roots(workspace_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for name in ("working", "models", "artifacts", "checkpoints"):
        path = workspace_dir / name
        if path.exists() and path.is_dir():
            roots.append(path)
    if not roots:
        roots.append(workspace_dir)
    return roots


def find_model_artifacts(workspace_dir: Path, node_id: str, limit: int = 64) -> list[Path]:
    """Return node-specific model artifacts under the workspace.

    The executor rewrites common model filenames to include the node id, so
    selecting by node id avoids copying artifacts from other parallel nodes.
    """
    workspace = Path(workspace_dir).resolve()
    node_text = str(node_id)
    if not node_text:
        return []

    found: list[Path] = []
    seen: set[str] = set()
    for root in _candidate_roots(workspace):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(found) >= limit:
                break
            if not path.is_file():
                continue
            try:
                rel = path.resolve().relative_to(workspace)
            except Exception:
                continue
            if rel.parts and rel.parts[0] in IGNORED_ARTIFACT_DIRS:
                continue
            if path.suffix.lower() not in MODEL_ARTIFACT_EXTENSIONS:
                continue
            if node_text not in path.name and node_text not in str(path.parent):
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    return sorted(found, key=lambda p: str(p).lower())
