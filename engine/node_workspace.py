"""Keep each candidate's configuration and auxiliary artifacts with its models."""
from __future__ import annotations

from pathlib import Path
import shutil


def node_workspace(workspace, node_id) -> Path:
    name = str(node_id)
    if not name or name in {".", ".."} or any(char in name for char in "/\\:"):
        raise ValueError("Invalid node workspace identifier")
    return Path(workspace).resolve() / "node_runs" / name


def prepare_node_workspace(workspace, node_id) -> Path:
    root = Path(workspace).resolve()
    target = node_workspace(root, node_id)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("input", "submission"):
        shared = root / name
        shared.mkdir(exist_ok=True)
        link = target / name
        if not link.exists():
            link.symlink_to(shared, target_is_directory=True)
        elif link.resolve() != shared.resolve():
            raise RuntimeError(f"Node {name} directory does not point to the shared task directory")
    (target / "working").mkdir(exist_ok=True)
    return target


def copy_node_outputs(source: Path, destination: Path) -> None:
    """Preserve relative paths while excluding shared inputs and transient files."""
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name in {"input", "submission", "__pycache__"} or child.name.startswith("runfile_") or child.is_symlink():
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        elif child.is_file():
            shutil.copy2(child, target)
