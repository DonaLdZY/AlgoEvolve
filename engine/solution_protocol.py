"""Finite generated-solution interfaces and deterministic pre-execution checks."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any


INTERFACE_VERSION = "mlevolve.solution.v1"


@dataclass(frozen=True)
class SolutionInterface:
    kind: str
    stateful: bool
    entrypoint: str
    train_entrypoint: str | None
    artifact_required: bool


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    required_functions: tuple[str, ...]
    expected_signatures: tuple[str, ...]
    discovered_functions: tuple[str, ...]
    missing_functions: tuple[str, ...]
    invalid_signatures: tuple[str, ...]
    dangerous_findings: tuple[str, ...]
    syntax_error: str | None = None


def interface_for(*, task_family: str, method_family: str) -> SolutionInterface:
    if task_family == "decision" and method_family in {"reinforcement_learning", "hybrid"}:
        return SolutionInterface("reinforcement_learning", True, "rollout", "train_policy", True)
    if task_family == "decision":
        return SolutionInterface("decision_solver", False, "solve", None, False)
    return SolutionInterface("prediction", True, "predict", "train", True)


def _function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _function_parameters(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    parameters: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters[node.name] = tuple(arg.arg for arg in node.args.args)
    return parameters


def expected_signatures(interface: SolutionInterface) -> dict[str, tuple[str, ...]]:
    signatures = {interface.entrypoint: ("model_path", "data")}
    if interface.train_entrypoint == "train":
        signatures["train"] = ("data", "artifact_dir")
    elif interface.train_entrypoint == "train_policy":
        signatures["train_policy"] = ("data", "artifact_dir")
    return signatures


def interface_contract_text(interface: SolutionInterface) -> str:
    """Return exact callable definitions for prompts and failure evidence."""

    return "\n".join(
        f"def {name}({', '.join(parameters)}): ..."
        for name, parameters in expected_signatures(interface).items()
    )


def _dangerous_findings(tree: ast.AST) -> list[str]:
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            base = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
            name = f"{base}.{node.func.attr}" if base else node.func.attr
        if name in {"eval", "exec", "os.system"}:
            findings.append(f"dangerous dynamic execution call: {name}")
        if name in {"subprocess.run", "subprocess.call", "subprocess.Popen"}:
            for keyword in node.keywords:
                if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    findings.append(f"shell=True is forbidden in {name}")
    return sorted(set(findings))


def preflight_code(code: str, interface: SolutionInterface) -> PreflightReport:
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return PreflightReport(
            ok=False,
            required_functions=(),
            expected_signatures=tuple(interface_contract_text(interface).splitlines()),
            discovered_functions=(),
            missing_functions=(),
            invalid_signatures=(),
            dangerous_findings=(),
            syntax_error=f"{exc.msg} at line {exc.lineno}",
        )
    discovered = _function_names(tree)
    parameters = _function_parameters(tree)
    required = [interface.entrypoint]
    if interface.train_entrypoint:
        required.insert(0, interface.train_entrypoint)
    missing = sorted(set(required) - discovered)
    invalid_signatures = sorted(
        f"{name}{parameters.get(name, ())} must start with {expected}"
        for name, expected in expected_signatures(interface).items()
        if name in parameters and parameters[name][: len(expected)] != expected
    )
    dangerous = _dangerous_findings(tree)
    return PreflightReport(
        ok=not missing and not invalid_signatures and not dangerous,
        required_functions=tuple(required),
        expected_signatures=tuple(interface_contract_text(interface).splitlines()),
        discovered_functions=tuple(sorted(discovered)),
        missing_functions=tuple(missing),
        invalid_signatures=tuple(invalid_signatures),
        dangerous_findings=tuple(dangerous),
    )


def solution_manifest(
    interface: SolutionInterface,
    *,
    artifact_path: str | None,
    node_id: str,
    method_family: str,
) -> dict[str, Any]:
    payload = asdict(interface)
    payload.update(
        {
            "interface_version": INTERFACE_VERSION,
            "node_id": str(node_id),
            "method_family": method_family,
            "artifact_path": artifact_path,
        }
    )
    return payload
