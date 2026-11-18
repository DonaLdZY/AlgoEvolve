"""Finite generated-solution interfaces and deterministic pre-execution checks."""

from __future__ import annotations

import ast
import inspect
from dataclasses import asdict, dataclass
from typing import Any


INTERFACE_VERSION = "algoevolve.solution.v1"
LEGACY_INTERFACE_VERSIONS = frozenset({"mlevolve.solution.v1"})


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
    policy_findings: tuple[str, ...] = ()
    syntax_error: str | None = None
    invalid_calls: tuple[str, ...] = ()


def interface_for(*, task_family: str, method_family: str) -> SolutionInterface:
    if task_family == "decision" and method_family in {"reinforcement_learning", "hybrid"}:
        return SolutionInterface("reinforcement_learning", True, "rollout", "train_policy", True)
    if task_family == "decision":
        return SolutionInterface("decision_solver", False, "solve", None, False)
    return SolutionInterface("prediction", True, "predict", "train", True)


def _function_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _function_parameters(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    parameters: dict[str, tuple[str, ...]] = {}
    # Exported entry points live in module scope; estimator methods may share names.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters[node.name] = tuple(arg.arg for arg in (*node.args.posonlyargs, *node.args.args))
    return parameters


def _module_interface_call_findings(tree: ast.Module, names: set[str]) -> list[str]:
    """Check statically known module/CLI calls without executing candidate code.

    Skip decorated/rebound definitions, nested scopes and dynamic *args/**kwargs:
    those need runtime checks rather than guessed argument values or bindings.
    """
    definitions: dict[str, list[ast.FunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            definitions.setdefault(node.name, []).append(node)
    calls: list[ast.Call] = []
    rebound: set[str] = set()

    class ModuleExecution(ast.NodeVisitor):
        def visit_FunctionDef(self, node):  # noqa: N802
            return

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef
        visit_ListComp = visit_FunctionDef
        visit_SetComp = visit_FunctionDef
        visit_DictComp = visit_FunctionDef
        visit_GeneratorExp = visit_FunctionDef

        def visit_Name(self, node):  # noqa: N802
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                rebound.add(node.id)

        def visit_Import(self, node):  # noqa: N802
            rebound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

        def visit_ImportFrom(self, node):  # noqa: N802
            rebound.update(alias.asname or alias.name for alias in node.names)

        def visit_Call(self, node):  # noqa: N802
            calls.append(node)
            self.generic_visit(node)

    ModuleExecution().visit(tree)
    findings: list[str] = []
    for call in calls:
        if not isinstance(call.func, ast.Name) or call.func.id in rebound:
            continue
        candidates = definitions.get(call.func.id, [])
        if len(candidates) != 1 or candidates[0].decorator_list:
            continue
        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
            continue
        args = candidates[0].args
        positional = [*args.posonlyargs, *args.args]
        required_count = len(positional) - len(args.defaults)
        parameters = [
            inspect.Parameter(
                arg.arg,
                inspect.Parameter.POSITIONAL_ONLY if index < len(args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty if index < required_count else None,
            )
            for index, arg in enumerate(positional)
        ]
        if args.vararg:
            parameters.append(inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
        parameters.extend(
            inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                              default=inspect.Parameter.empty if default is None else None)
            for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        )
        if args.kwarg:
            parameters.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
        try:
            inspect.Signature(parameters).bind(
                *[None for _ in call.args], **{kw.arg: None for kw in call.keywords},
            )
        except TypeError as exc:
            findings.append(f"module call {call.func.id} at line {call.lineno}: {exc}")
    return findings


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


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        parts: list[str] = []
        current: ast.AST | None = call.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def _constant_text(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip().lower()
    return None


def _search_phase_policy_findings(
    tree: ast.Module,
    *,
    search_roots: set[str],
) -> list[str]:
    """Reject final-population access reachable from candidate search execution.

    A top-level helper may be called exclusively by the engine-owned
    ``final_evaluate`` entrypoint. Function nesting is not a reliable boundary,
    so policy enforcement follows statically resolvable calls from both search
    entrypoints and executable module-level code.
    """

    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    class FunctionCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            qualified = ".".join((*self.scope, node.name))
            functions[qualified] = node
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    FunctionCollector().visit(tree)
    names_by_leaf: dict[str, list[str]] = {}
    for qualified in functions:
        names_by_leaf.setdefault(qualified.rsplit(".", 1)[-1], []).append(qualified)

    def resolve_call(function_name: str, call: ast.Call) -> set[str]:
        if isinstance(call.func, ast.Name):
            parts = function_name.split(".")[:-1]
            for length in range(len(parts), -1, -1):
                candidate = ".".join((*parts[:length], call.func.id))
                if candidate in functions:
                    return {candidate}
            return set()
        if not isinstance(call.func, ast.Attribute):
            return set()
        leaf = call.func.attr
        owner = call.func.value
        parts = function_name.split(".")
        if isinstance(owner, ast.Name) and owner.id in {"self", "cls"}:
            for length in range(len(parts) - 1, 0, -1):
                candidate = ".".join((*parts[:length], leaf))
                if candidate in functions:
                    return {candidate}
        candidates = set(names_by_leaf.get(leaf, ()))
        top_level = {name for name in candidates if "." not in name}
        return top_level or candidates

    class CallCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.calls: list[ast.Call] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            self.calls.append(node)
            self.generic_visit(node)

    def calls_in_statements(statements: list[ast.stmt]) -> list[ast.Call]:
        collector = CallCollector()
        for statement in statements:
            collector.visit(statement)
        return collector.calls

    calls_by_function = {
        name: calls_in_statements(function.body)
        for name, function in functions.items()
    }
    module_calls = calls_in_statements(tree.body)

    def reachable(roots: set[str], initial_calls: list[ast.Call] | None = None) -> set[str]:
        visited: set[str] = set()
        pending = [
            target
            for call in (initial_calls or [])
            for target in resolve_call("", call)
        ]
        pending.extend(name for name in roots if name in functions)
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            pending.extend(
                target
                for call in calls_by_function.get(name, [])
                for target in resolve_call(name, call)
            )
        return visited

    search_reachable = reachable(search_roots, module_calls)
    final_reachable = reachable({"final_evaluate"})
    findings: list[str] = []

    def policy_reason(call: ast.Call) -> str | None:
        name = _call_name(call)
        leaf = name.rsplit(".", 1)[-1]
        partition = next(
            (
                _constant_text(keyword.value)
                for keyword in call.keywords
                if keyword.arg in {"partition", "split", "phase", "population", "dataset"}
            ),
            None,
        )
        allow_holdout = next(
            (
                bool(keyword.value.value)
                for keyword in call.keywords
                if keyword.arg in {"_allow_holdout", "allow_holdout"}
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, bool)
            ),
            False,
        )
        if leaf in {
            "evaluate_holdout",
            "evaluate_final",
            "final_evaluate",
            "run_holdout_evaluation",
            "score_holdout",
        }:
            return f"final/holdout evaluation call `{name}`"
        if partition in {"holdout", "test", "final", "留出", "测试"}:
            return f"call with final population `{partition}`"
        if allow_holdout:
            return f"call enables holdout access with `{name or 'call'}`"
        return None

    for call in module_calls:
        reason = policy_reason(call)
        if reason:
            findings.append(f"search-time {reason} at line {call.lineno} is forbidden")

    for name, calls in calls_by_function.items():
        is_search_reachable = name in search_reachable
        is_final_only = name in final_reachable and not is_search_reachable
        for call in calls:
            reason = policy_reason(call)
            if not reason:
                continue
            if is_search_reachable:
                findings.append(
                    f"search-time {reason} in `{name}` at line {call.lineno} is forbidden"
                )
            elif not is_final_only:
                findings.append(
                    f"{reason} in `{name}` is not reachable exclusively from engine `final_evaluate`"
                )
    return sorted(set(findings))


def _final_result_contract_findings(
    tree: ast.Module,
) -> list[str]:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    final_function = functions.get("final_evaluate")
    if final_function is None:
        return []

    required = {"metrics", "model_path", "inference_checks"}
    missing_by_return: list[set[str]] = []

    def literal_mapping_keys(value: ast.AST) -> set[str] | None:
        if not isinstance(value, ast.Dict):
            return None
        keys = {
            str(key.value)
            for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        return keys if len(keys) == len(value.keys) else None

    def literal_mapping_status(value: ast.AST) -> bool:
        return isinstance(value, ast.Dict) and any(
            isinstance(key, ast.Constant)
            and key.value == "status"
            and isinstance(item, ast.Constant)
            and item.value == "completed"
            for key, item in zip(value.keys, value.values)
        )

    def assigned_names(statement: ast.Assign | ast.AnnAssign) -> set[str]:
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        return {
            target.id
            for root in targets
            for target in ast.walk(root)
            if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)
        }

    def mutates_result(statement: ast.stmt, name: str) -> bool:
        for item in ast.walk(statement):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and item is not statement:
                continue
            if (
                isinstance(item, ast.Subscript)
                and isinstance(item.ctx, ast.Store)
                and isinstance(item.value, ast.Name)
                and item.value.id == name
            ):
                return True
            if (
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and isinstance(item.func.value, ast.Name)
                and item.func.value.id == name
                and item.func.attr in {"clear", "pop", "popitem", "setdefault", "update"}
            ):
                return True
        return False

    top_level_assignments = [
        statement
        for statement in final_function.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
    ]
    returns = [
        node
        for node in ast.walk(final_function)
        if isinstance(node, ast.Return)
        and not any(
            isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            and parent is not final_function
            for parent in _ast_parents(final_function, node)
        )
    ]
    for node in returns:
        value = node.value
        if value is None:
            continue
        keys = literal_mapping_keys(value)
        completed = literal_mapping_status(value)
        if isinstance(value, ast.Name):
            candidates = [
                statement
                for statement in top_level_assignments
                if statement.lineno < node.lineno
                and value.id in assigned_names(statement)
            ]
            if not candidates:
                continue
            source = max(candidates, key=lambda statement: (statement.lineno, statement.col_offset))
            if any(
                mutates_result(statement, value.id)
                for statement in final_function.body
                if source.lineno < statement.lineno < node.lineno
            ):
                continue
            keys = literal_mapping_keys(source.value) if source.value is not None else None
            completed = literal_mapping_status(source.value) if source.value is not None else False
        if keys is not None and completed:
            missing_by_return.append(required - keys)

    if not missing_by_return:
        return []
    missing = sorted(set().union(*missing_by_return))
    return (
        ["final_evaluate completed return must include fields: " + ", ".join(missing)]
        if missing
        else []
    )


def _ast_parents(root: ast.AST, target: ast.AST) -> list[ast.AST]:
    """Return the parent chain for a node in a small generated candidate AST."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(root):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    chain: list[ast.AST] = []
    current = target
    while id(current) in parents:
        current = parents[id(current)]
        chain.append(current)
    return chain


def preflight_code(code: str, interface: SolutionInterface, *, require_final_evaluation: bool = False) -> PreflightReport:
    try:
        tree = ast.parse(code or "")
        compile(tree, "<generated-solution>", "exec")
    except SyntaxError as exc:
        return PreflightReport(
            ok=False,
            required_functions=(),
            expected_signatures=tuple(interface_contract_text(interface).splitlines()),
            discovered_functions=(),
            missing_functions=(),
            invalid_signatures=(),
            dangerous_findings=(),
            policy_findings=(),
            syntax_error=f"{exc.msg} at line {exc.lineno}",
        )
    discovered = _function_names(tree)
    parameters = _function_parameters(tree)
    required = [interface.entrypoint]
    if interface.train_entrypoint:
        required.insert(0, interface.train_entrypoint)
    signatures = expected_signatures(interface)
    if require_final_evaluation or "final_evaluate" in discovered:
        required.append("final_evaluate")
        signatures["final_evaluate"] = ("data", "artifact_dir")
    missing = sorted(set(required) - discovered)
    invalid_signatures = sorted(
        f"{name}{parameters.get(name, ())} must start with {expected}"
        for name, expected in signatures.items()
        if name in parameters and parameters[name][: len(expected)] != expected
    )
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) or function.name not in signatures:
            continue
        supplied = len(signatures[function.name])
        args = function.args
        required_positionals = len(args.posonlyargs) + len(args.args) - len(args.defaults)
        required_keywords = [arg.arg for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is None]
        if required_positionals > supplied or required_keywords:
            invalid_signatures.append(f"{function.name} cannot be called with only {supplied} positional arguments; extra required parameters exist")
        if isinstance(function, ast.AsyncFunctionDef):
            invalid_signatures.append(f"{function.name} must be synchronous for the generated-solution runner")
    invalid_calls = _module_interface_call_findings(tree, set(signatures))
    dangerous = _dangerous_findings(tree)
    search_roots = {interface.entrypoint}
    if interface.train_entrypoint:
        search_roots.add(interface.train_entrypoint)
    policy_findings = _search_phase_policy_findings(tree, search_roots=search_roots)
    final_functions = {"final_evaluate"} if "final_evaluate" in discovered else set()
    if final_functions:
        policy_findings.extend(
            _final_result_contract_findings(tree)
        )
        policy_findings = sorted(set(policy_findings))
    # The outer engine imports and invokes this entrypoint only after selection.
    # Disallow candidate-side references, including aliases and CLI callbacks.
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == "final_evaluate":
            continue
        for item in ast.walk(statement):
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id == "final_evaluate":
                dangerous.append(f"final_evaluate is engine-only; candidate reference at line {item.lineno}")
    return PreflightReport(
        ok=not missing and not invalid_signatures and not invalid_calls and not dangerous and not policy_findings,
        required_functions=tuple(required),
        expected_signatures=tuple(f"def {name}({', '.join(args)}): ..." for name, args in signatures.items()),
        discovered_functions=tuple(sorted(discovered)),
        missing_functions=tuple(missing),
        invalid_signatures=tuple(invalid_signatures),
        dangerous_findings=tuple(dangerous),
        policy_findings=tuple(policy_findings),
        invalid_calls=tuple(invalid_calls),
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
