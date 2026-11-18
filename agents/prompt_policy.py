"""Shared prompt policy for language, task/method taxonomy, and scoped search memory."""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Iterable

from agents.prompts import infer_task_mode
from engine.expansion_profile import ExpansionProfile
from utils.response import trim_long_string


OUTPUT_LANGUAGE_ALIASES = {
    "en": "english",
    "eng": "english",
    "english": "english",
    "英文": "english",
    "zh": "chinese",
    "zh-cn": "chinese",
    "cn": "chinese",
    "chinese": "chinese",
    "中文": "chinese",
}


def normalize_output_language(value: Any) -> str:
    normalized = OUTPUT_LANGUAGE_ALIASES.get(str(value or "english").strip().lower())
    if normalized is None:
        raise ValueError("agent.output_language must be one of: english, chinese")
    return normalized


def output_language_instruction(value: Any) -> str:
    language = normalize_output_language(value)
    if language == "chinese":
        return (
            "重要：所有面向用户的方案、评审理由、调试建议、总结和洞察都必须使用中文输出。"
            "不要改用英文。代码标识符、API 名称以及必要的技术术语可以保留英文。"
        )
    return (
        "IMPORTANT: Write every user-facing plan, review reason, debug hint, summary, and insight in English. "
        "Do not switch to Chinese. Code identifiers, API names, and necessary technical terms may remain as-is."
    )


def configured_output_language(agent: Any) -> str:
    return normalize_output_language(getattr(getattr(agent, "acfg", None), "output_language", "english"))


def infer_task_family(agent: Any) -> str:
    mode = infer_task_mode(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
        autorealize_context=getattr(agent, "data_preview", "") or getattr(agent, "autorealize_context", ""),
    )
    return "decision" if mode in {"optimization", "rl", "decision"} else "prediction"


def ensure_expansion_profile(
    agent: Any,
    parent_node: Any,
    profile: ExpansionProfile | None,
    operator: str,
) -> ExpansionProfile:
    if profile is None:
        ordinal = max(1, len(getattr(parent_node, "children", set())) + 1)
        profile = ExpansionProfile.create(ordinal, task_family=infer_task_family(agent))
    return profile.with_operator(operator)


def apply_expansion_profile(
    agent: Any,
    node: Any,
    profile: ExpansionProfile,
    operator: str,
) -> None:
    node.sibling_ordinal = profile.sibling_ordinal
    node.expansion_complexity = profile.complexity
    node.expansion_operator = operator
    node.method_family = infer_method_family(
        getattr(node, "plan", ""), code=getattr(node, "code", ""),
        task_family=profile.task_family or infer_task_family(agent),
    )


def _method_symbols(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError):
        return None
    symbols: list[str] = []
    for item in ast.walk(tree):
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            if isinstance(item, ast.ImportFrom) and item.module:
                # sklearn.ensemble is also the module for individual tree estimators.
                symbols.append(item.module.replace("sklearn.ensemble", "sklearn"))
            symbols.extend(alias.name.replace("sklearn.ensemble", "sklearn") for alias in item.names)
        elif isinstance(item, ast.Call):
            symbols.append(ast.unparse(item.func))
        elif isinstance(item, ast.ClassDef):
            symbols.append(item.name)
    source = " ".join(symbols)
    source = re.sub(r"\b(?:ColumnTransformer|FunctionTransformer|PowerTransformer|QuantileTransformer|TransformedTargetRegressor)\b", "preprocessing", source)
    source = re.sub(r"([a-z])([A-Z])", r"\1 \2", source).replace("_", " ")
    return source.lower()


def infer_method_family(text: Any, *, task_family: str = "prediction", code: str = "") -> str:
    source = str(text or "").lower()
    if code:
        symbols = _method_symbols(code)
        if symbols is not None:
            source = symbols
    has_rl = bool(re.search(r"\b(ppo|dqn|sac|td3|actor.?critic|reinforcement learning|q.?learning|policy gradient)\b", source))
    has_opt = bool(re.search(r"\b(milp|mip|linear program|cp.?sat|ortools|dynamic programming|branch.?and.?bound)\b", source))
    has_search = bool(re.search(r"\b(genetic|evolutionary|simulated annealing|tabu|large neighborhood|beam search|mcts)\b", source))
    has_local = bool(re.search(r"\b(local search|hill climb|2.?opt|3.?opt|swap neighborhood|coordinate descent)\b", source))
    has_heuristic = bool(re.search(r"\b(greedy|heuristic|constructive|rule.?based)\b", source))
    if has_rl and (has_opt or has_search or has_local or has_heuristic):
        return "hybrid"
    if has_rl:
        return "reinforcement_learning"
    if task_family == "decision":
        if has_opt:
            return "mathematical_optimization"
        if has_search:
            return "metaheuristic_search"
        if has_local:
            return "local_search"
        if has_heuristic:
            return "heuristic"
        return "decision_other"
    if re.search(r"\b(ensemble|stacking|blending|bagging|voting)\b", source):
        return "ensemble"
    if re.search(r"\b(transformer|cnn|rnn|lstm|neural|pytorch|tensorflow|keras)\b", source):
        return "deep_learning"
    if re.search(r"\b(xgboost|lightgbm|catboost|random forest|gradient boost(?:ing)?)\b", source):
        return "tree_boosting"
    if re.search(r"\b(linear regression|logistic regression|ridge|lasso|svm)\b", source):
        return "linear_or_kernel"
    return "prediction_other"


def method_family_for_node(node: Any, *, task_family: str) -> str:
    code = str(getattr(node, "code", "") or "")
    if code:
        detected = infer_method_family("", task_family=task_family, code=code)
        if detected not in {"prediction_other", "decision_other"}:
            return detected
    explicit = str(getattr(node, "method_family", "") or "").strip()
    if explicit and explicit != "unknown":
        return explicit
    return infer_method_family(
        f"{getattr(node, 'plan', '')}\n{getattr(node, 'code_summary', '')}\n{getattr(node, 'code', '')}",
        task_family=task_family, code=code,
    )


def _node_metric(node: Any) -> tuple[float | None, bool | None]:
    metric = getattr(node, "metric", None)
    value = getattr(metric, "value", None) if metric is not None else None
    direction = getattr(metric, "maximize", None) if metric is not None else None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = None
    return value, direction if isinstance(direction, bool) else None


def _node_record(node: Any, *, task_family: str, direction: bool | None) -> str:
    score, node_direction = _node_metric(node)
    method = method_family_for_node(node, task_family=task_family)
    label = str(getattr(node, "id", ""))[:8] or "unknown"
    plan = trim_long_string(str(getattr(node, "plan", "") or ""), threshold=260, k=140)
    if getattr(node, "is_buggy", None) is True:
        status = "failed"
    elif getattr(node, "is_buggy", None) is None:
        status = "pending"
    else:
        status = "completed"

    parts = [f"- {label} | {method} | {status}"]
    if score is not None and direction is not None and node_direction is direction:
        parts[0] += f" | score {score:.6g}"
    if plan:
        parts.append(f"  Approach: {plan}")
    if status == "failed":
        failure = trim_long_string(
            str(getattr(node, "analysis_for_prompt", "") or getattr(node, "analysis", "") or ""),
            threshold=320,
            k=170,
        )
        if failure and failure != "N/A":
            parts.append(f"  Observed failure: {failure}")
    return "\n".join(parts)


def _ancestors(node: Any) -> list[Any]:
    result: list[Any] = []
    current = node
    seen: set[str] = set()
    while current is not None and str(getattr(current, "id", "")) not in seen:
        seen.add(str(getattr(current, "id", "")))
        result.append(current)
        current = getattr(current, "parent", None)
    return list(reversed(result))


def scoped_search_memory(
    agent: Any,
    parent_node: Any,
    operator: str,
    *,
    source_nodes: Iterable[Any] | None = None,
    max_nodes: int = 8,
) -> str:
    """Return a compact, score-aware evidence summary for the active search operator."""

    family = infer_task_family(agent)
    op = str(operator or "").lower()
    selected: list[Any] = []
    if op == "draft":
        selected = list(getattr(getattr(agent, "virtual_root", None), "children", set()))
    elif op == "improve":
        selected = [parent_node]
        grandparent = getattr(parent_node, "parent", None)
        if grandparent is not None:
            selected.extend(child for child in getattr(grandparent, "children", set()) if child is not parent_node)
    elif op == "evolution":
        selected = _ancestors(parent_node)
        direct_parent = getattr(parent_node, "parent", None)
        if direct_parent is not None:
            selected.extend(child for child in getattr(direct_parent, "children", set()) if child is not parent_node)
    elif op == "debug":
        selected = _ancestors(parent_node)
    elif op in {"fusion", "fusion_draft"}:
        selected = [parent_node, *(source_nodes or [])]
    else:
        selected = [parent_node]

    deduped: list[Any] = []
    seen: set[str] = set()
    for node in selected:
        node_id = str(getattr(node, "id", ""))
        if not node_id or node_id in seen or str(getattr(node, "stage", "")) == "root":
            continue
        seen.add(node_id)
        deduped.append(node)
    directions = {
        direction
        for node in deduped
        if getattr(node, "is_buggy", None) is False
        for score, direction in [_node_metric(node)]
        if score is not None and direction is not None
    }
    _, parent_direction = _node_metric(parent_node)
    direction = parent_direction if parent_direction is not None else (next(iter(directions)) if len(directions) == 1 else None)

    def recent(nodes: list[Any]) -> list[Any]:
        return sorted(
            nodes,
            key=lambda item: (float(getattr(item, "ctime", 0.0) or 0.0), str(getattr(item, "id", ""))),
            reverse=True,
        )

    scored = [
        node for node in deduped
        if getattr(node, "is_buggy", None) is False
        and _node_metric(node)[0] is not None
        and direction is not None
        and _node_metric(node)[1] is direction
    ]
    scored.sort(
        key=lambda item: (
            _node_metric(item)[0] * (1 if direction else -1),
            float(getattr(item, "ctime", 0.0) or 0.0),
            str(getattr(item, "id", "")),
        ),
        reverse=True,
    )
    failed = recent([node for node in deduped if getattr(node, "is_buggy", None) is True])
    pending = recent([node for node in deduped if getattr(node, "is_buggy", None) is None])
    unscored = recent([
        node for node in deduped
        if getattr(node, "is_buggy", None) is False and node not in scored
    ])
    ordered = []
    if parent_node in deduped:
        ordered.append(parent_node)
    ordered.extend(scored[:3])
    ordered.extend(failed[:1])
    ordered.extend(pending[:1])
    ordered.extend(unscored)
    ordered.extend(scored[3:])
    ordered.extend(failed[1:])
    ordered.extend(pending[1:])
    ordered = list(dict.fromkeys(ordered))[:max(1, max_nodes)]

    if not ordered:
        return ""
    if direction is None:
        objective = "Score direction is unresolved; do not compare unverified or mixed-direction scores."
    else:
        objective = f"Score direction: {'higher' if direction else 'lower'} is better."
    heading = f"Search evidence ({len(ordered)} selected attempts). {objective}"
    records = [_node_record(node, task_family=family, direction=direction) for node in ordered]
    return heading + "\n" + "\n".join(records)


def method_family_coverage(agent: Any) -> dict[str, int]:
    family = infer_task_family(agent)
    counts: dict[str, int] = {}
    for node in getattr(getattr(agent, "journal", None), "nodes", []):
        if str(getattr(node, "stage", "")) == "root":
            continue
        name = method_family_for_node(node, task_family=family)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def autonomous_method_selection_guidance() -> list[str]:
    """Stable method-selection rules shared by creative search operators."""

    return [
        (
            "- Select the method only after analyzing the objective structure, constraint coupling, data and instance "
            "scale, sequential or static decision structure, uncertainty, available dependencies, runtime/memory budget, "
            "the frozen evaluator, and evidence from completed search nodes."
        ),
        (
            "- Methods mentioned by the user, cold-start references, retrieved experience cards, and earlier nodes are "
            "candidate hypotheses rather than a whitelist or mandate. Unless the task contract explicitly requires a "
            "method, do not prefer it merely because it appears in context, and freely choose a better unmentioned method."
        ),
        (
            "- The simple/normal/complex profile controls acceptable implementation complexity, runtime, and risk only. "
            "It never assigns an algorithm family. Choose the best-fitting coherent method within that budget."
        ),
        (
            "- Do not add complexity for its own sake. State internally why the chosen method fits this problem and why "
            "plausible alternatives are less suitable under the current evidence and budget."
        ),
    ]


def dynamic_expansion_instruction(
    agent: Any,
    profile: ExpansionProfile,
    *,
    operator: str | None = None,
) -> str:
    op = str(operator or profile.operator or "auto").lower()
    family = profile.task_family or infer_task_family(agent)
    coverage = json.dumps(method_family_coverage(agent), ensure_ascii=False, sort_keys=True)
    operator_rules = {
        "draft": "Create a genuinely new method trajectory, not a parameter-only variant of an existing draft.",
        "improve": "Make one evidence-backed coherent change that targets the diagnosed bottleneck; do not anchor on methods named in context.",
        "evolution": "Make a deeper coherent structural change supported by branch evidence; preserve the trusted evaluator.",
        "debug": "Repair only the evidenced defect. Do not turn debugging into a new method proposal.",
        "fusion": "Transfer one complementary technique from the selected source; do not concatenate whole solutions.",
        "fusion_draft": "Synthesize complementary evidence across successful families into one coherent new trajectory.",
    }
    if family == "decision":
        complexity_rules = {
            "simple": (
                "Choose the best-fitting fast, inspectable approach with a short feedback loop and low implementation risk. "
                "Simple describes the execution budget, not the algorithm family."
            ),
            "normal": (
                "Choose a materially stronger or different approach when the problem structure and prior evidence justify it, "
                "while keeping implementation and runtime bounded."
            ),
            "complex": (
                "Permit a high-upside, higher-cost approach, but select it from the actual problem structure rather than from "
                "methods named in the prompt. Confirm that the remaining runtime and evaluator make the attempt credible."
            ),
        }
    else:
        complexity_rules = {
            "simple": (
                "Get one complete, reliable method running with a short feedback loop, leakage-safe validation, and a reusable interface. "
                "Use a single bounded training recipe: no multi-model tournament, stacking, broad grid or nested CV unless the task explicitly requires it. "
                "A small neural model is allowed when appropriate; simple limits experiment cost, not model family. "
                "Compare heavier alternatives in subsequent search nodes after the first runnable result."
            ),
            "normal": "Choose a materially stronger or different approach supported by task structure and completed-node evidence.",
            "complex": "Permit a high-upside, higher-cost approach only when evidence, validation, and remaining runtime justify it.",
        }
    return (
        "# Latest expansion control (dynamic; authoritative for this child)\n"
        f"- Task family: {family}\n"
        f"- Operator: {op}\n"
        f"- Parent-local sibling ordinal: {profile.sibling_ordinal}\n"
        f"- Complexity profile: {profile.complexity}\n"
        f"- Existing method-family coverage: {coverage}\n"
        f"- Operator contract: {operator_rules.get(op, operator_rules['improve'])}\n"
        f"- Complexity guidance: {complexity_rules[profile.complexity]}\n"
        f"- {output_language_instruction(configured_output_language(agent))}"
    )
