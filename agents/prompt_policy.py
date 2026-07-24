"""Shared prompt policy for language, task/method taxonomy, and scoped search memory."""

from __future__ import annotations

import json
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
        f"{getattr(node, 'plan', '')}\n{getattr(node, 'code', '')}",
        task_family=profile.task_family or infer_task_family(agent),
    )


def infer_method_family(text: Any, *, task_family: str = "prediction") -> str:
    source = str(text or "").lower()
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
    if re.search(r"\b(xgboost|lightgbm|catboost|random forest|gradient boost)\b", source):
        return "tree_boosting"
    if re.search(r"\b(linear regression|logistic regression|ridge|lasso|svm)\b", source):
        return "linear_or_kernel"
    return "prediction_other"


def method_family_for_node(node: Any, *, task_family: str) -> str:
    explicit = str(getattr(node, "method_family", "") or "").strip()
    if explicit and explicit != "unknown":
        return explicit
    return infer_method_family(
        f"{getattr(node, 'plan', '')}\n{getattr(node, 'code_summary', '')}\n{getattr(node, 'code', '')}",
        task_family=task_family,
    )


def _node_record(node: Any, *, task_family: str) -> dict[str, Any]:
    metric = getattr(node, "metric", None)
    return {
        "node_id": str(getattr(node, "id", "")),
        "operator": str(getattr(node, "stage", "unknown")),
        "sibling_ordinal": getattr(node, "sibling_ordinal", None),
        "complexity": str(getattr(node, "expansion_complexity", "unknown")),
        "method_family": method_family_for_node(node, task_family=task_family),
        "score": getattr(metric, "value", None) if metric is not None else None,
        "runtime_seconds": getattr(node, "exec_time", None),
        "change": trim_long_string(str(getattr(node, "plan", "") or ""), threshold=700, k=350),
        "failure_evidence": trim_long_string(
            str(getattr(node, "analysis_for_prompt", "") or getattr(node, "analysis", "") or ""),
            threshold=900,
            k=450,
        ),
    }


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
    max_nodes: int = 12,
) -> str:
    """Return operator-specific evidence rather than a global unbounded transcript."""

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
    deduped = sorted(deduped, key=lambda item: float(getattr(item, "ctime", 0.0) or 0.0))[-max_nodes:]
    records = [_node_record(node, task_family=family) for node in deduped]
    return json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2, default=str)


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
            "simple": "Choose the best-fitting reliable approach with a short feedback loop, leakage-safe validation, and a reusable interface.",
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
