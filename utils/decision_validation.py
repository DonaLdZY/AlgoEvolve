"""Lightweight parsing helpers for optimization/RL decision validation output."""

from __future__ import annotations

import json
import re


DECISION_SUMMARY_PREFIX_RE = re.compile(
    r"^\s*Decision\s+Validation\s+Summary\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE,
)

OPTIMIZATION_RL_KEYWORDS = (
    "reinforcement learning",
    "offline rl",
    "online rl",
    "mdp",
    "markov decision",
    "policy learning",
    "reward function",
    "gymnasium",
    "gym env",
    "environment step",
    "simulator",
    "sequential decision",
    "dynamic decision",
    "routing",
    "vehicle routing",
    "scheduling",
    "assignment problem",
    "resource allocation",
    "portfolio optimization",
    "knapsack",
    "combinatorial optimization",
    "optimization problem",
    "constraint solver",
    "cp-sat",
    "mixed integer",
    "integer programming",
    "linear programming",
    "decision problem",
    "decision optimization",
    "vehicle dispatch",
    "dispatching",
    "capacity constraint",
    "feasible solution",
    "hard constraint",
    "local search",
    "simulated annealing",
    "tabu search",
    "large neighborhood search",
    "minimize",
    "maximize",
    "objective",
    "constraint",
    "penalty",
    "强化学习",
    "离线强化学习",
    "在线强化学习",
    "马尔可夫决策",
    "状态空间",
    "动作空间",
    "奖励函数",
    "策略学习",
    "仿真环境",
    "序贯决策",
    "路径规划",
    "路径优化",
    "车辆路径",
    "车辆调度",
    "配送调度",
    "调度",
    "排程",
    "分配问题",
    "资源分配",
    "组合优化",
    "运筹优化",
    "整数规划",
    "线性规划",
    "约束求解",
    "可行解",
    "硬约束",
    "优化",
    "决策",
    "最小化",
    "最大化",
    "目标函数",
    "罚分",
    "惩罚",
    "约束",
)


def is_optimization_or_rl_text(task_desc: str = "", coldstart_description: str = "") -> bool:
    """Lightweight task-type detector that does not import prompt/LLM modules."""
    raw = f"{task_desc}\n{coldstart_description}"
    text = raw.lower()
    if not text.strip():
        return False
    if "model" in text and "optimization" in text:
        return True
    return any(keyword in text for keyword in OPTIMIZATION_RL_KEYWORDS)


def parse_bool_like(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "ok", "pass", "passed", "1"}:
            return True
        if normalized in {"false", "no", "fail", "failed", "0"}:
            return False
    return None


def extract_decision_validation_summary(text: str) -> dict | None:
    """Extract the last JSON Decision Validation Summary line from execution output."""
    summaries: list[dict] = []
    for line in (text or "").splitlines():
        match = DECISION_SUMMARY_PREFIX_RE.match(line)
        if not match:
            continue
        payload = match.group(1).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            summaries.append(parsed)
    return summaries[-1] if summaries else None


def decision_signal_summary(summary: dict | None) -> dict:
    """Return compact candidate-reported signals for logs and reviewer prompts.

    This function only labels values found in candidate output. It deliberately
    does not infer trust, validity, feasibility, or search eligibility.
    """
    if not isinstance(summary, dict):
        return {}
    score_components = summary.get("score_components")
    if not isinstance(score_components, dict):
        score_components = {}
    signals = {}
    score_source = summary.get("final_score_source")
    if score_source not in (None, "", [], {}):
        signals["reported_final_score_source"] = score_source
    if score_components:
        signals["reported_score_component_count"] = len(score_components)
    if "evaluator_self_tests_passed" in summary:
        signals["reported_evaluator_self_tests_passed"] = summary.get(
            "evaluator_self_tests_passed"
        )
    reported_feasible = _bool_from_keys(summary, ("is_feasible", "feasible"))
    if reported_feasible is not None:
        signals["reported_feasible"] = reported_feasible
    reported_valid = _bool_from_keys(
        summary,
        ("contract_valid", "validation_passed", "is_valid", "valid"),
    )
    if reported_valid is not None:
        signals["reported_valid"] = reported_valid
    violations = summary.get("violations")
    if isinstance(violations, (list, tuple, set)):
        signals["reported_violation_count"] = len(violations)
    elif violations not in (None, "", {}):
        signals["reported_violation_count"] = 1
    return signals


def _bool_from_keys(summary: dict, keys: tuple[str, ...]) -> bool | None:
    for key in keys:
        if key not in summary:
            continue
        parsed = parse_bool_like(summary.get(key))
        if parsed is not None:
            return parsed
    return None
