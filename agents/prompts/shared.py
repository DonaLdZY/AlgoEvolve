"""Shared, task-scoped generation and evaluation contracts."""

from __future__ import annotations


PLAN_AND_CODE_RESPONSE_FORMAT = (
    "Return exactly two items: (1) `Plan: ...` with one sentence of at most 24 words; "
    "(2) exactly one fenced Python code block containing the complete implementation. "
    "Do not reveal analysis, alternatives, examples, extra code fences, or text after the code block."
)


SEARCH_EVALUATION_GUIDANCE = (
    "This is one candidate inside the outer search engine; the engine owns the global search budget, worker admission, resource limits and checkpoints. "
    "Implement one coherent method per candidate and finish within its per-node timeout; the search tree compares methods. "
    "Compute Final Validation Score only on the frozen development population with the task's exact metric, formula and direction. "
    "Trace the printed scalar to its prediction population, row IDs, split membership and evaluator; a printed claim alone is not proof. "
    "Never access a reserved final holdout/test population during candidate search. "
)


FORMAL_SEARCH_EVALUATION_GUIDANCE = (
    "In AutoRealize contracts, primary_metric/validation_protocol define search; final_evaluation_metric/final_evaluation_protocol define post-selection audit. "
    "When task reserves final holdout, do not evaluate, print metrics, fit on, or use it during candidate execution. "
    "Formal AutoRealize tasks must define final_evaluate(data, artifact_dir), never called by default search. "
    "It receives complete input and frozen artifact dirs, evaluates selected recipe once, returns metrics without retraining/refitting. "
    "Never write holdout predictions or evidence into artifact_dir. "
    "Before final_evaluate, engine copies ./working/artifact to artifact_dir root. Load frozen files from artifact_dir only. "
    "When status='completed', return JSON dict with metrics (finite scalars), model_path (relative), inference_checks. "
    "When no final evaluation applies, return status='not_applicable' with reason. "
    "Make imports side-effect-free: training under if __name__ == '__main__'. "
    "Put reusable models/preprocessing in ./working/artifact; dev-only data in ./working/evidence. "
    "final_evaluate receives frozen artifact at root; never expect artifact_dir/artifact or read other candidates. "
    "Persist an Evaluation Protocol Summary with evaluation_phase='search', scoring_split, final_evaluation_accessed=false. "
    "Write that JSON to ./working/evidence/evaluation_protocol.json (exact filename, at most 16 KiB), "
    "including the supplied contract_id and evaluation_id. Put full row-ID lists in a separate file; split_fingerprint is a digest string. "
    "Do not add unrequested stdout lines. "
    "Package selected recipe so train() works in fresh directory. "
    "Keep predict(model_path, data) as importable API."
)


FUSION_EVALUATION_GUIDANCE = [
    "Choose technique transfer or ensemble based on complementary errors, constraints, runtime. Name sources and operation.",
    "For blending/stacking: align members' OOF predictions by IDs, folds, columns, evaluation_id. Reject incompatible.",
    "Fit blend weights/meta-models on train-fold OOF or calibration split, never on reported dev rows.",
    "For temporal: purge label-horizon overlap using contract.",
    "Evaluate combined model on same population, metric, constraints as members.",
    "For decisions: transfer solver component or generate new feasible solution. Recheck constraints.",
    "Package all components in portable artifact with relative paths and one inference entrypoint.",
]


def plan_and_code_response_format(scope: str = "the requested implementation") -> str:
    """One canonical response contract for all full-code generation prompts."""
    return f"{PLAN_AND_CODE_RESPONSE_FORMAT} The code block must contain only {scope}."


OPTIMIZATION_RL_KEYWORDS = (
    "reinforcement learning", "offline rl", "online rl", "mdp", "markov decision",
    "policy learning", "reward function", "gymnasium", "gym env", "environment step", "simulator",
    "sequential decision", "dynamic decision", "routing", "vehicle routing", "scheduling",
    "assignment problem", "resource allocation", "portfolio optimization", "knapsack",
    "combinatorial optimization", "optimization problem", "constraint solver", "cp-sat",
    "mixed integer", "integer programming", "linear programming", "decision problem",
    "decision optimization", "vehicle dispatch", "dispatching", "capacity constraint",
    "feasible solution", "hard constraint", "local search", "simulated annealing",
    "tabu search", "large neighborhood search",
    "强化学习", "离线强化学习", "在线强化学习", "马尔可夫决策", "状态空间", "动作空间",
    "奖励函数", "策略学习", "仿真环境", "序贯决策", "路径规划", "路径优化",
    "车辆路径", "车辆调度", "配送调度", "调度", "排程", "分配问题", "资源分配",
    "组合优化", "运筹优化", "整数规划", "线性规划", "约束求解", "可行解", "硬约束",
)

RL_METHOD_KEYWORDS = (
    "reinforcement learning", "offline rl", "online rl", "policy learning",
    "reward function", "gymnasium", "gym env",
    "强化学习", "离线强化学习", "在线强化学习",
)


def is_optimization_or_rl_task(task_desc: str = "", coldstart_description: str = "") -> bool:
    """Return True only when the task text strongly suggests optimization/RL."""
    text = f"{task_desc}\n{coldstart_description}".lower()
    if not text.strip():
        return False
    if "Model" in coldstart_description and "Optimization" in coldstart_description:
        return True
    return any(keyword in text for keyword in OPTIMIZATION_RL_KEYWORDS)


def infer_task_mode(
    task_desc: str = "",
    coldstart_description: str = "",
    autorealize_context: str = "",
) -> str:
    """Route generation by problem structure and required method."""
    if autorealize_context:
        from utils.autorealize_context import infer_autorealize_task_mode
        structured_mode = infer_autorealize_task_mode(autorealize_context)
        if structured_mode:
            return structured_mode

    text = f"{task_desc}\n{coldstart_description}".lower()
    if any(keyword in text for keyword in RL_METHOD_KEYWORDS):
        return "rl"
    if is_optimization_or_rl_task(task_desc, coldstart_description):
        return "optimization"
    return "prediction"


def get_optimization_rl_strategy(task_desc: str = "", coldstart_description: str = "", *, enabled: bool | None = None) -> dict:
    """Concise guidance for optimization and RL tasks."""
    if enabled is False or (enabled is None and not is_optimization_or_rl_task(task_desc, coldstart_description)):
        return {}

    return {
        "Optimization / RL Strategy (conditional)": [
            "",
            "**Problem Family**:",
            "- Static assignment/routing/scheduling: MILP/CP-SAT/OR-Tools, or local search/SA/tabu/GA/LNS.",
            "- Sequential decision with simulator: RL formulation.",
            "- Static with RL required: construct episodes, define state/action/reward aligned with evaluator.",
            "- Historical decisions only: offline RL or imitation learning.",
            "- For static: check if nonlinear terms allow enumeration/MILP/DP before local heuristic.",
            "- A small bounded integer aggregate state can enable exact enumeration or DP; reconstruct and validate original decisions.",
            "- Fast heuristic = valuable incumbent. Get relaxation/bound for pruning if possible.",
            "- Never equate an incumbent with a proof of optimality; report solver status and bound gap.",
            "",
            "**RL Requirements** (if chosen):",
            "- State: info at decision time (no future leakage).",
            "- Action: discrete/continuous/composite; use feasibility masks, prefer masking before selection.",
            "- Transition: how action changes state (inventory, capacity, time, etc.).",
            "- Reward: aligned with official metric; penalties for violations.",
            "- Terminal: success, infeasible, timeout, horizon.",
            "- Evaluation: scenarios, seeds, diagnostics, official metric.",
            "",
            "**Algorithms**: Discrete: DQN/Rainbow. General: PPO. Continuous: SAC/TD3. Offline: BC/CQL/IQL/TD3+BC/DT.",
            "",
            "**Implementation**:",
            "- Gymnasium env: reset(seed, options), step(action) → (obs, reward, terminated, truncated, info).",
            "- Combinatorial: valid_action_mask(obs); check mask.any() before selection. An all-invalid mask requires fallback/backtrack/terminate.",
            "- One method per node. RL claim requires policy rollout, not placeholder.",
            "",
            "**Curriculum** (optional): Small→full, relaxed→original, demos→RL. Advance by success/feasible rate, not steps. Evaluate the saved policy on the original full-scale environment.",
            "",
        ]
    }


def get_decision_solution_protocol(task_desc: str = "", coldstart_description: str = "", *, enabled: bool | None = None) -> dict:
    """Concise protocol for decision/optimization tasks."""
    if enabled is False or (enabled is None and not is_optimization_or_rl_task(task_desc, coldstart_description)):
        return {}

    return {
        "Decision / Optimization Solution Protocol (conditional, strict)": [
            "",
            "**1. Evaluator First**",
            "- Required: load_problem_data(), build_solution(), validate_solution(), score_solution(), run_evaluator_self_tests().",
            "- Use exact AutoRealize schema column names. Resolve aliases explicitly before pandas ops.",
            "- Define eval population before scoring. Handle missing per contract (report, fallback, default).",
            "- validate_solution: feasibility, completeness, schema, IDs, constraints from task/contract.",
            "- score_solution: ONE finite scalar from task contract. No reward-only, loss, or proxy.",
            "- Empty/placeholder/constant invalid unless authorized.",
            "- If contract incomplete, report missing formula/parameter for repair.",
            "",
            "**2. Solution Artifact**",
            "- Non-RL: solve(model_path, data). Stateless: model_path=None ok.",
            "- RL: train_policy(data, artifact_dir) + rollout(model_path, data). No retraining in rollout.",
            "- Artifact: config, model, policy, preprocessing, params sufficient to reproduce without retraining.",
            "- Build feasible candidates with constraint masks. Empty mask: record infeasible/fallback/terminate, never crash/loop.",
            "- Output columns = generated results, NOT input. Define OUTPUT_COLUMNS, use pd.DataFrame(rows, columns=OUTPUT_COLUMNS).",
            "- Empty/infeasible handled by validate+score, produce schema-correct empty table if submission enabled.",
            "",
            "**3. Constraints Gate**",
            "- Violated → mark infeasible, apply penalty, or worst score. Never let infeasible look best.",
            "- Feasibility from validate_solution(), not \"code ran\".",
            "",
            "**4. Output**",
            "- Optional: one JSON Decision Validation Summary with task diagnostics.",
            "- MUST: Final score from score_solution() after validation and invalid handling.",
            "",
        ]
    }


ROBUSTNESS_GENERALIZATION_STRATEGY = {
    "💡 Robustness & Generalization": [
        "",
        "**Improve robustness on unseen data**:",
        "- Architecture: match inductive bias to data (CNN/ViT for grids, Transformer/RNN for sequences, GNN for graphs).",
        "- Input: windowing or patches for variable-length/large-scale.",
        "- Regularization: Dropout, Batch/Layer Norm, Weight Decay, Label Smoothing.",
        "- Loss: adapt to class distribution (weighted, Focal, task-specific).",
        "- LR: adaptive schedules (Cosine, ReduceLROnPlateau, Warmup).",
        "- Augmentation: domain-appropriate (geometric, masking, mixup).",
        "- Validation: monitor strictly, use early stopping.",
        "- Note: prioritize capturing data structure over model size.",
        "",
    ]
}


def prompt_leakage_prevention():
    """Data leakage prevention."""
    return {
        "🚨 DATA LEAKAGE PREVENTION": [
            "",
            "**Strict Isolation**: Val/Test unseen during training.",
            "- Sequence: Split FIRST, then process.",
            "- Stateful transforms: fit on Train only, .transform() on Val/Test.",
            "- Features: stats from Train set only.",
            "- Target: never use Val target info.",
            "",
        ]
    }


def prompt_resp_fmt():
    """Response format for plan + code."""
    return {
        "Response format": plan_and_code_response_format(
            "the complete runnable solution, including the required validation metric output"
        )
    }


def get_internet_clarification(pretrain_model_dir: str = ""):
    """Network and resource policy."""
    lines = [
        "**Network/Resource Policy**",
        "- Follow the active task, execution contract and sandbox network policy. Do not reinterpret a network restriction as permission.",
        "- Use external resources only through explicitly enabled tools.",
    ]
    if pretrain_model_dir:
        lines.append(f"- {pretrain_model_dir} is configured, not proof that the files exist or load. Verify before depending on them.")
    return lines
