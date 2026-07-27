"""Static shared prompt fragments."""

from __future__ import annotations


PLAN_AND_CODE_RESPONSE_FORMAT = (
    "Return only these two items in this order: "
    "(1) a 1-3 sentence implementation plan, then "
    "(2) exactly one fenced Python code block. "
    "Do not add headings, extra code fences, or text after the code block."
)


def plan_and_code_response_format(scope: str = "the requested implementation") -> str:
    """One canonical response contract for all full-code generation prompts."""

    return f"{PLAN_AND_CODE_RESPONSE_FORMAT} The code block must contain only {scope}."


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
)

RL_METHOD_KEYWORDS = (
    "reinforcement learning",
    "offline rl",
    "online rl",
    "policy learning",
    "reward function",
    "gymnasium",
    "gym env",
    "强化学习",
    "离线强化学习",
    "在线强化学习",
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
    """Route generation by problem structure and required method.

    AutoRealize's structured contract is authoritative. Keyword fallback is
    used only when that contract is unavailable.
    """

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


def get_optimization_rl_strategy(task_desc: str = "", coldstart_description: str = "") -> dict:
    """Conditional guidance for optimization and reinforcement-learning tasks."""
    if not is_optimization_or_rl_task(task_desc, coldstart_description):
        return {}

    return {
        "Optimization / Reinforcement Learning Strategy (conditional)": [
            "",
            "**Activate this section only because the task appears to involve optimization, sequential decision-making, or RL. Problem structure and solution method are separate: a static optimization problem may still require an RL method constructed from static data. Do not force RL when it is neither required nor justified.**",
            "",
            "**First decide the problem family:**",
            "- Static one-shot assignment/routing/scheduling/resource allocation: consider MILP/CP-SAT/OR-Tools when constraints are explicit, or local search, simulated annealing, tabu search, genetic/evolutionary search, or large-neighborhood search.",
            "- Sequential decision task with a simulator, transition dynamics, delayed rewards, or logged trajectories: consider an RL formulation.",
            "- Static optimization with RL required: construct episodes from problem instances or subproblems, define partial-solution state transitions and legal-action masks, and align terminal reward with the same evaluator used for every method.",
            "- If there are only historical decisions and no safe simulator for exploration, treat it as offline RL or imitation learning; do not use naive online exploration.",
            "- For static optimization, explicitly check whether nonlinear terms depend on a small bounded integer aggregate state or short local state transition. When they do, consider enumeration/linearization, flow-consistency constraints, MILP/CP-SAT, or dynamic programming before defaulting to a purely local heuristic.",
            "- Treat a fast feasible heuristic as a potentially valuable incumbent and upper bound. When possible, obtain a relaxation/lower bound, use the bound gap for safe variable or state pruning, and consider fix-and-optimize or exact large-neighborhood search around the incumbent.",
            "- Exact reformulation is conditional, not mandatory. Reject it when state ranges, global nonlinear dependence, missing evaluator facts, solver availability, memory, or runtime make it unsuitable.",
            "- Never equate an incumbent with a proof. Preserve solver status, best bound, gap, model size, warm-start source, and independent evaluator replay when those facts are available.",
            "",
            "**If proposing RL, the plan must explicitly define:**",
            "- State / observation: all information available at decision time only; no future leakage.",
            "- Action space: discrete, continuous, or composite actions; include feasibility masks or repair rules for invalid actions. Prefer masking illegal actions before selection rather than selecting first and rejecting later.",
            "- Transition: how an action changes the current partial solution, inventory, capacity, time, position, budget, or other state variables.",
            "- Reward: aligned with the official objective/evaluation metric; include penalties for constraint violations and optional dense shaping that preserves the final objective.",
            "- Terminal condition: when an episode ends, including success, infeasible dead-end, timeout, horizon, or completed assignment.",
            "- Evaluation protocol: validation scenarios, seeds, constraint diagnostics, and final objective/metric.",
            "",
            "**Algorithm recommendations by action/data setting:**",
            "- Discrete online RL: DQN/Rainbow DQN for manageable discrete actions; Rainbow combines Double DQN, prioritized replay, dueling networks, multi-step returns, distributional RL, and noisy exploration.",
            "- General online RL: PPO is a robust default when the environment is available and actions may be discrete or continuous; add action masks/feasibility repair for combinatorial tasks.",
            "- Continuous online RL: SAC is often a strong sample-efficient off-policy default; TD3 is a strong deterministic actor-critic option.",
            "- Offline RL: consider behavior cloning, CQL, IQL, TD3+BC, or Decision Transformer when logged trajectories are available.",
            "- Model-based/world-model methods such as Dreamer-style agents are powerful but usually too complex unless the task provides a rich simulator and enough runtime.",
            "",
            "**Implementation expectations if RL is chosen:**",
            "- Implement a Gymnasium-like environment with `reset(seed=None, options=None)` and `step(action)` returning `(obs, reward, terminated, truncated, info)`.",
            "- Keep reward calculation and constraint checking as deterministic, testable functions.",
            "- For combinatorial actions, expose `valid_action_mask(obs)` and apply it in policy sampling / greedy action selection. Check `mask.any()` before masked logits or softmax: an all-invalid mask must take a deterministic task-valid fallback, backtrack/repair, or terminate with an infeasible reason; never softmax all `-inf`/large-negative logits, create NaN probabilities, or silently pick an illegal action.",
            "- A search node should represent one coherent method. Do not hide a competing internal comparison method inside an RL node; AlgoEvolve compares different solution nodes by their final scalar scores.",
            "- If you claim the node uses RL, the evaluated solution must come from environment interaction plus policy selection/training/configured rollout, and the saved artifact plus `rollout` must reproduce that rollout without retraining.",
            "",
            "**Curriculum learning when RL is chosen:**",
            "- Explicitly decide whether curriculum learning is useful. Suitable progressions include small to full instances, relaxed to original constraints, short to full horizons, or heuristic demonstrations / behavior cloning followed by RL fine-tuning. If none is suitable, explain the concrete reason instead of forcing a curriculum.",
            "- Define level advancement with measured thresholds such as episode success rate, feasible-solution rate, or evaluator score over a fixed validation window; do not advance by training step count alone when the policy is still failing.",
            "- Keep state/action semantics and the final objective consistent across levels. Dense shaping and relaxed constraints may aid learning but must not replace the official terminal objective or hide infeasibility.",
            "- End training and model selection on the original full-scale environment, original hard constraints, complete horizon, and official evaluator. A curriculum-level score is never the final comparable score.",
            "",
        ]
    }


def get_decision_solution_protocol(task_desc: str = "", coldstart_description: str = "") -> dict:
    """Strict protocol for optimization, RL, and decision-output tasks."""
    if not is_optimization_or_rl_task(task_desc, coldstart_description):
        return {}

    return {
        "Decision / Optimization Solution Protocol (conditional, strict)": [
            "",
            "Use this protocol because the task appears to require decisions, assignments, schedules, routes, actions, or an optimized plan rather than only supervised predictions.",
            "",
            "**1. Freeze the evaluator before optimizing.**",
            "- Implement `load_problem_data(input_dir)`, `build_solution(data)`, `validate_solution(solution, data)`, `score_solution(solution, data)`, and `run_evaluator_self_tests(data)`.",
            "- If AutoRealize provides an Exact Source Schema Contract, all raw pandas `sheet_name` and column access must use exact names from that contract. Natural-language field meanings, business aliases, and English variable names are not raw dataframe columns.",
            "- If a task concept name is not an exact physical column, resolve it by explicit alias mapping against available columns before use. If no reliable match exists, fail with a diagnostic listing available columns; never hard-code the absent alias in `groupby`, `agg`, joins, filters, or renames.",
            "- Before any `groupby`, `agg`, merge, filter, or sort that uses a described business concept, bind a local variable to the resolved exact source column and use that variable. If you want a semantic/code-local column name, create it only after reading from the exact source column.",
            "- Define the evaluation population before scoring. Do not silently drop orders/items/jobs/resources because a time, target, capacity, or feature field is missing. If the task/AutoRealize contract defines eligibility or exclusion rules for records lacking mandatory evaluation fields, apply those rules explicitly and report excluded/missing counts and examples in validation details; otherwise use a documented fallback field, an `UNKNOWN`/default bucket, or a conservative assumption.",
            "- `validate_solution` must check the task-defined feasibility, completeness, uniqueness, schema, unknown IDs/entities, capacity/time/budget/inventory rules, and any leakage guard stated in the task or AutoRealize context.",
            "- `score_solution` must compute the single scalar objective from the task description or AutoRealize evaluation contract. Do not invent a second metric, reward-only metric, training loss, or proxy objective as the reported final score.",
            "- If the official objective is incomplete, implement the most conservative scalar explicitly supported by the task text and state missing assumptions in the validation summary; do not silently optimize an unrelated score.",
            "",
            "**2. A solution is an artifact, not just a model output.**",
            "- Non-RL decision methods expose `solve(model_path, data)`; a stateless heuristic/rule/exact solver may accept `model_path=None`.",
            "- RL or hybrid methods expose `train_policy(data, artifact_dir)` and `rollout(model_path, data)`. The evaluated rollout must load the saved policy without retraining.",
            "- The saved artifact may be a solver configuration, fitted estimator, policy checkpoint, preprocessing state, learned cost model, or heuristic parameters, but it must be sufficient to reproduce solve/rollout without retraining.",
            "- Static assignment/routing/scheduling problems may use repair/local search/OR methods. Use RL only when a real state/action/transition/reward formulation is natural and evaluable, or when the task explicitly requests an RL/hybrid RL solution.",
            "- For optimization and RL, build feasible candidates with a hard-constraint mask before scoring or sampling actions. The mask should cover known contract/route/resource availability, capacity, time-window, uniqueness, inventory, budget, and schema/entity validity constraints when they apply.",
            "- If the feasible-candidate mask is empty for an item/job/state, handle it explicitly according to the task contract: record an infeasible/undecided case with a reason, try an allowed repair/backtracking/new-resource fallback, or terminate the episode with an infeasible flag. Do not crash, loop forever, or choose a known-illegal action just to keep going.",
            "- Output/submission columns are generated result-schema columns, not raw input dataframe columns. Never do `raw_df[output_columns]`, and never assume output names such as solution name, delivery day, trip id, carrier, vehicle type, total cost, validity, or failure reason exist in source tables unless the Exact Source Schema Contract lists them exactly.",
            "- When building an output table from generated rows/actions/trips, define a constant such as `OUTPUT_COLUMNS = [...]` and construct it with `pd.DataFrame(rows, columns=OUTPUT_COLUMNS)`. Do not create `pd.DataFrame(rows)` and then slice `[OUTPUT_COLUMNS]`, because empty solutions produce a zero-column dataframe and crash before validation evidence is printed.",
            "- Empty, infeasible, or diagnostic solutions must be handled by `validate_solution` and `score_solution`, not by a pandas crash. They should still produce a schema-correct empty output table when submission generation is enabled, plus an actionable validation summary with task-defined diagnostics.",
            "",
            "**3. Hard constraints gate the metric.**",
            "- If hard constraints are violated, the run must either mark the solution infeasible and apply the official invalid-solution penalty, or report the worst score. Never let an infeasible solution look best.",
            "- Empty, no-op, duplicate, constant, placeholder, or unknown-ID solutions are invalid unless the task explicitly defines them as valid and penalized.",
            "- Do not claim feasibility just because code executed. Feasibility must come from `validate_solution`.",
            "",
            "**4. Execution evidence for improvement.**",
            "- The result reviewer decides whether the returned solution and its `Final Validation Score` are valid and comparable under this task contract; candidate-printed summaries are supporting evidence, never an automatic pass.",
            "- When useful, print one JSON line prefixed with `Decision Validation Summary:` from the validator / evaluator report.",
            "- If you print the JSON, include task-relevant diagnostics, but do not invent universal progress or violation fields for tasks that do not define them.",
            "- If validation is not clean and you print diagnostics, include actionable details using the task's own concepts, such as invalid action examples, infeasibility reasons, duplicate IDs, unknown IDs, violated constraints, or objective-component diagnostics.",
            "- A partial, infeasible, diagnostic, or empty solution can still be useful when the official scalar score handles that case and the summary explains it.",
            "- Use `json.dumps(summary, ensure_ascii=False, sort_keys=True)` so the result reviewer can inspect it.",
            "- The final score must be the value returned by `score_solution`, after validation and official invalid-solution handling.",
            "",
        ]
    }


ROBUSTNESS_GENERALIZATION_STRATEGY = {
    "💡 Recommendation: Robustness & Generalization Strategy": [
        "",
        "**To improve model robustness and generalization on unseen data:**",
        "",
        "✅ **Architecture**: Match model inductive bias to data structure (e.g., CNNs/ViTs for spatial grids, Transformers/RNNs for sequences, GNNs/GCNs for graphs/topology)",
        "✅ **Input Strategy**: Handle variable-length or large-scale inputs via **windowing strategies** or patch-based processing (consider overlap for smoother predictions)",
        "✅ **Regularization**: Consider using Dropout, Batch/Layer Norm, Weight Decay, or Label Smoothing",
        "✅ **Loss Function**: Inspect class distribution and adapt loss accordingly (e.g., weighted loss, FocalLoss, or task-specific objectives)",
        "✅ **Learning Rate**: Consider using adaptive schedules like Cosine Annealing or ReduceLROnPlateau or Warmup with differential rates if needed",
        "✅ **Data Augmentation**: Apply domain-appropriate augmentation based on data modality (e.g., geometric transforms, masking, mixup)",
        "✅ **Validation**: Monitor validation metrics strictly and use early stopping to prevent overfitting",
        "",
        "⚠️ **Note**:",
        "Prioritize capturing the intrinsic structure of the data (Inductive Bias) over simply increasing model size.",
        "",
    ]
}


def prompt_leakage_prevention():
    """Data leakage prevention."""
    return {
        "🚨 DATA LEAKAGE PREVENTION": [
            "",
            "⚠️ **Strict Isolation Principle**: Validation/Test data must remain strictly unseen during training.",
            "",
            "✅ **Sequence**: Always **Split Data FIRST**, then apply processing.",
            "✅ **Stateful Transformations**: Fit all Scalers, Encoders, Imputers, and Tokenizers **ONLY on Training data**, then use `.transform()` on Validation/Test.",
            "✅ **Feature Engineering**: Calculate global statistics (e.g., mean, variance, vocabulary) solely from the Training set.",
            "✅ **Target Leakage**: Never use target information (e.g., Target Encoding) from the validation set.",
            "",
        ]
    }


def prompt_resp_fmt():
    """Response format for plan + code"""
    return {
        "Response format": plan_and_code_response_format(
            "the complete runnable solution, including the required validation metric output"
        )
    }


def get_internet_clarification(pretrain_model_dir: str = ""):
    """Internet access clarification for improve/debug stages."""
    lines = [
        "**⚠️ IMPORTANT: Internet Access During Code Development**",
        "- The \"no internet access\" restriction mentioned in the task description applies **ONLY to submission evaluation after code generation** (for mle-bench test set).",
        "- **During code development, you CAN and SHOULD use online resources** such as torch.hub.load(), HuggingFace transformers, timm, etc.",
    ]
    if pretrain_model_dir:
        lines.append(
            f"- **Model paths under `{pretrain_model_dir}/` are GUARANTEED to exist and be available** (e.g., DINOv3, Siglip2 etc.). You can directly use them without `Path question`."
        )
    lines.append(
        "- **Do NOT question internet access concerns - all standard ML libraries and pretrained models are available during development."
    )
    return lines
