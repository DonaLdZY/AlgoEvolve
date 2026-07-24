"""Improve Agent: generate improved plan/code from a successful parent node (diff or full mode)."""

import logging
from typing import Any

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from utils.response import wrap_code
from agents.triggers import get_patience_counter, register_node
from agents.prompts import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    prompt_leakage_prevention,
    prompt_resp_fmt,
    get_internet_clarification,
    get_impl_guideline_from_agent,
    infer_task_mode,
)
from agents.planner import run_planner, generate_initial_plan, refine_plan_to_json, build_planner_task, build_planner_suffix, build_chat_prompt_for_model
from agents.coder import plan_and_code_query
from agents.coder.diff_coder import diff_generate_and_apply
from agents.prompt_cache import dataset_reference_sentence, routed_data_context, task_section
from agents.memory.optimization_experience import build_optimization_experience_for_agent
from engine.expansion_profile import ExpansionProfile
from agents.prompt_policy import (
    autonomous_method_selection_guidance,
    dynamic_expansion_instruction,
    ensure_expansion_profile,
    scoped_search_memory,
)

logger = logging.getLogger("MLEvolve")


def run(
    agent,
    parent_node: SearchNode,
    expansion_profile: ExpansionProfile | None = None,
) -> SearchNode:
    expansion_profile = ensure_expansion_profile(
        agent, parent_node, expansion_profile, "improve"
    )
    data_context = routed_data_context(agent, "merge")
    task_mode = infer_task_mode(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
        autorealize_context=getattr(agent, "data_preview", "") or data_context,
    )
    optimization_or_rl = task_mode in {"optimization", "rl"}
    improvement_standards = (
        "🎯 As a Grandmaster, make MEANINGFUL improvements that boost leaderboard performance.\n\n"
        "**Acceptable**: Advanced architectures, ensemble techniques, feature engineering, hyperparameter optimization, improved pipelines.\n"
        "**NOT Acceptable**: Cosmetic changes, minor tweaks without justification, breaking functionality.\n\n"
    )
    if task_mode == "optimization":
        improvement_standards = (
            "You are an expert optimization/decision solver improving a runnable candidate solution.\n\n"
            "**Acceptable**: better data loading/schema mapping, stricter task-defined validation, better task score, and improvements within the selected solver family.\n"
            "**NOT Acceptable**: throwing away a scorable partial solution without preserving its evaluator, replacing the official score with reward/training loss, or treating no-op/placeholder output as competitive.\n\n"
        )
    elif task_mode == "rl":
        improvement_standards = (
            "You are an RL solution engineer improving a scored policy-based decision candidate.\n\n"
            "**Acceptable**: repair the environment, state/action representation, legal-action mask, reward alignment, policy architecture, training stability, curriculum, rollout, or policy artifact while preserving the shared evaluator.\n"
            "**NOT Acceptable**: replacing the required RL path with a non-RL solver, evaluating an unused RL scaffold, or reporting reward/training loss as the final task score.\n\n"
        )

    introduction = (
        improvement_standards +
        "You are provided with a previously developed solution below and should improve it "
        "in order to further increase the (test time) performance. "
        "For this you should first outline a brief plan in natural language for how the solution can be improved and "
        "then implement this improvement in Python based on the provided previous solution."
    )

    scoped_memory = scoped_search_memory(agent, parent_node, "improve")
    expansion_control = dynamic_expansion_instruction(
        agent, expansion_profile, operator="improve"
    )
    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Memory": f"{scoped_memory}\n\n{expansion_control}",
        "Instructions": {},
    }
    prompt["Previous solution"] = {
        "Code": wrap_code(parent_node.code),
    }
    prompt["Instructions"]["Method selection autonomy"] = autonomous_method_selection_guidance()

    success_patience, total_patience, branch_best_score = get_patience_counter(agent, parent_node)
    use_magnitude_prompt = (success_patience >= 2) or (total_patience >= 5)

    if use_magnitude_prompt:
        trigger_reason = []
        if success_patience >= 2:
            trigger_reason.append(f"success_patience={success_patience}>=2")
        if total_patience >= 5:
            trigger_reason.append(f"total_patience={total_patience}>=5")
        logger.warning(f"🔥 PLATEAU DETECTED! Triggered by: {' AND '.join(trigger_reason)}, using Magnitude-Based prompt")
        if branch_best_score is None:
            best_score_str = "N/A (no successful nodes yet)"
        else:
            best_score_str = f"{branch_best_score:.4f}"
        prompt["Instructions"] |= {
            "🔥 Improvement Strategy: Magnitude-Based Reasoning": [
                "",
                "⚠️ **CRITICAL: The current approach has hit a plateau.**",
                "",
                "Do NOT just tweak parameters unless we are very close to the target.",
                "Classify your thoughts into these 3 Tiers based on the **Magnitude of Change**:",
                "",
                "**Tier 1: Optimization (The \"How\")**",
                "- Definition: Keep the model architecture and data fixed. Only change *how* we train.",
                "- Scope: Hyperparameters, Learning Rate Schedulers, Random Seeds, Post-processing thresholds.",
                "- *When to use: We are fine-tuning a working solution.*",
                "",
                "**Tier 2: Representation & Components (The \"What\")**",
                "- Definition: Change specific modules of the pipeline, but keep the overall paradigm.",
                "- Scope:",
                "    - **Data**: New feature engineering, different augmentations, input normalization.",
                "    - **Model**: Swapping the backbone (e.g., larger model), changing the loss function, adding regularization layers (Dropout/BN).",
                "- *When to use: The current model is underfitting or overfitting.*",
                "",
                "**Tier 3: Systemic Paradigm Shift (The \"Architecture\")**",
                "- Definition: Fundamentally change the approach. The old code structure might need a rewrite.",
                "- Scope:",
                "    - **Paradigm**: Switching from GBDT to Neural Net (or vice versa), Single Model -> Ensemble.",
                "    - **Objective**: Changing from Regression to Classification (binning), Multi-task learning.",
                "    - **Data Flow**: Pseudo-labeling, Self-supervised pre-training.",
                "- *When to use: The current approach has hit a hard ceiling (plateau).*",
                "",
                "**Current Status**:",
                f"- Best Score: {best_score_str}",
                f"- Successful nodes without improvement: {success_patience}",
                f"- Total nodes since best: {total_patience} (including failed attempts)",
                "",
                "**⚠️ CRITICAL INSTRUCTION**:",
                f"The branch has stagnated (success_patience={success_patience}, total_patience={total_patience}).",
                "You MUST propose a **Tier 2 or Tier 3** change to break the plateau.",
                "Do NOT propose Tier 1 (hyperparameter tuning). The current approach needs a more fundamental change.",
                "",
                "You can refer to the expert technique suggestions above, which are distilled from the kaggle award-winning solutions.",
                "",
                "After deciding your Tier 2/3 strategy, briefly describe:",
                "- Which Tier you're using and why",
                "- What specific components will change",
                "- Why this addresses the root cause of the plateau",
            ],
        }
    else:
        prompt["Instructions"] |= {
            "🔬 Critical: Scientific Approach to Optimization": [
                "Choose one specific change supported by the parent node's metric, result review, execution output, or memory evidence.",
                "Explain internally why that change addresses the observed bottleneck and preserve the evaluator, output contract, data split, and unrelated working interfaces.",
                "Keep the visible plan to 1-3 information-dense sentences and follow the active JSON, diff, or plan-plus-code response contract; do not add a separate plan template.",
            ],
        }

    prompt["Instructions"] |= {
        "Solution improvement guidelines": [
            "- Propose a single, specific, actionable improvement (atomic change for controlled experiment).\n",
            "- Your improvement must be distinctly different from existing attempts in the Memory section.\n",
            "",
            "⚠️ **IMPORTANT: Depth of Improvement**",
            "Consider TWO types of improvements (both are valid, but think about which is more appropriate):",
            "",
            "**Type A: Architectural Deepening (Often More Powerful)**",
            "- ADD components to existing model (e.g., add relevant mechanisms for this task)",
            "- MODIFY internal structure (e.g., enhance feature extraction for task characteristics)",
            "- DESIGN task-specific modules based on domain knowledge and data patterns",
            "Example: 'Keep current backbone, but ADD [mechanism] to address [specific task challenge]'",
            "",
            "**Type B: Model/Method Replacement (Simpler but may miss potential)**",
            "- REPLACE entire model/algorithm with a different approach",
            "- This is valid when current architecture is fundamentally unsuitable for this task",
            "- But ask yourself: Could I improve the current approach by adding/modifying instead of replacing?",
            "",
            "- Keep the visible plan to 1-3 information-dense sentences covering the specific change, evidence from the parent node, and the expected effect. Do detailed reasoning internally.\n",
            "- Don't suggest to do EDA.\n",
        ],
    }
    if optimization_or_rl:
        prompt["Instructions"] |= {
            "Decision optimization improvement guidelines": [
                "- Build on any partial solution with a usable scalar score. Preserve `load_problem_data`, `validate_solution`, `score_solution`, optional diagnostics, and the exact output schema unless they are the defect being fixed.",
                "- Improvement priority is task-specific: fix schema/read errors first, then improve the official scalar score by targeting the objective components and validator evidence reported by the parent node.",
                "- Use the validation summary counts/examples to target the actual bottleneck. Do not assume every optimization task uses the same progress signals.",
                "- Improve feasible candidate generation before model tuning: construct task-defined constraint masks and select only from legal actions. If the mask is empty, add a documented task-approved fallback, repair step, or backtracking/new-resource branch and report examples.",
                "- Keep or add task-relevant diagnostics when useful. The result reviewer checks the actual returned solution and task-aligned score; no specific diagnostic field and no printed scalar automatically accepts a node.",
                (
                    "- RL is the required method for this task. Improve the existing environment-policy-training-rollout chain and ensure the evaluated solution still comes from the saved policy artifact. Do not switch to a non-RL solver."
                    if task_mode == "rl"
                    else "- Preserve the selected optimization method family for a controlled improvement unless parent evidence and an applicable experience justify a hybrid extension that reuses the incumbent and evaluator, such as heuristic warm start plus exact neighborhood optimization. An unrelated replacement belongs in a separate draft branch."
                ),
                "- If the current approach is empty/no-op/placeholder, first repair feasible candidate generation and data mapping; do not spend the next node only tuning model hyperparameters.",
            ],
        }
    if task_mode == "optimization":
        optimization_experience = build_optimization_experience_for_agent(
            agent,
            task_mode=task_mode,
            extra_context="\n".join(
                [
                    str(parent_node.plan or ""),
                    str(parent_node.analysis_for_prompt or ""),
                    str(parent_node.code_summary or ""),
                ]
            ),
        )
        prompt["Instructions"]["Conditional optimization experience for improvement"] = [
            "- Diagnose whether the current bottleneck is candidate quality, formulation weakness, loose bounds, model size, neighborhood size, solver status, or evaluator/output correctness before changing code.",
            "- A scorable heuristic incumbent may be retained as a warm start or upper bound while an exact/relaxed/local-exact component improves it; do not discard a working solution merely to imitate an experience card.",
            "- If solver bound/gap or formulation-size facts are available in Decision/Optimization Solver Summary, use those facts in the plan. Do not invent missing bounds or claim optimality from an incumbent alone.",
            optimization_experience,
        ]

    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    prompt["Instructions"] |= prompt_leakage_prevention()
    internet_clarification = get_internet_clarification(getattr(agent.cfg, "pretrain_model_dir", ""))
    prompt["Instructions"]["Implementation guideline"].extend(internet_clarification)
    if task_mode == "prediction":
        prompt["Instructions"] |= ROBUSTNESS_GENERALIZATION_STRATEGY

    output = wrap_code(parent_node.term_out, lang="")

    if not agent.acfg.use_diff_mode:
        prompt["Instructions"] |= prompt_resp_fmt()

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    memory_section = ""
    if prompt.get("Memory", "").strip():
        memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"

    user_prompt = f"{task_section(prompt['Task description'], data_context)}\n{instructions}{memory_section}"
    assistant_prefix = (
        "Let me approach this systematically.\n"
        f"{dataset_reference_sentence(prompt['Task description'], data_context)}\n"
        f"The current solution uses the following code:\n{prompt['Previous solution']['Code']}\n"
        f"Its output was:\n{output}\n"
        "Building on this, I'll develop an improved approach."
    )
    prompt_complete = build_chat_prompt_for_model(agent.acfg.code.model, introduction, user_prompt, assistant_prefix)

    parent_node.add_expected_child_count()

    if agent.acfg.use_diff_mode:
        try:
            logger.info(f"Using diff improve for node {parent_node.id}")
            plan, code = _diff_improve(agent, prompt, data_context, parent_node)
        except Exception as e:
            logger.warning(f"Diff improve failed: {e}, falling back to full rewrite")
            plan, code = plan_and_code_query(agent, prompt_complete)
    else:
        plan, code = plan_and_code_query(agent, prompt_complete)

    from_topk = getattr(parent_node, '_topk_triggered', False)

    new_node = SearchNode(plan=plan, code=code, parent=parent_node, stage="improve",
                        local_best_node=parent_node.local_best_node, from_topk=from_topk)
    register_node(agent, new_node, prompt_complete, parent_node=parent_node)

    if hasattr(parent_node, '_topk_triggered'):
        parent_node._topk_triggered = False

    logger.info(f"[improve] {parent_node.id} → node {new_node.id}")
    return new_node


# ============ Diff improve pipeline ============

_IMPROVE_STAGE_INTRO = (
    "Based on the task requirements, data characteristics, and execution results, carefully analyze "
    "the current solution to identify improvement opportunities that will enhance the final test set "
    "performance. Then select which component(s) to modify and provide detailed, actionable modification plans."
)
_IMPROVE_EXTRA_GUIDELINE = (
    "1. **Analyze the task description and data type carefully** before proposing enhancements. "
    "Your improvements must be based on the current task."
)
_IMPROVE_PLANNER_TASK = build_planner_task(_IMPROVE_STAGE_INTRO, _IMPROVE_EXTRA_GUIDELINE)

_IMPROVE_DIFF_INTRODUCTION = (
    "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
    "solution and a detailed improvement plan. Your task is to implement the improvement plan to enhance "
    "the solution's test set performance."
)


_IMPROVE_SUFFIX_EXTRA = (
    "Building on the current solution, I'll develop an improved approach "
    "that addresses identified limitations while preserving what works well."
)


def _diff_improve(agent, prompt_base, data_preview, parent_node):
    context = {
        "stage": "improve",
        "memory": prompt_base["Memory"],
        "previous_code": parent_node.code,
        "previous_code_summary": parent_node.code_summary if hasattr(parent_node, 'code_summary') and parent_node.code_summary else None,
        "execution_output": parent_node.term_out if hasattr(parent_node, 'term_out') else "",
        "parent_node": parent_node,
    }

    use_memory = (
        getattr(agent.acfg, 'use_global_memory', False)
        and agent.global_memory is not None
        and len(agent.global_memory.records) > 0
    )

    if use_memory:
        logger.info("[DiffImprove] Using two-stage planning with memory")
        initial_plan = generate_initial_plan(agent, prompt_base, data_preview, context)
        planning_result = refine_plan_to_json(agent, initial_plan, prompt_base, data_preview, context)
    else:
        logger.info("[DiffImprove] Using direct planner (memory disabled or empty)")
        planning_result = run_planner(
            agent_instance=agent,
            prompt_base=prompt_base,
            data_preview=data_preview,
            context=context,
            your_task_section=_IMPROVE_PLANNER_TASK,
            assistant_suffix=build_planner_suffix(prompt_base, data_preview, context, extra_text=_IMPROVE_SUFFIX_EXTRA),
            stage_name="ImprovePlanning",
        )

    modules = planning_result.get('module', [])
    plans = planning_result.get('plan', {})

    if not planning_result.get("parse_success", False):
        raise RuntimeError("Planner returned empty result after retries, triggering outer fallback")

    if not modules and plans:
        planning_result['module'] = list(plans.keys())

    return diff_generate_and_apply(
        agent_instance=agent,
        planning_result=planning_result,
        parent_code=parent_node.code,
        data_preview=data_preview,
        execution_output=context["execution_output"],
        introduction=_IMPROVE_DIFF_INTRODUCTION,
    )
