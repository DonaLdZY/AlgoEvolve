"""Draft Agent: initial plan and code draft."""

import logging
from pathlib import Path
from typing import Any, Optional

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from agents.coder import plan_and_code_query, stepwise_plan_and_code_query
from agents.triggers import register_node
from agents.prompts import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    prompt_leakage_prevention,
    prompt_resp_fmt,
    get_prompt_environment,
    get_impl_guideline_from_agent,
    infer_task_mode,
)
from agents.planner import build_chat_prompt_for_model
from agents.prompt_cache import dataset_reference_sentence, task_section
from agents.memory.optimization_experience import build_optimization_experience_for_agent
from engine.expansion_profile import ExpansionProfile
from agents.prompt_policy import (
    autonomous_method_selection_guidance,
    dynamic_expansion_instruction,
    ensure_expansion_profile,
    scoped_search_memory,
)

logger = logging.getLogger("AlgoEvolve")


def run(
    agent,
    init_solution_path: Optional[str] = None,
    expansion_profile: ExpansionProfile | None = None,
    *,
    fast_draft_mode: bool | None = None,
    use_stepwise_generation: bool | None = None,
) -> SearchNode:
    fast_draft_mode = (
        bool(getattr(agent, "fast_draft_mode", False))
        if fast_draft_mode is None
        else bool(fast_draft_mode)
    )
    use_stepwise_generation = (
        bool(getattr(agent, "use_stepwise_generation", True))
        if use_stepwise_generation is None
        else bool(use_stepwise_generation)
    )
    expansion_profile = ensure_expansion_profile(
        agent, agent.virtual_root, expansion_profile, "draft"
    )
    prompt_data_context = agent.data_preview
    """Generate initial draft. If init_solution_path is provided and readable, use file content directly."""
    if init_solution_path:
        try:
            code = Path(init_solution_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read init_solution from {init_solution_path}: {e}, falling back to LLM generation")
            init_solution_path = None
        else:
            plan = "User-provided init solution."
            agent.virtual_root.add_expected_child_count()
            new_node = SearchNode(
                plan=plan,
                code=code,
                parent=agent.virtual_root,
                stage="draft",
                local_best_node=agent.virtual_root,
            )
            register_node(agent, new_node, "User-provided init solution (no LLM).", new_branch=True)
            logger.info(f"[draft] → node {new_node.id} (branch={new_node.branch_id}) [init_solution]")
            return new_node

    professional_identity = (
        "🏆 You are a Kaggle Grandmaster - a top-tier ML expert competing to WIN.\n\n"
        "**Your Standards**:\n"
        "✓ Design complete ML pipelines (data → model → training → inference)\n"
        "✓ Implement real models that LEARN from data (not baseline scripts with constants)\n"
        "✓ Generate predictions through ACTUAL MODEL INFERENCE on each sample\n"
        "✓ Compete for TOP performance, not trivial baselines\n\n"
        "Your solution will be evaluated on a real leaderboard. Treat this with professionalism.\n\n"
    )

    task_mode = infer_task_mode(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
        autorealize_context=getattr(agent, "data_preview", ""),
    )
    if task_mode in {"optimization", "rl"}:
        professional_identity = (
            "You are an expert competition solver for optimization, reinforcement learning, and decision problems.\n\n"
            "**Your Standards**:\n"
            "- Freeze the shared evaluator, output schema, and task-defined constraints before optimizing.\n"
            "- Build a real feasible solution/action plan, not placeholder predictions.\n"
            "- Validate the generated decision artifact and report the one task-aligned scalar score.\n"
            "- Keep each search node to one coherent method; comparison between methods belongs to the search tree.\n\n"
            "Your solution will be evaluated on a real leaderboard. Treat this with professionalism.\n\n"
        )

    introduction = (
        professional_identity +
        "Now, let's begin the competition. "
        "You need to come up with an excellent and creative plan for a competitive solution "
        "and then implement this solution in Python with the quality expected of a Kaggle Grandmaster. "
        "We will now provide a description of the task."
    )
    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Memory": scoped_search_memory(agent, agent.virtual_root, "draft"),
        "Instructions": {},
    }
    prompt["Instructions"] |= prompt_resp_fmt()
    prompt["Instructions"]["Method selection autonomy"] = autonomous_method_selection_guidance()

    prompt["Instructions"] |= {
        "🔬 Critical: Scientific Approach to Design": [
            "",
            "Before designing your solution, you must answer three fundamental questions:",
            "",
            "1. **WHAT makes this task unique?**",
            "   - Not generic observations like 'it's a classification task'",
            "   - What SPECIFIC patterns, challenges, or domain characteristics?",

            "",
            "2. **WHY is your approach suitable for this task?**",
            "   - Not just 'this model is good' - explain the MATCH between approach and task",
            "   - What properties of your method address the task characteristics?",

            "",
            "3. **HOW will you validate your hypothesis?**",
            "   - What outcome would confirm your approach is right?",
            "   - What outcome would suggest you need to reconsider?",

            "",
            "---",
            "",
            "⚠️ This is not a template to fill - this is how scientists think.",
            "Blindly applying standard methods without understanding WHY is not acceptable.",
            "",
            "Your plan should naturally reflect this reasoning process.",
        ],
    }

    prompt["Instructions"] |= {
        "Solution sketch guideline": [
            "- Follow the active simple/normal/complex profile at the end of this prompt. The profile controls implementation cost and risk, not the method family.\n",
            "- 🎯 **CRITICAL: NOVELTY & DIVERSITY REQUIREMENT**:\n",
            "  • **Mandatory**: Your solution MUST be NOVEL compared to ALL existing attempts in Memory.\n",
            "  • **Step 1**: Carefully analyze the core idea of EACH previous attempt in Memory.\n",
            "  • **Step 2 - Choose Strategy**:\n",
            "    → **Option A (Preferred)**: Propose a COMPLETELY DIFFERENT approach exploring an untried direction.\n",
            "    → **Option B**: Build upon an existing approach BUT add significant novel insights that fundamentally change the solution.\n",
            "  • **Forbidden**: Minor variations (changing hyperparameters, swapping similar models, tweaking preprocessing).\n",
            "  • **Think**: 'Does my approach explore a fundamentally different hypothesis?' If NO → redesign.\n",
            "- Don't propose the same modelling solution but keep the evaluation the same.\n",
            "- Keep the visible plan to 1-3 information-dense sentences covering WHAT changes, WHY they fit this task, and HOW validation will confirm them. Do the detailed reasoning internally.\n",
            "- Use the task description / AutoRealize context evaluation metric exactly. Do not invent a separate metric; if the metric is incomplete, implement the most conservative supported scalar and make the assumption explicit.\n",
            "- Don't suggest to do EDA.\n",
            "- The data is already prepared in `./input` directory. No need to unzip files.\n",
            "- If AutoRealize context contains an Exact Source Schema Contract, use its exact `sheet_name` and `physical_columns_exact` strings for all pandas reads. Business concepts or English variable names are code-local derived variables, not raw dataframe column names.\n",
            "- When a description concept is not an exact physical column, resolve it against actual columns before use and keep the mapping near the load code. Never call `groupby`, `agg`, joins, or filters on a column name that is absent from `df.columns`.\n",
            "- Before `groupby`, `agg`, merge, filter, or sort, bind each business concept to a resolved exact source column variable. Create semantic aliases only after exact source access, not as guessed raw column names.\n",
            "- For decision/optimization tasks, first define the evaluation population. Do not silently drop rows/orders because one date/time/feature field is missing; if the task contract defines an evaluable subset or exclusion rule, apply it explicitly and report excluded counts/examples, otherwise use a documented fallback field, an UNKNOWN/default bucket, or validation details.\n",
        ],
        "Coding & Execution Guidelines (CRITICAL)": [
            "- **NO PROGRESS BARS**: You MUST NOT use `tqdm`. Assume `tqdm` is not installed. Use standard Python loops only. Do not use `verbose=1`.",
            "- **MINIMAL LOGGING**: Print ONLY 1 line per epoch (e.g. loss/accuracy). Do NOT print batch-level logs.",
            "- **FINAL OUTPUT**: The VERY LAST line of execution MUST be `print(f'Final Validation Score: {score}')` so the result reviewer can identify the candidate-reported task score."
        ]
    }
    if task_mode in {"optimization", "rl"}:
        method_requirement = (
            "- REQUIRED METHOD: this draft must implement an actual RL solution path. Construct or use a decision environment, "
            "train or configure a policy, use that policy for the evaluated rollout, save its artifact, and expose `train_policy(data, artifact_dir)` plus `rollout(model_path, data)` without retraining.\n"
            if task_mode == "rl"
            else "- METHOD SCOPE: implement one coherent optimization method in this node; do not add an unrelated competing method inside the same draft.\n"
        )
        prompt["Instructions"]["Solution sketch guideline"].extend(
            [
                method_requirement,
                "- Build `load_problem_data`, `validate_solution`, and `score_solution` once and use that same evaluator for method selection, training reward alignment, rollout validation, and the final score.\n",
                "- Candidate generation must mask illegal actions before scoring/selection whenever constraints are known. Use the task's exact entities, feasibility rules, resource availability, capacity, time/budget limits, uniqueness rules, and other hard constraints before choosing any action.\n",
                "- If no legal action exists for an item/job/state, handle it as a first-class branch according to the task contract before logits/softmax: mark it infeasible/undecided when allowed, try an allowed repair/new-resource/backtracking fallback, and include task-defined counts/examples in `Decision Validation Summary`. Never softmax an all-invalid mask into NaN probabilities or pick an illegal action just to avoid an empty candidate set.\n",
                "- If this node chooses RL/hybrid, explicitly assess curriculum learning. When suitable, progress from small instances, relaxed constraints, short horizons, or heuristic demonstrations toward the full task using success/feasibility/evaluator thresholds; otherwise state why it is unsuitable. Final selection must replay the original full-scale environment, hard constraints, complete horizon, and official evaluator.\n",
                "- A partial, infeasible, empty, or diagnostic solution may still be useful if it is scored by the official deterministic evaluator/formula and the limitations are explicit.\n",
                "- If you print `Decision Validation Summary`, keep it task-appropriate: objective components, validator status, and short examples for the dominant failure reason. It is diagnostic evidence for the result reviewer, not an automatic acceptance signal.\n",
                "- The draft must finish the selected method end to end: data contract, evaluator, solver or policy, reusable artifact, the documented `solve` or `train_policy`/`rollout` interface, solution generation, and final evaluation.\n",
            ]
        )
    if task_mode == "optimization":
        optimization_experience = build_optimization_experience_for_agent(
            agent,
            task_mode=task_mode,
            extra_context=str(prompt.get("Memory", "") or ""),
        )
        experience_guidance = [
            "- For optimization, 'relatively simple first solution' means one coherent executable method, not necessarily a greedy heuristic. A compact exact formulation is valid when the structure assessment shows it fits the configured time and memory budget.",
            "- Before choosing the method, assess whether the task has bounded aggregate integer states, local/adjacent nonlinear costs, useful relaxations, computable lower/upper bounds, or a feasible incumbent that can seed exact or large-neighborhood optimization.",
            "- Use a retrieved experience only when its applicability checks match this task. It is a reusable hypothesis, not a requirement to use MILP, CP-SAT, a commercial solver, or any particular algorithm.",
            "- Let the search tree provide method diversity: if Memory already contains a heuristic branch, consider an untried exact or hybrid branch when applicable, and vice versa. Keep this node itself to one coherent method trajectory.",
            "- Preserve any fast feasible solution as an incumbent/upper bound when useful, and independently replay `validate_solution` plus `score_solution` after every solver path.",
        ]
        # Stepwise mode injects the full card only into solver_design; avoid
        # repeating the same stable context in every specialized step.
        if not use_stepwise_generation and not fast_draft_mode:
            experience_guidance.append(optimization_experience)
        prompt["Instructions"]["Optimization structure and experience guidance"] = experience_guidance
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    prompt["Instructions"] |= prompt_leakage_prevention()

    if fast_draft_mode and task_mode in {"optimization", "rl"}:
        coldstart_guideline = [
            "Fast draft: use the one method family most directly supported by the task contract and installed runtime. "
            "Do not enumerate unrelated ML/RL/optimization families inside this node."
        ]
    elif agent.use_coldstart and (agent.coldstart_description != "None model"):
        if "Reference pattern" in str(agent.coldstart_description):
            coldstart_guideline = [
                f"""
            **Cold-start Method Strategy**:

            **Recommended references**: {agent.coldstart_description}

            These are method-level references, not fixed pretrained-model snippets.
            - Adapt the reference pattern to the current task; do NOT copy it blindly.
            - Follow the AutoRealize required-method contract. A static optimization task may require RL constructed from static problem instances.
            - If RL is required, implement the complete RL path in this node rather than replacing it with another solver family.
            - If a `Reference pattern` conflicts with the task description, the task description and metric take priority.
            """
            ]
        else:
            coldstart_guideline = [
                f"""
            **Pretrained Model Strategy**:

            **Option A [RECOMMENDED]**: {agent.coldstart_description}
              Strong pretrained models with proven transfer performance. Use for end-to-end fine-tuning OR as frozen feature extractors.

            **Option B**: Alternative pretrained models if better suited to task characteristics.

            **Option C**: Train from scratch / non-DL methods only when pretraining provides no advantage.

            **CRITICAL: When using any recommended pretrained model (Option A), you MUST copy the Code template EXACTLY as provided, including model variant names, file paths, and checkpoint filenames. Only the listed weights are available locally; other variants may fail to load.**

            **Key Techniques**:
            1. **Feature Extractor Pattern**: If dataset is small or domain mismatch exists, freeze backbone + train only final layers or feed extracted features to another model.
            2. **Mixed Precision**: Use `torch.cuda.amp` or `torch.amp` autocast/GradScaler where appropriate. Do NOT manually convert pretrained models to `.half()` unless the code is designed for it.
            3. **Avoid Timeouts**: Use DataLoader with num_workers>=2 and cache extracted features for large datasets/heavy backbones.
            """
            ]
    else:
        coldstart_guideline = [""]


    prompt["Instructions"]["Implementation guideline"].extend(coldstart_guideline)
    prompt["Instructions"] |= get_prompt_environment()
    if task_mode == "prediction":
        prompt["Instructions"] |= ROBUSTNESS_GENERALIZATION_STRATEGY

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    memory_section = ""
    if prompt.get("Memory", "").strip():
        memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"

    expansion_control = dynamic_expansion_instruction(
        agent, expansion_profile, operator="draft"
    )
    user_prompt = (
        f"{task_section(prompt['Task description'], prompt_data_context)}\n"
        f"{instructions}{memory_section}\n{expansion_control}"
    )
    assistant_prefix = (
        "Let me approach this systematically.\n"
        f"{dataset_reference_sentence(prompt['Task description'], prompt_data_context)}"
    )
    prompt_complete = build_chat_prompt_for_model(
        agent.acfg.code.model, introduction, user_prompt, assistant_prefix
    )
    agent.virtual_root.add_expected_child_count()

    if use_stepwise_generation and expansion_profile.complexity != "simple":
        plan, code = stepwise_plan_and_code_query(
            agent_instance=agent,
            prompt_base=prompt,
            data_preview=agent.data_preview,
            context={
                "stage": "draft",
                "memory": prompt.get("Memory", ""),
                "expansion_control": expansion_control,
            },
        )
    else:
        plan, code = plan_and_code_query(agent, prompt_complete)
    new_node = SearchNode(plan=plan, code=code, parent=agent.virtual_root, stage="draft",
                        local_best_node=agent.virtual_root)
    register_node(agent, new_node, prompt_complete, new_branch=True)

    logger.info(f"[draft] → node {new_node.id} (branch={new_node.branch_id})")
    return new_node
