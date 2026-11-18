"""Implementation requirements scoped to the task and current execution budget."""

import time
import humanize

from .shared import (
    FORMAL_SEARCH_EVALUATION_GUIDANCE,
    SEARCH_EVALUATION_GUIDANCE,
    get_decision_solution_protocol,
    get_optimization_rl_strategy,
    infer_task_mode,
)


def get_impl_guideline_from_agent(agent, *, fast_draft: bool = False):
    """Build implementation guideline from agent config."""
    tot_time_remaining = agent.acfg.time_limit - (time.time() - agent.start_time)
    exec_timeout = int(min(agent.cfg.exec.timeout, tot_time_remaining))
    data_context = getattr(agent, "data_preview", "") or getattr(agent, "autorealize_context", "")
    formal_autorealize = bool(
        getattr(
            agent,
            "has_autorealize_context",
            "AutoRealize Structured Context" in str(data_context),
        )
    )
    task_mode = infer_task_mode(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
        autorealize_context=data_context,
    )
    optimization_rl = task_mode in {"optimization", "decision", "rl"}
    guideline = get_impl_guideline(
        tot_time_remaining=tot_time_remaining,
        steps_remaining=agent.acfg.steps - agent.current_step,
        exec_timeout=exec_timeout,
        expose_prediction=getattr(agent.acfg, "expose_prediction", False),
        k_fold_validation=getattr(agent.acfg, "k_fold_validation", 0),
        pretrain_model_dir=getattr(agent.cfg, "pretrain_model_dir", ""),
        generate_submission=getattr(agent.acfg, "generate_submission", True),
        optimization_rl=optimization_rl,
        formal_autorealize=formal_autorealize,
        auto_install_missing_dependencies=bool(
            getattr(
                getattr(agent.cfg, "exec", None),
                "auto_install_missing_dependencies",
                False,
            )
        ),
    )
    if fast_draft and task_mode == "prediction" and not formal_autorealize:
        guideline["Implementation guideline"] = _fast_prediction_draft_guideline(
            generate_submission=bool(getattr(agent.acfg, "generate_submission", True)),
            expose_prediction=bool(getattr(agent.acfg, "expose_prediction", False)),
        )
    runtime_budget = guideline.pop("Current execution budget")
    guideline |= get_decision_solution_protocol(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
        enabled=optimization_rl,
    )
    guideline |= get_optimization_rl_strategy(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
        enabled=optimization_rl,
    )
    guideline["Current execution budget"] = runtime_budget
    return guideline


def _fast_prediction_draft_guideline(
    *,
    generate_submission: bool,
    expose_prediction: bool,
) -> list[str]:
    """Minimal first-draft requirements for simple predictions."""
    lines = [
        "Follow the task and data contract exactly. Use only supplied files under ./input.",
        "Use the task's frozen split and metric. Split before fitting; fit only on training rows.",
        "Implement one coherent candidate suited to the data and budget.",
        "Define train(data, artifact_dir) and predict(model_path, data); save artifacts under ./working/artifact/.",
        "Print Final Validation Score from actual validation predictions.",
        "Keep standard output limited to the required score line; put evidence under ./working/evidence/.",
    ]
    if generate_submission:
        lines.append("Write submission only when the task specifies its path and schema.")
    if expose_prediction:
        lines.append("Keep predict() importable for downstream use without retraining.")
    return lines


def _format_time(time_in_sec):
    """Format seconds for display."""
    return f"{int(time_in_sec) // 3600}h {(int(time_in_sec) % 3600) // 60}m {int(time_in_sec) % 60}s"


def get_impl_guideline(
    tot_time_remaining: float,
    steps_remaining: int,
    exec_timeout: int,
    expose_prediction: bool = False,
    k_fold_validation: int = 0,
    pretrain_model_dir: str = "",
    generate_submission: bool = True,
    optimization_rl: bool = False,
    formal_autorealize: bool = False,
    auto_install_missing_dependencies: bool = False,
) -> dict:
    """Build implementation guideline from time and config."""

    # Core inference requirements
    if optimization_rl:
        inference_core = [
            "**Decision/Solution Generation**",
            "- Every decision/assignment/route/action must come from a real solver, policy, or heuristic.",
            "- Required: load_problem_data(), build_solution(), validate_solution(), score_solution(), run_evaluator_self_tests().",
            "- Use exact physical column names from AutoRealize schema. Resolve business aliases explicitly before pandas operations.",
            "- Define evaluation population before scoring. Handle missing fields per task contract.",
            "- validate_solution checks feasibility, completeness, schema, IDs, constraints.",
            "- score_solution returns one finite scalar from the task contract. No reward-only, training loss, or proxy metrics.",
            "- Empty/placeholder/constant solutions are invalid unless task authorizes them.",
            "- Output columns describe generated results, not input columns. Use OUTPUT_COLUMNS = [...] with pd.DataFrame(rows, columns=OUTPUT_COLUMNS).",
            "",
        ]
        metric_output = [
            "**Validation Output**",
            "- Optional: print one JSON line Decision Validation Summary with task-relevant diagnostics.",
            "- MUST print: print(f'Final Validation Score: {score}')",
            "- Score from score_solution() after validate_solution() and official invalid-solution handling.",
            "- If submission enabled, save schema-correct table even with zero rows.",
            "",
        ]
        interface_spec = [
            "- Non-RL: define solve(model_path, data). Stateless solvers may use model_path=None.",
            "- RL/hybrid: define train_policy(data, artifact_dir) and rollout(model_path, data).",
        ]
        self_check = [
            "Generated real solution (not placeholder)?",
            "validate_solution checks feasibility?",
            "score_solution computes exactly one official scalar?",
            "Output columns declared explicitly?",
        ]
    else:
        inference_core = [
            "**Model Inference**",
            "- EVERY prediction must come from trained model's forward pass: Load data → Preprocess → model.predict() → Save.",
            "- FORBIDDEN: constants, placeholders, statistics, random numbers, fake metric functions.",
            "- For large temporal tables, use bounded windows, pre-aggregation, or rolling ops. Avoid full history rescan per origin.",
            "- Use exact physical fields from AutoRealize schema. Report gaps instead of guessing column names.",
            "",
        ]
        metric_output = [
            "**Validation Metric**",
            "- MUST print: print(f'Final Validation Score: {score}')",
            "- Compute on engine-frozen development population only. Final holdout is in engine-owned final_evaluate().",
            "- CRITICAL: validation and test must use IDENTICAL processing logic.",
            "",
        ]
        interface_spec = [
            "- MUST define train(data, artifact_dir) to fit and save artifacts.",
            "- MUST define predict(model_path, data) that loads artifacts and returns predictions without retraining.",
        ]
        self_check = ["Predictions pass through learned weights? (If NO → INVALID)"]

    # Submission requirements
    submission_section = (
        [
            "**Submission**",
            "- Path: ./submission/submission.csv with all required predictions/decisions.",
            "- Format: follow task description exactly.",
            "",
        ]
        if generate_submission
        else [
            "**Submission generation disabled**",
            "- Do not force ./submission/submission.csv. Focus on train/validation pipeline and reusable inference.",
            "",
        ]
    )

    # Artifact requirements
    artifact_core = (
        "Save policy, fitted models, preprocessing, solver config under ./working/artifact/. Stateless solvers may use solve(None, data)."
        if optimization_rl
        else "MUST save trained model and all preprocessing state under ./working/artifact/."
    )

    impl_guideline = [
        "**CRITICAL REQUIREMENTS**:",
        "",
        "**AutoRealize Contract Priority**",
        "- If ./input/autorealize_context.md exists, read and obey it first.",
        "- If Execution Contract present, preserve contract_id and evaluation_id. Treat blocking issues as unresolved constraints.",
        "- Data reading examples, output contract, evaluation contract, leakage guards override generic templates.",
        "- Prefer AutoRealize Data Read Recipes; verify shape/columns after loading and report mismatches.",
        "- Do not invent submission columns, target fields, seeds, matrices, or formulas not provided by AutoRealize.",
        "- Source Alias Guard: listed aliases unsafe for raw pandas access without exact_physical_column.",
        "",
        *inference_core,
        *submission_section,
        "**Reusable Artifacts**",
        f"- {artifact_core}",
        "- Use standard names: model.pkl, best_model.pt, best_model.joblib inside ./working/artifact/.",
        "- Keep dev-only data and row-level evidence under ./working/evidence/.",
        "- Artifacts must support later inference without retraining: include preprocessing, metadata, weights, config.",
        "- For PyTorch: save state_dict plus config. For sklearn/boosting: save pipeline or model+preprocessing.",
        "- Save metadata: feature order/types, package versions, seeds, evaluation protocol, relative paths.",
        "- Persist selected hyperparameters and preprocessing choices in a recipe listed in the artifact manifest, relative to the solution.",
        "- train(data, a_new_artifact_dir) must reproduce that selected recipe in a fresh directory. Load it relative to the solution or use an identical embedded fallback; never silently fall back to untuned defaults or a previous working directory.",
        "- Reload artifact and verify inference on small batch before reporting success.",
        "",
        "**Inference API**",
        *interface_spec,
        "",
        "**Comparable Evaluation**",
        SEARCH_EVALUATION_GUIDANCE,
        *([FORMAL_SEARCH_EVALUATION_GUIDANCE] if formal_autorealize else []),
        "- Freeze metric formula, population, split, horizon, seeds, feasibility rules, output schema across all candidates.",
        "- For forecasts: chronological holdouts or rolling origins matching horizon. Respect entity grouping and label delay.",
        "- For repeated entities: honor contract grouping so related records stay in same split.",
        "- Fit preprocessing, scaling, encoders, transforms on training data only.",
        "- When task requires protocol diagnostics, persist Evaluation Protocol Summary under ./working/evidence/.",
        "",
        *metric_output,
        f"**Directories**: input in ./input/, {'submission in ./submission/, ' if generate_submission else ''}temp in ./working/",
        "",
        f"**Packages**: common data-science packages normally available.{f' Offline models at {pretrain_model_dir}.' if pretrain_model_dir else ''}",
        "",
        "**Dependency Management**:",
        *(
            [
                "- For missing packages: # ALGOEVOLVE_PIP_INSTALL[<import>]: pip install <distribution>",
                "- Examples: # ALGOEVOLVE_PIP_INSTALL[sklearn]: pip install scikit-learn",
                "- Never call pip/conda/subprocess from code.",
            ]
            if auto_install_missing_dependencies
            else ["- Auto-install disabled. Use only available packages; never execute pip/conda from code."]
        ),
        "- Avoid tqdm and verbose output. Print at most one line per epoch. Choose DataLoader workers within CPU budget.",
        "",
        "**Self-Check**:",
        *[f"- {item}" for item in self_check],
        f"- Saved all required artifacts{' or documented stateless solve(None, data)' if optimization_rl else ''}?",
        "- Exposed reusable interface and verified reload?",
        *([f"- Generated submission at correct path?" if generate_submission else "- Avoided forcing submission (disabled)?"]),
        "- Printed validation metric as last line?",
        "- Used complete training dataset (not tiny subset)?",
    ]

    if expose_prediction:
        impl_guideline.append(
            "- Documented predict/solve/rollout input type for downstream use."
        )
    if k_fold_validation > 1:
        impl_guideline.append(
            f"- Used {k_fold_validation}-fold CV when appropriate for the task."
        )

    return {
        "Implementation guideline": impl_guideline,
        "Current execution budget": [
            f"**Budget**: Time ≤ {_format_time(max(0, tot_time_remaining))} | Steps = {max(0, steps_remaining)} | Max exec = {humanize.naturaldelta(max(0, exec_timeout))}",
        ],
    }
