#!/usr/bin/env python3
"""Prompt templates for code review in the search pipeline."""

from typing import Any, Dict

from utils.response import wrap_code


def get_code_review_prompt(
    task_desc: str,
    code: str,
    submission_required: bool = True,
) -> Dict[str, Any]:
    """Build the complete code-review prompt."""
    introduction = (
        "You are a Senior Data Science Code Reviewer. Your goal is to ensure the solution is "
        "legally valid and logically sound.\n\n"
        "CRITICAL INSTRUCTION:\n"
        "You must strictly follow the Code Review Guidelines below. Do not rely on general "
        "knowledge when it conflicts with the listed environment facts. Focus only on data "
        "leakage and critical integrity. Do not replace the user's model architecture with "
        "another backbone merely to make code executable.\n"
    )
    return {
        "Introduction": introduction,
        "Task description": task_desc,
        "Code to review": wrap_code(code),
        "Instructions": {
            "Code review guidelines": get_code_review_guidelines(
                submission_required=submission_required
            ),
            "Response format": get_code_review_response_format(),
        },
    }


def get_code_review_guidelines(submission_required: bool = True) -> list[str]:
    """Return task-neutral code-review guidelines."""
    submission_location_fact = (
        "- Submission File Location: save the final file to `./submission/submission.csv`."
        if submission_required
        else "- Submission File Location: `./submission/` exists, but final `submission.csv` "
        "generation is disabled. Do not require it."
    )
    return [
        "# Code Review Guidelines",
        "",
        "## Environment Facts (Do Not Flag)",
        "Trust these facts unless the task contract explicitly says otherwise:",
        "- `./input/`, `./working/`, and `./submission/` exist.",
        submission_location_fact,
        "- Assume configured libraries and explicitly selected model identifiers are available.",
        "- Do not replace or downgrade a requested model because it is unfamiliar.",
        "- Fix concrete logic or API errors without changing the intended architecture.",
        "- The configured execution budget is authoritative.",
        "",
        "## P0 - Data Leakage",
        "Flag preprocessing fitted before a train/validation split, including:",
        "- scaler or PCA fitted on all rows before splitting;",
        "- target encoding or other supervised features built from all rows;",
        "- SMOTE or other resampling applied before splitting.",
        "Correct order: split first -> fit on training only -> transform each split separately.",
        "Only flag a split strategy when it concretely violates the data dependency structure.",
        "",
        "## P1 - Critical Correctness",
        "### AutoRealize Contract Compliance",
        "- If the task contains `AutoRealize Structured Context` or `./input/autorealize_context.md` exists, follow its exact reading, output, evaluation, constraint, leakage, and scalar-score contracts.",
        "- Flag code that ignores non-standard CSV dialects, repeated-file groups, official output columns, optimization/RL protocols, or the scalar score formula.",
        "- Exact Source Schema Contract sheet and column names are authoritative for raw dataframe access unless code resolves actual names with explicit diagnostics.",
        "- Source Alias Guard aliases are unsafe raw names unless mapped to the listed exact physical column.",
        "- Business meanings may be used as derived variables only after resolving exact physical source fields.",
        "- Do not silently drop decision or prediction units because a date, time, or feature is missing. Apply an explicit contract-defined exclusion, fallback, default, or validation diagnostic.",
        "- Output/submission columns are generated result columns, not assumed raw input columns.",
        "- Empty generated tables must use `pd.DataFrame(rows, columns=OUTPUT_COLUMNS)` so validation and scoring do not fail with a zero-column KeyError.",
        "",
        "### Metric And Logic",
        "- Flag a metric that differs from the task contract, such as accuracy instead of F1 or MSE instead of RMSE.",
        "- Validation and final inference must use consistent preprocessing and postprocessing.",
        "",
        "### Optimization, Decision, And RL",
        "- Assignment, routing, scheduling, action, or optimization tasks need a deterministic solution validator and scorer.",
        "- A Final Validation Score is not trustworthy if code bypasses task-defined feasibility, population, identifier, schema, or duplicate checks.",
        "- If the validator reports a contract-invalid result and the contract says such a result is unscored, flag code that still prints it as a comparable `Final Validation Score` or saves it as a successful artifact.",
        "- Do not evaluate an objective formula outside the domain guaranteed by its hard constraints merely to obtain a finite scalar; invalid-domain values, sign reversals, and artificial negative costs are evaluator bugs.",
        "- Reward, training loss, proxy cost, and constants are not an official final score unless the contract defines them as such.",
        "- Empty, no-op, placeholder, or random decisions are invalid unless the official evaluator explicitly scores them.",
        "- Diagnostic or no-feasible solutions must still reach validation and print actionable diagnostics instead of crashing during output construction. They may print a comparable final score only when the official contract defines a penalty for them.",
        "- RL code needs a defined transition/environment, constrained-action handling such as an action mask or repair, and a deterministic final evaluator.",
        "- Flag an RL action-mask path that can send an all-invalid mask into softmax/sampling and produce NaN probabilities; it needs an explicit fallback, repair, backtrack, or infeasible termination path.",
        "- When RL uses curriculum learning, require level advancement evidence and final replay on the original full-scale environment and official evaluator; curriculum proxy scores are not final scores.",
        "- Flag unused RL scaffolds: defining Env, PolicyNetwork, PPO, DQN, or training helpers while the evaluated solution uses an unrelated non-RL path without disclosing it.",
        "- If a node claims RL, the final evaluated rollout must define and use `train_policy(data, artifact_dir)` plus `rollout(model_path, data)`, and rollout must use the saved policy artifact.",
        "- A non-RL decision solver must define `solve(model_path, data)`; prediction code must define both `train(data, artifact_dir)` and `predict(model_path, data)`.",
        "- Static optimization may use greedy, OR, local-search, or other non-RL solvers when they obey the same evaluator.",
        "- Relaxation, decomposition, or aggregate-state formulations must reconstruct a solution in the original output space and validate it with the shared evaluator; a relaxation objective alone is not a final solution score.",
        "- Incremental objective deltas may guide search only when they are mathematically consistent with `score_solution`; final output must be fully recomputed by the shared evaluator.",
        "- Lower bounds, best bounds, LP/MIP objectives, training rewards, and incumbent estimates must not be printed as `Final Validation Score` unless they are the independently evaluated returned solution.",
        "- Flag unconditional optimality claims. `exact_certified=true` or equivalent is valid only when code checks a trustworthy solver-proven optimal status or a closed bound gap and still replays the final solution through the evaluator.",
        "",
        "### Inference And Artifacts",
        "- Flag constant, random, or train-mean validation/test predictions.",
        "- Validation, test, and submission inference must use `predict(model_path, data)` or one equivalent shared inference path.",
        "- Use the selected best checkpoint rather than an arbitrary final epoch.",
        "- Save a reusable model or solver artifact under `./working/`, `./models/`, `./artifacts/`, or `./checkpoints/`.",
        "- The artifact must include preprocessing state needed for later inference.",
        "",
        "### Common API Checks",
        "- LightGBM: prefer callbacks such as `lgb.early_stopping(...)` where required by the installed API.",
        "- XGBoost: configure early stopping on the estimator when required by the installed API.",
        "- AdamW: import from `torch.optim`, not deprecated transformer aliases.",
        "- Avoid noisy progress bars and verbose training output.",
        "",
        "## Decision Rule",
        "Set `needs_revision=true` only for a concrete P0 leakage or P1 critical bug.",
        "Set `needs_revision=false` when no concrete P0/P1 issue exists.",
        "Approve by default rather than speculating about unknown models or library versions.",
    ]


def get_code_review_response_format() -> list[str]:
    """Return the structured review response contract."""
    return [
        "# Required Output",
        "- `needs_revision` (boolean): true only when a concrete issue must be fixed.",
        "- `reasoning` (string): exactly 2-4 concise sentences.",
        "- `revised_code` (string): only when `needs_revision=true`; otherwise null or omitted.",
        "",
        "When revision is required, use targeted SEARCH/REPLACE blocks:",
        "<<<<<<< SEARCH",
        "<exact original text>",
        "=======",
        "<replacement text>",
        ">>>>>>> REPLACE",
        "",
        "Preserve the intended method and architecture. Fix only the identified leakage, metric, API, or critical logic issue.",
    ]
