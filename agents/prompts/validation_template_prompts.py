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
    """Return compact review rules without dropping evaluation or delivery boundaries."""
    submission_location_fact = (
        "- Submission File Location: save the final file to `./submission/submission.csv`."
        if submission_required
        else "- Submission File Location: submission.csv generation disabled."
    )
    return [
        "# Code Review Guidelines",
        "",
        "## Environment Facts (Do Not Flag)",
        "Trust unless contract says otherwise:",
        "- `./input/`, `./working/`, `./submission/` exist.",
        submission_location_fact,
        "- Assume configured libraries and models are available; fix concrete logic errors without changing architecture.",
        "",
        "## P0 - Data Leakage",
        "Flag preprocessing fitted before split (scaler/PCA/SMOTE on all rows before train/val split). Correct: split first -> fit on train only.",
        "Category vocabularies are fitted state too: do not concat train with development/test for get_dummies, category discovery, imputation or encoding. Fit the vocabulary on train, transform other rows with unknown-category handling. Trace actual operations even when a helper's docstring claims 'fit-free'.",
        "Follow the task's fitting population through final model export. A train-only contract also prohibits a final train+dev refit; no holdout access does not make it permissible. Save the same model/preprocessing that earned the development score unless the task explicitly authorizes a separately labelled refit.",
        "Follow the task's invalid-prediction policy: never replace non-finite predictions with a constant or nan_to_num when the contract requires an error. Check before clipping; preserve every evaluation row.",
        "Trace execution from __main__; final holdout access anywhere (prediction/metrics/refitting/comparison) is P0 defect. Move to engine-only final_evaluate(data, artifact_dir).",
        "Only flag split strategy when it violates data dependency structure.",
        "",
        "## P1 - Critical Correctness",
        "### AutoRealize Contract",
        "- If AutoRealize context exists, follow exact contracts for reading, output, evaluation, leakage, score.",
        "- Source Schema exact sheet/column names are authoritative unless code resolves with diagnostics.",
        "- Whitelist columns limit raw access; derived transformations OK unless forbidden.",
        "- Don't silently drop units; apply explicit contract-defined exclusion/fallback.",
        "- Empty tables: use `pd.DataFrame(rows, columns=OUTPUT_COLUMNS)`.",
        "",
        "### Metric And Logic",
        "- Flag metric mismatch (accuracy vs F1, MSE vs RMSE).",
        "- Search score: one finite scalar with frozen formula/direction. Reject tuple/vector/Pareto scores or candidate-specific scalarization.",
        "- Validation and final inference: consistent preprocessing/postprocessing.",
        "- When training uses log/normalized targets, every early-stopping eval_set must use that same target scale. Check the values passed to the model, not only the final metric function.",
        "- If an interface signature is edited, update every CLI/helper call to it; train(data, artifact_dir) must still be called with its required arguments.",
        "- All candidates: same evaluation population, metric/direction, time horizon, split membership.",
        "- One method per candidate is intentional; the outer engine compares candidates. Missing model comparisons or an unwritten final report is not a candidate bug.",
        "- Fit blend weights/calibration/thresholds on train-fold OOF or a separate calibration split, never on the same development rows used for the reported score.",
        "- Temporal: check target timestamps vs prediction cutoffs; purge labels after earliest validation cutoff per task horizon.",
        "- Stacking: align validation IDs/folds/columns; reject in-sample training or tuning on final test.",
        "",
        "### Optimization, Decision, RL",
        "- Assignment/routing/scheduling: need deterministic validator and scorer.",
        "- Don't print invalid-domain objective values as Final Validation Score.",
        "- Reward/training-loss/proxy-cost are not official score unless contract defines them.",
        "- Empty/placeholder/random decisions invalid unless contract scores them.",
        "- RL: need environment, action-mask handling, deterministic evaluator. Flag all-invalid mask without fallback.",
        "- RL curriculum: show level advancement and final replay on full-scale env; proxy scores aren't final.",
        "- Flag unused RL scaffolds: the final evaluated rollout must actually load and use the saved policy artifact, not bypass it with a heuristic.",
        "- If curriculum learning is used, verify advancement and final evaluation on the original full-scale environment.",
        "- Relaxation/decomposition: reconstruct solution in original space and validate with shared evaluator.",
        "- A relaxation objective alone is not a final solution score.",
        "- Flag unconditional optimality claims; exact_certified valid only when solver proves optimal or checks bound gap.",
        "",
        "### Inference And Artifacts",
        "- Flag constant/random/train-mean as selected solution (dummy baselines OK for comparison).",
        "- Use selected best checkpoint, not arbitrary final epoch.",
        "- Save models/preprocessing in ./working/artifact/; dev-only payloads in ./working/evidence/.",
        "- Artifact: include preprocessing state for inference.",
        "- Time-series deployment: requested target hours have no observed labels and may be absent from history. Trace the same explicit target pairs with target labels removed and with target rows absent; prior-hour features and output coverage must stay identical. Never condition lag/predecessor assignment on target-label presence. Keep missing history as NaN, not zero labels.",
        "- train(data, artifact_dir) must actually consume its data argument, including auxiliary files, in a fresh directory. An ignored argument or a cwd-only source file is a broken training interface. JSON metadata must not contain tuple-key dictionaries.",
        "- The artifact manifest must include selected hyperparameters and preprocessing recipe relative to the solution. Fresh-directory train() must load that recipe or an identical embedded fallback, never silently use untuned defaults.",
        "- In final_evaluate: load from supplied artifact_dir root (engine promotes ./working/artifact to root); never use __file__/cwd paths.",
        "- Completed final_evaluate: return metrics (finite scalars), inference_checks, model_path (relative to artifact_dir).",
        "",
        "### Common API",
        "- LightGBM: use callbacks like lgb.early_stopping(...).",
        "- XGBoost: configure early stopping on estimator.",
        "- AdamW: import from torch.optim.",
        "- Avoid noisy progress bars.",
        "",
        "## Decision Rule",
        "Set needs_revision=true only for concrete P0 leakage or P1 critical bug. Approve by default.",
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
