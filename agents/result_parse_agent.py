import logging
import json
import math
import re
import time
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from engine.executor import ExecutionResult
from utils.metric import MetricValue, WorstMetricValue
from utils.response import trim_long_string, wrap_code
from utils.decision_validation import (
    decision_signal_summary as _dv_decision_signal_summary,
    extract_decision_validation_summary as _dv_extract_decision_validation_summary,
)
from engine.validation import call_validate, _validate_submission_with_retry, validate_submission_content_quality
from agents.prompt_cache import task_section
from agents.prompts import infer_task_mode
from agents.prompt_policy import (
    configured_output_language,
    output_language_instruction,
)
from engine.model_artifacts import find_model_artifacts

logger = logging.getLogger("AlgoEvolve")


FINAL_SCORE_RE = re.compile(
    r"Final\s+Validation\s+Score\s*[:=]\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
OPTIMIZATION_SOLVER_SUMMARY_PREFIX_RE = re.compile(
    r"^\s*Optimization\s+Solver\s+Summary\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE,
)
OPTIMIZATION_SOLVER_SIGNAL_KEYS = (
    "solver_family",
    "solve_status",
    "incumbent",
    "best_bound",
    "absolute_gap",
    "relative_gap",
    "variable_count",
    "constraint_count",
    "relaxation_used",
    "pruning_rule",
    "warm_start_used",
    "warm_start_source",
    "neighborhood",
    "exact_certified",
)

def _resolve_exp_id(agent) -> str:
    explicit = str(getattr(agent.cfg, "exp_id", "") or "").strip()
    if explicit:
        return explicit
    exp_name = str(getattr(agent.cfg, "exp_name", "") or "").strip()
    parts = exp_name.split("_", 2)
    if len(parts) >= 3 and parts[2].strip():
        return parts[2].strip()
    return exp_name or "task"


def _is_optimization_or_rl_agent(agent) -> bool:
    return infer_task_mode(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
        autorealize_context=getattr(agent, "data_preview", ""),
    ) in {"optimization", "rl"}


def _set_review_analysis(node: SearchNode, text: str | None) -> None:
    node.analysis = text
    # Kept as a serialized compatibility field for existing journals.
    node.parser_analysis = text


def _decision_signals_for_node(
    summary: dict | None,
    metric=None,
    optimization_summary: dict | None = None,
) -> dict | None:
    signals = dict(_dv_decision_signal_summary(summary)) if isinstance(summary, dict) else {}
    if metric is not None:
        signals["final_score"] = metric
    if isinstance(optimization_summary, dict) and optimization_summary:
        signals["optimization_solver"] = optimization_summary
    return signals or None


def _evaluation_review_context(agent) -> str:
    return str(
        getattr(agent, "autorealize_context", "")
        or getattr(agent, "data_preview", "")
        or ""
    )


def _fallback_human_insight(
    node: SearchNode,
    review_summary: str | None,
    *,
    language: str = "english",
) -> str:
    summary = (review_summary or "").strip()
    if summary:
        return trim_long_string(summary.replace("\n", " "), threshold=500, k=250)
    if language == "chinese":
        return "结果评审未返回可用洞察，请查看运行输出和调试理由。"
    return "The result review did not return a usable insight; inspect the execution output and debug reason."


def _set_fallback_human_insight(node: SearchNode, agent=None) -> str:
    language = configured_output_language(agent) if agent is not None else "english"
    fallback = _fallback_human_insight(
        node,
        node.parser_analysis or node.analysis,
        language=language,
    )
    node.llm_insight = fallback
    return fallback


def _extract_optimization_solver_summary(text: str) -> dict | None:
    """Extract compact, factual solver telemetry without turning it into a gate."""
    summaries: list[dict] = []
    for line in (text or "").splitlines():
        match = OPTIMIZATION_SOLVER_SUMMARY_PREFIX_RE.match(line)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        compact = {
            key: parsed[key]
            for key in OPTIMIZATION_SOLVER_SIGNAL_KEYS
            if key in parsed and parsed[key] is not None
        }
        if compact:
            summaries.append(compact)
    return summaries[-1] if summaries else None


_extract_decision_validation_summary = _dv_extract_decision_validation_summary


def _short_json(value, *, limit: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + (" ..." if len(text) > limit else "")


metric_direction_func_spec = FunctionSpec(
    name="determine_metric_direction",
    json_schema={
        "type": "object",
        "properties": {
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE, RMSE, MAE, loss, error rate), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy, F1 score, AUC, precision, recall, Jaccard score, IoU).",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this metric direction is chosen based on the task's evaluation metric description.",
            },
        },
        "required": [
            "lower_is_better",
            "reasoning",
        ],
    },
    description="Determine whether the evaluation metric should be minimized or maximized based on the task description.",
)


def determine_metric_direction(agent) -> None:
    logger.info("=" * 80)
    logger.info("Starting pre-determination of metric optimization direction...")
    logger.info("=" * 80)

    authoritative_context = str(getattr(agent, "autorealize_context", "") or "")
    direction_match = re.search(
        r"(?im)^\s*[-*]?\s*metric_direction\s*:\s*`?(minimize|maximize)\b",
        authoritative_context,
    )
    if direction_match:
        direction = direction_match.group(1).lower()
        agent.metric_maximize = direction == "maximize"
        agent.metric_maximize_reasoning = (
            f"AutoRealize evaluation contract explicitly sets metric_direction={direction}."
        )
        logger.info(
            "Metric direction loaded directly from AutoRealize contract: maximize=%s",
            agent.metric_maximize,
        )
        return

    prompt = """You are analyzing a machine learning competition task. Your task is to determine whether the evaluation metric should be minimized or maximized.

    **IMPORTANT: Focus on the EVALUATION section in the task description, which specifies the metric used to score submissions.**

    Based on the evaluation metric mentioned in the task description, determine:
    - If the metric should be MINIMIZED (lower is better), set lower_is_better to TRUE.
    Examples: MSE, RMSE, MAE, Cross-Entropy Loss, Log Loss, Error Rate
    - If the metric should be MAXIMIZED (higher is better), set lower_is_better to FALSE.
    Examples: Accuracy, F1 Score, AUC-ROC, Precision, Recall, Jaccard Score, IoU, mAP

    **Pay special attention to:**
    1. The "Evaluation" or "Metric" section in the task description
    2. Common metric conventions (e.g., accuracy is always maximized, MSE is always minimized)
    3. Whether the metric measures error/loss (minimize) or performance/quality (maximize)

    Provide clear reasoning based on the evaluation metric specified in the task.
    """
    user_prompt = task_section(agent.task_desc, _evaluation_review_context(agent))

    retry_cfg = getattr(agent.acfg, "retries", None)
    max_retries = max(1, int(getattr(retry_cfg, "metric_direction_max_attempts", 3)))
    retry_delay = max(0.0, float(getattr(retry_cfg, "metric_direction_delay_seconds", 1.0)))
    for attempt in range(1, max_retries + 1):
        try:
            if attempt == 1:
                logger.info(f"Attempt {attempt}/{max_retries} to determine metric direction...")
            else:
                logger.info(f"Retry attempt {attempt}/{max_retries} to determine metric direction...")
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=user_prompt,
                    func_spec=metric_direction_func_spec,
                    model=agent.acfg.feedback.model,
                    temperature=agent.acfg.feedback.temp,
                    stage_name="feedback",
                    cfg=agent.cfg
                ),
            )

            lower_is_better = response["lower_is_better"]
            agent.metric_maximize = not lower_is_better
            reasoning = response.get("reasoning", "")
            agent.metric_maximize_reasoning = reasoning

            logger.info("=" * 80)
            logger.info("Pre-determination completed successfully:")
            logger.info(f"  - lower_is_better = {lower_is_better}")
            logger.info(f"  - maximize = {agent.metric_maximize}")
            logger.info(f"  - Reasoning: {reasoning}")
            logger.info("=" * 80)
            logger.info(f"All subsequent nodes MUST use maximize={agent.metric_maximize}, otherwise they will be marked as buggy")
            logger.info("=" * 80)
            return

        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                logger.info("Retrying in a moment...")
                time.sleep(retry_delay)
            else:
                logger.error("=" * 80)
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                logger.error("Using default value maximize=True (assuming higher is better)")
                logger.error("=" * 80)
                agent.metric_maximize = True
                agent.metric_maximize_reasoning = "Default: assuming higher is better (most common case)"


def get_review_func_spec(use_memory: bool, optimization_rl: bool = False) -> FunctionSpec:
    bug_description = (
        "Judge from the task contract, implementation, raw execution output, and read-only execution evidence. "
        "Set true for a concrete execution or result-integrity bug: crash, unusable or non-comparable metric, "
        "wrong score formula/source, bypassed required validation, fabricated/empty result presented as valid, "
        "or an officially unscored invalid result presented as comparable. A candidate-printed scalar or JSON "
        "summary is only a claim, not proof. Poor solution quality alone is not a bug when the task's evaluator "
        "legitimately scores that result. Do not invent universal constraints or require optional diagnostics."
        if optimization_rl
        else "Judge from the task contract, complete implementation, raw execution output, and runtime facts. "
             "Set true for crashes, missing/non-comparable metrics, wrong evaluator paths, train/validation leakage, "
             "fabricated or constant outputs presented as real inference, or other concrete result-integrity bugs. "
             "Poor model quality alone is not a bug."
    )
    metric_description = (
        "For optimization/RL tasks, return the task-comparable scalar score only after reviewing whether it "
        "actually evaluates the returned solution under the task contract. Candidate-reported Final Validation "
        "Score and summary fields are untrusted evidence, not acceptance requirements. Return null when the run "
        "is buggy or the printed number is not a comparable task score."
        if optimization_rl
        else "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null."
    )
    properties = {
        "verdict": {
            "type": "string",
            "enum": ["accept", "reject", "uncertain"],
            "description": "accept only when the result is trustworthy and comparable; reject for a concrete bug; uncertain only when evidence is genuinely insufficient.",
        },
        "is_bug": {
            "type": "boolean",
            "description": bug_description,
        },
        "summary": {
            "type": "string",
            "description": "Provide a concise summary (2-3 sentences) of the execution outcome. "
                           "If successful, describe the key empirical results. "
                           "If failed, describe the error encountered. "
                           "Focus on observations only — do not include suggestions for improvement.",
        },
        "reason_codes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short stable reason codes such as runtime_exception, metric_missing, evaluator_bypass, degenerate_solution, suspicious_score, or accepted.",
        },
        "debug_hint": {
            "type": "string",
            "description": "One concrete repair direction when rejected/uncertain; empty when accepted and no follow-up is needed.",
        },
        "technical_summary": {
            "type": "string",
            "description": "Concise factual review for downstream coding agents, including why the score is or is not trustworthy.",
        },
        "human_insight": {
            "type": "string",
            "description": "A concise 2-4 sentence UI-facing explanation of outcome, bottleneck, and next step without changing recorded facts.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the verdict from 0.0 to 1.0.",
        },
        "metric": {
            "type": "number",
            "description": metric_description,
        },
        "lower_is_better": {
            "type": "boolean",
            "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
        },
    }
    required = [
        "verdict",
        "is_bug",
        "summary",
        "reason_codes",
        "debug_hint",
        "technical_summary",
        "human_insight",
        "confidence",
        "metric",
        "lower_is_better",
    ]
    if use_memory:
        properties["code_summary"] = {
            "type": "string",
            "description": "Write a summary including the methods used in each stage of the code, such as data preprocessing, feature engineering, model architecture, etc.",
        }
        required.append("code_summary")
    return FunctionSpec(
        name="submit_review",
        json_schema={"type": "object", "properties": properties, "required": required},
        description="Submit a review evaluating the output of the training script.",
    )


RESULT_ADJUDICATOR_SPEC = FunctionSpec(
    name="submit_result_adjudication",
    json_schema={
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["accept", "reject"]},
            "is_bug": {"type": "boolean"},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
            "debug_hint": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["verdict", "is_bug", "reason_codes", "debug_hint", "confidence"],
    },
    description="Resolve only an anomalous or uncertain result-review verdict.",
)


def _build_introduction(agent) -> str:
    use_memory = getattr(agent.acfg, "use_global_memory", False)
    submission_required = getattr(agent.acfg, "generate_submission", True)
    optimization_rl = _is_optimization_or_rl_agent(agent)
    intro = (
        "You are a Kaggle grandmaster attending a competition. "
        "You have written code to solve this task and now need to evaluate the output of the code execution. "
        "You should determine if there were any bugs as well as report the empirical findings.\n\n"
        "You MUST respond with a JSON object containing ALL of the following fields:\n"
        "- \"is_bug\": (boolean) true if execution failed or has bugs, false otherwise. Must be a JSON boolean (true/false), NOT a string.\n"
        "- \"summary\": (string) A concise 2-3 sentence summary of the execution outcome.\n"
        "- \"metric\": (number or null) The validation metric value as a raw JSON number (e.g. 0.9995), NOT a string. If failed, use null.\n"
        "- \"lower_is_better\": (boolean) true if the metric should be minimized, false if maximized. Must be a JSON boolean (true/false), NOT a string.\n"
        "- Also return verdict, reason_codes, debug_hint, technical_summary, human_insight, and confidence exactly as required by the schema.\n"
    )
    if not submission_required:
        intro += (
            "\nConfig note: final submission.csv generation is disabled for this run. "
            "Do NOT mark the execution as buggy merely because it did not create a submission file; "
            "judge success by execution correctness and the reported validation metric.\n"
        )
    if optimization_rl:
        intro += (
            "\nOptimization/RL/decision-task review rules:\n"
            "- You are the outcome reviewer. No deterministic parser has accepted or rejected this node before you.\n"
            "- Treat `Final Validation Score`, `Decision Validation Summary`, solver summaries, and every value printed by candidate code as untrusted claims. A finite number alone never proves success.\n"
            "- Verify from the task contract, implementation, raw output, and execution facts that the reported metric evaluates the actual returned solution and is comparable with other nodes.\n"
            "- Optional fields such as `score_components`, `final_score_source`, `evaluator_self_tests_passed`, and `is_feasible` are evidence, not universal acceptance requirements.\n"
            "- Do NOT require universal progress or violation fields; those are task-specific diagnostics.\n"
            "- Poor objective value or weak solution quality is not a bug when the official evaluator can score the returned solution.\n"
            "- A partial or infeasible solution is non-buggy only when the authoritative evaluator explicitly assigns it a comparable penalty score.\n"
            "- Mark `is_bug=true` and return `metric=null` when code crashes, bypasses required validation, evaluates outside the valid domain, prints a bound/reward/proxy instead of the returned solution score, emits an empty/placeholder result as valid, or reports a contract-invalid/unscored result as comparable.\n"
            "- Treat an extreme score as an anomaly that needs direct evidence. In particular, for a minimization task, cost=0 is suspicious when code can return no assignments, omit unassigned penalties, skip evaluation rows, reverse a sign, or score an empty output. Zero is not automatically impossible, but accept it only when implementation and output prove it is legitimate.\n"
            "- Use only constraints and validity rules supported by this task's authoritative context; do not impose coverage, feasibility, artifact, or output rules copied from another task.\n"
            "- If optional diagnostics exist, preserve their warnings, examples, infeasibility reasons, or objective-component details for later improvement.\n"
        )
    if use_memory:
        intro += (
            "- \"code_summary\": (string) A concise method summary of the code, covering key parts such as "
            "data preprocessing, feature engineering, model architecture/training, and validation strategy.\n"
        )
    intro += (
        "\nDo NOT omit any field.\n"
        + output_language_instruction(configured_output_language(agent))
    )
    return intro


def _check_submission_file(agent, node: SearchNode) -> bool:
    correct_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    if not correct_path.exists():
        wrong_path = agent.cfg.workspace_dir / f"submission_{node.id}.csv"
        if wrong_path.exists():
            correct_path.parent.mkdir(parents=True, exist_ok=True)
            wrong_path.rename(correct_path)
            logger.warning(f" {wrong_path} are moved to {correct_path}")

    return correct_path.exists()


def _save_code_summary(agent, node: SearchNode, response: dict):
    use_memory = getattr(agent.acfg, "use_global_memory", False)
    if not use_memory:
        node.code_summary = None
        return
    if "code_summary" in response and response["code_summary"]:
        node.code_summary = response["code_summary"]
        logger.info(f"Saved code summary for node {node.id}")
    else:
        logger.warning(f"Node {node.id} missing code_summary in response")
        node.code_summary = None


def _determine_buggy(
    node: SearchNode,
    response: dict,
    has_csv_submission: bool,
    requires_submission: bool = True,
    allow_missing_submission: bool = False,
):
    failure_reasons = []
    if response["is_bug"]:
        failure_reasons.append("execution error detected")
    if node.exc_type is not None:
        failure_reasons.append(f"exception raised: {node.exc_type}")
    if response["metric"] is None:
        failure_reasons.append("no metric value reported")
    if requires_submission and not has_csv_submission and not allow_missing_submission:
        failure_reasons.append("submission file not found")

    node.is_buggy = len(failure_reasons) > 0
    if node.is_buggy:
        logger.warning(f"Node {node.id} marked as buggy: {'; '.join(failure_reasons)}")


def _validate_format_with_retry(agent, node: SearchNode):
    exp_id = _resolve_exp_id(agent)
    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    status, res = _validate_submission_with_retry(
        exp_id=exp_id,
        submission_path=submission_path,
        cfg=agent.cfg,
        max_attempts=2,
        sample_path=None,
    )

    if status:
        if not res['is_valid']:
            logger.warning(f"[validate] node {node.id}: invalid after retry attempts.")
            node.is_valid = False
            node.is_buggy = True
            node._term_out.append(f"\n{res['result']}")
            _set_review_analysis(
                node,
                f"FORMAT_ERROR: Execution succeeded but submission file failed format validation.\n\nDetails:\n{res['result']}",
            )
        else:
            _check_content_quality(agent, node, submission_path)
    else:
        logger.error(f"An unexpected error occurred: {res}, skip this stage.")
        logger.info(f"Node {node.id} format validation passed. Now checking content quality...")
        content_valid, content_error = validate_submission_content_quality(
                submission_path=submission_path,
                sample_path=None,
                constant_threshold=0.95,
            )

        if not content_valid:
            _mark_content_quality_failure(node, content_error)
        else:
            logger.info(f"[validate] node {node.id}: valid")
            node.is_valid = True


def _append_analysis_note(node: SearchNode, note: str) -> None:
    if not note:
        return
    if node.analysis:
        if note not in node.analysis:
            node.analysis = f"{node.analysis}\n\n[Non-fatal warning] {note}"
    else:
        node.analysis = f"[Non-fatal warning] {note}"
    node.parser_analysis = node.analysis


def _validate_format_simple(agent, node: SearchNode):
    exp_id = _resolve_exp_id(agent)
    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    status, res = call_validate(exp_id=exp_id, submission_path=submission_path)
    if status:
        if not res['is_valid']:
            logger.warning(f"[validate] node {node.id}: invalid.")
            node.is_valid = False
            node.is_buggy = True
            node._term_out.append(f"\n{res['result']}")
            _set_review_analysis(
                node,
                f"FORMAT_ERROR: Execution succeeded but submission file failed format validation.\n\nDetails:\n{res['result']}",
            )
        else:
            _check_content_quality(agent, node, submission_path)
    else:
        logger.error(f"An unexpected error occurred: {res}, skip this stage.")


def _check_content_quality(agent, node: SearchNode, submission_path):
    logger.info(f"Node {node.id} format validation passed. Now checking content quality...")
    content_valid, content_error = validate_submission_content_quality(
            submission_path=submission_path,
            sample_path=None,
            constant_threshold=0.95,
        )

    if not content_valid:
        _mark_content_quality_failure(node, content_error)
    else:
        logger.info(f"✅ Node {node.id} passed both format and content quality checks.")
        node.is_valid = True


def _mark_content_quality_failure(node: SearchNode, content_error):
    logger.warning(f"Node {node.id} is marked as buggy due to content quality check failure.")
    node.is_valid = False
    node.is_buggy = True
    error_message = (
        "Submission format is correct, but content quality check FAILED:\n\n"
        f"{content_error}\n\n"
        "🚨 CRITICAL: All predictions must come from actual model inference.\n"
        "You must:\n"
        "1. Load each test sample\n"
        "2. Preprocess it with the same transformations as training\n"
        "3. Run model.predict() / model.forward() on the sample\n"
        "4. Use the model's output as the prediction\n\n"
        "Filling submissions with constants, placeholders, or dummy values is STRICTLY FORBIDDEN."
    )
    node._term_out.append(f"\n{error_message}")
    _set_review_analysis(
        node,
        f"CONTENT_QUALITY_ERROR: This previous solution runs without bugs and has correct format, but failed content quality check.\n\nDetails:\n{content_error}",
    )


def _validate_metric_direction(agent, node: SearchNode, response: dict):
    returned_maximize = not response["lower_is_better"]
    if agent.metric_maximize is not None and returned_maximize != agent.metric_maximize:
        logger.error("=" * 80)
        logger.error(f"METRIC DIRECTION MISMATCH for Node {node.id}!")
        logger.error(f"  - Returned lower_is_better = {response['lower_is_better']} (maximize={returned_maximize})")
        logger.error(f"  - Pre-determined maximize = {agent.metric_maximize}")
        logger.error(f"  - Marking this node as BUGGY, will NOT update top candidates")
        logger.error("=" * 80)
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = (
            f"{node.analysis}\n\n[ERROR] Metric direction mismatch detected:\n"
            f"- Returned lower_is_better={response['lower_is_better']} (maximize={returned_maximize})\n"
            f"- Expected maximize={agent.metric_maximize}\n"
            f"- Pre-determination reasoning: {agent.metric_maximize_reasoning or 'N/A'}\n"
            f"This node is marked as buggy and will not be considered for best/top candidates."
        )
        node.parser_analysis = node.analysis
    else:
        logger.info(f"Node {node.id} metric direction validated: maximize={agent.metric_maximize}")
        node.metric = MetricValue(
            response["metric"], maximize=agent.metric_maximize
        )


def _save_to_global_memory(agent, node: SearchNode):
    if agent.global_memory and not node.is_buggy and node.metric and node.metric.value is not None:
        try:
            parent_node = node.parent
            agent.global_memory.save_node(node, parent_node)
        except Exception as e:
            logger.warning(f"[AgentSearch] Failed to save node {node.id} to global memory: {e}")


def _extract_final_validation_score(text: str) -> float | None:
    matches = FINAL_SCORE_RE.findall(text or "")
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _normalize_review_response(agent, response: dict) -> dict:
    response.setdefault("is_bug", True)
    response.setdefault("summary", "No summary returned by model.")
    response.setdefault("technical_summary", response.get("summary"))
    response.setdefault("human_insight", response.get("technical_summary") or response.get("summary"))
    response.setdefault("reason_codes", ["review_incomplete"] if response.get("is_bug") else ["accepted"])
    response.setdefault("debug_hint", "")
    response.setdefault("confidence", 0.8)
    response.setdefault("verdict", "reject" if response.get("is_bug") else "accept")
    response.setdefault("metric", None)
    response.setdefault(
        "lower_is_better",
        not agent.metric_maximize if agent.metric_maximize is not None else False,
    )

    metric_val = response.get("metric")
    if not isinstance(metric_val, (int, float)):
        try:
            response["metric"] = float(metric_val)
        except (TypeError, ValueError):
            response["metric"] = None
    if isinstance(response.get("metric"), (int, float)) and not math.isfinite(float(response["metric"])):
        response["metric"] = None

    for bool_field in ("is_bug", "lower_is_better"):
        v = response.get(bool_field)
        if isinstance(v, str):
            response[bool_field] = v.strip().lower() not in ("false", "0", "no", "")
    verdict = str(response.get("verdict") or "").strip().lower()
    if verdict not in {"accept", "reject", "uncertain"}:
        verdict = "reject" if response.get("is_bug") else "accept"
    response["verdict"] = verdict
    if verdict != "accept":
        response["is_bug"] = True
    try:
        response["confidence"] = min(1.0, max(0.0, float(response.get("confidence", 0.5))))
    except (TypeError, ValueError):
        response["confidence"] = 0.8
    if not isinstance(response.get("reason_codes"), list):
        response["reason_codes"] = [str(response.get("reason_codes") or "review_incomplete")]
    return response


def _apply_review_response(agent, node: SearchNode, response: dict) -> SearchNode:
    response = _normalize_review_response(agent, response)

    requires_submission = getattr(agent.acfg, "generate_submission", True)
    has_csv_submission = _check_submission_file(agent, node) if requires_submission else True
    scorable_decision_run = (
        _is_optimization_or_rl_agent(agent)
        and response.get("is_bug") is False
        and response.get("metric") is not None
    )

    decision_summary = response.pop("_decision_summary", None)
    optimization_summary = _extract_optimization_solver_summary(node.term_out)
    technical_summary = str(response.get("technical_summary") or response["summary"])
    debug_hint = str(response.get("debug_hint") or "").strip()
    review_text = technical_summary
    if debug_hint:
        review_text = f"{review_text}\n\nDebug hint: {debug_hint}"
    _set_review_analysis(node, review_text)
    node.llm_insight = str(response.get("human_insight") or "").strip() or _fallback_human_insight(
        node,
        technical_summary,
        language=configured_output_language(agent),
    )
    node.review_verdict = response.get("verdict")
    node.review_reason_codes = list(response.get("reason_codes") or [])
    node.review_confidence = response.get("confidence")
    node.decision_signals = _decision_signals_for_node(
        decision_summary,
        response.get("metric"),
        optimization_summary,
    )
    _save_code_summary(agent, node, response)
    _determine_buggy(
        node,
        response,
        has_csv_submission,
        requires_submission=requires_submission,
        allow_missing_submission=scorable_decision_run,
    )

    if not node.is_buggy and requires_submission and scorable_decision_run:
        node.is_valid = True
        if has_csv_submission:
            _append_analysis_note(
                node,
                "Generic Kaggle-style submission format/content validation was skipped because "
                "the final reviewer accepted this optimization/RL node's scalar decision score. "
                "Task-specific quality issues remain visible in the Decision Validation Summary "
                "and should be improved in later nodes.",
            )
        else:
            _append_analysis_note(
                node,
                "No submission file was found, but the final reviewer accepted the node's scalar decision score. "
                "The node remains a valid search result and is retained for improve/debug; submission generation "
                "remains a separate delivery follow-up.",
            )
    elif not node.is_buggy and requires_submission:
        _validate_format_with_retry(agent, node)
    elif not node.is_buggy:
        node.is_valid = True

    if node.is_buggy:
        node.metric = WorstMetricValue()
    else:
        _validate_metric_direction(agent, node, response)
    node.parser_analysis = node.analysis

    status = "FAIL" if node.is_buggy else "PASS"
    metric_val = node.metric.value if node.metric else None
    logger.info(f"[parse] node {node.id}: {status} | metric={metric_val}")

    _save_to_global_memory(agent, node)

    return node


def _needs_result_adjudication(agent, response: dict, reported_score: float | None) -> bool:
    retry_cfg = getattr(agent.acfg, "retries", None)
    if not bool(getattr(retry_cfg, "result_adjudicator_on_anomaly", True)):
        return False
    normalized = _normalize_review_response(agent, dict(response))
    if normalized.get("verdict") == "reject":
        return False
    if normalized.get("verdict") == "uncertain" or normalized.get("confidence", 1.0) < 0.65:
        return True
    if normalized.get("verdict") != "accept":
        return False
    if reported_score is None:
        return False
    if agent.metric_maximize:
        return math.isclose(float(reported_score), 1.0, rel_tol=0.0, abs_tol=1e-12)
    return math.isclose(float(reported_score), 0.0, rel_tol=0.0, abs_tol=1e-12)


def _adjudicate_result(
    agent,
    *,
    review_user_message: str,
    primary_response: dict,
) -> dict:
    system_message = (
        "You are the final adjudicator for one anomalous or uncertain AutoML result. "
        "Resolve only whether a concrete result-integrity bug exists. Recheck the task contract, complete code, "
        "raw output, runtime facts, evaluator path, returned solution population, and penalties. Do not write a "
        "second summary and do not reject merely because solution quality is poor. Extreme scores require direct proof; "
        "for prediction also check target/future leakage, split contamination, and transforms fitted outside training data. "
        + output_language_instruction(configured_output_language(agent))
    )
    user_message = (
        f"{review_user_message}\n\n# Primary review requiring adjudication\n"
        + json.dumps(primary_response, ensure_ascii=False, sort_keys=True, default=str)
        + "\n\n# Latest instruction\nReturn only the adjudication fields in the required schema."
    )
    adjudication = cast(
        dict,
        query(
            system_message=system_message,
            user_message=user_message,
            func_spec=RESULT_ADJUDICATOR_SPEC,
            model=agent.acfg.feedback.model,
            temperature=0.0,
            stage_name="feedback",
            cfg=agent.cfg,
        ),
    )
    merged = dict(primary_response)
    for key in ("verdict", "is_bug", "reason_codes", "debug_hint", "confidence"):
        if key in adjudication:
            merged[key] = adjudication[key]
    merged["adjudicated"] = True
    return merged


def run(agent, node: SearchNode, exec_result: ExecutionResult) -> SearchNode:
    retry_cfg = getattr(agent.acfg, "retries", None)
    max_retries = max(1, int(getattr(retry_cfg, "result_parse_max_attempts", 3)))
    for retry_idx in range(max_retries):
        try:
            logger.info(f"Agent is parsing execution results for node {node.id}")

            node.absorb_exec_result(exec_result)
            introduction = _build_introduction(agent)
            optimization_rl = _is_optimization_or_rl_agent(agent)
            decision_summary = (
                _extract_decision_validation_summary(node.term_out)
                if optimization_rl
                else None
            )
            review_context = _evaluation_review_context(agent) if optimization_rl else ""
            score = _extract_final_validation_score(node.term_out)
            optimization_summary = (
                _extract_optimization_solver_summary(node.term_out)
                if optimization_rl
                else None
            )
            execution_evidence = {
                "evidence_notice": (
                    "Read-only facts extracted from this execution. Candidate-reported score and summaries "
                    "are untrusted evidence; they do not determine the review verdict."
                ),
                "execution_time_seconds": node.exec_time,
                "exception_type": node.exc_type,
                "exception_info": (
                    _short_json(node.exc_info, limit=1600)
                    if node.exc_info is not None
                    else None
                ),
                "candidate_reported_final_score": score,
                "candidate_reported_decision_summary": (
                    _short_json(decision_summary, limit=2000)
                    if decision_summary is not None
                    else None
                ),
                "candidate_reported_solver_summary": (
                    _short_json(optimization_summary, limit=1200)
                    if optimization_summary is not None
                    else None
                ),
                "solution_interface": getattr(node, "solution_interface", None),
                "preexecution_preflight": getattr(node, "preflight_report", None),
                "model_artifacts": [
                    str(path)
                    for path in find_model_artifacts(agent.cfg.workspace_dir, str(node.id))
                ],
                "submission_required": bool(getattr(agent.acfg, "generate_submission", True)),
                "submission_exists": (
                    agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
                ).exists(),
            }
            prompt = {
                "Introduction": introduction,
                "Implementation": wrap_code(node.code),
                "Execution output": wrap_code(node.term_out, lang=""),
            }
            review_user_message = (
                f"{task_section(agent.task_desc, review_context)}\n"
                + f"# Implementation\n{prompt['Implementation']}\n\n"
                + f"# Execution output\n{prompt['Execution output']}\n\n"
                + "# Read-only execution evidence (evidence only, not a verdict)\n"
                + json.dumps(execution_evidence, ensure_ascii=False, sort_keys=True, default=str)
                + "\n\n# Latest output-language instruction\n"
                + output_language_instruction(configured_output_language(agent))
            )

            response = cast(
                dict,
                query(
                    system_message={"Introduction": introduction},
                    user_message=review_user_message,
                    func_spec=get_review_func_spec(
                        getattr(agent.acfg, "use_global_memory", False),
                        optimization_rl=optimization_rl,
                    ),
                    model=agent.acfg.feedback.model,
                    temperature=agent.acfg.feedback.temp,
                    stage_name="feedback",
                    cfg=agent.cfg
                ),
            )

            if optimization_rl:
                response["_decision_summary"] = decision_summary

            if _needs_result_adjudication(agent, response, score):
                logger.info("Result anomaly/uncertainty triggers adjudication for node %s", node.id)
                try:
                    response = _adjudicate_result(
                        agent,
                        review_user_message=review_user_message,
                        primary_response=response,
                    )
                except Exception as adjudication_error:  # noqa: BLE001
                    logger.warning(
                        "Result adjudication failed for node %s; preserving conservative primary verdict: %s",
                        node.id,
                        adjudication_error,
                    )

            return _apply_review_response(agent, node, response)
        except Exception as e:
            logger.warning(f"[parse] tool call failed: {e}")
            continue

    logger.error(f"All {max_retries} parse attempts failed for node {node.id}, marking as buggy")
    node.is_buggy = True
    node.metric = WorstMetricValue()
    _set_review_analysis(node, "Execution result review failed after multiple attempts.")
    _set_fallback_human_insight(node, agent)
    return node
