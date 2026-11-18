"""Code Review Agent: LLM-based code review and fix for node code."""

import logging
import ast
import json
import time
from typing import cast

from llm import FunctionSpec, query
from agents.coder import plan_and_code_query
from agents.planner import build_chat_prompt_for_model
from engine.search_node import SearchNode
from agents.prompts.validation_template_prompts import get_code_review_prompt
from agents.prompts import get_internet_clarification
from agents.prompts.shared import SEARCH_EVALUATION_GUIDANCE
from agents.prompt_cache import candidate_review_section, task_section, routed_data_context
from llm.model_profiles import is_deepseek_model
from agents.prompt_policy import (
    configured_output_language,
    infer_task_family,
    output_language_instruction,
)
from engine.solution_protocol import (
    SolutionInterface,
    interface_contract_text,
    interface_for,
    preflight_code,
)

from agents.coder.diff_coder import SearchReplacePatcher

logger = logging.getLogger("AlgoEvolve")


class CodeReviewFailed(RuntimeError):
    """A candidate must not execute when critical review could not finish."""


def validate_review_edit_scope(original: str, revised: str, allowed_definitions) -> str:
    if not allowed_definitions:
        return revised
    allowed = set(allowed_definitions)
    def protected(code):
        tree = ast.parse(code)
        tree.body = [item for item in tree.body if not (
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in allowed)]
        return ast.dump(tree, include_attributes=False)
    if protected(original) != protected(revised):
        raise CodeReviewFailed("Focused interface repair changed protected code outside allowed function definitions")
    return revised

CODE_REVIEW_SPEC = FunctionSpec(
    name="submit_code_review",
    json_schema={
        "type": "object",
        "properties": {
            "needs_revision": {
                "type": "boolean",
                "description": (
                    "true if the code has issues that must be fixed "
                    "(metric mismatch, data leakage, or missing packages), "
                    "false if the code is correct."
                )
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "CONCISE explanation in EXACTLY 2-4 sentences. Explain: "
                    "(1) what issues were found, (2) why they matter, (3) what will be fixed. "
                    "DO NOT write detailed analysis or step-by-step checks - keep it brief."
                )
            },
            "revised_code": {
                "type": ["string", "null"],
                "description": (
                    "ONLY if needs_revision=true: Provide targeted fixes using SEARCH/REPLACE diff format.\n\n"
                    "**REQUIRED FORMAT** (use this for each fix):\n"
                    "<<<<<<< SEARCH\n"
                    "[exact code to find - copy verbatim with exact indentation]\n"
                    "=======\n"
                    "[corrected code]\n"
                    ">>>>>>> REPLACE\n\n"
                    "**CRITICAL**: \n"
                    "- SEARCH block must match original code EXACTLY (character-by-character, including all spaces/tabs)\n"
                    "- Only include the specific buggy lines that need fixing\n"
                    "- Can provide multiple SEARCH/REPLACE blocks for different bugs\n"
                    "- Do NOT output complete code - only diff blocks\n"
                    "- Do NOT wrap output in markdown code fences (``` or ```python) - output raw diff only\n\n"
                    "If needs_revision=false: MUST be null (DO NOT output code)."
                )
            }
        },
        "required": ["needs_revision", "reasoning"]
    },
    description="Submit code review for search node solution."
)


def run(agent, node: SearchNode) -> str:
    logger.debug(f"[review] node {node.id}")

    review_context = routed_data_context(agent, "code_review")
    review_task_desc = str(agent.task_desc or "")
    task_family = infer_task_family(agent)
    interface = interface_for(
        task_family=task_family,
        method_family=str(getattr(node, "method_family", "unknown") or "unknown"),
    )
    preflight = preflight_code(node.code, interface, require_final_evaluation=bool(getattr(agent, "has_autorealize_context", False)))
    prompt = get_code_review_prompt(
        task_desc=review_task_desc,
        code=node.code,
        submission_required=getattr(agent.acfg, "generate_submission", True),
    )
    internet_clarification = get_internet_clarification(getattr(agent.cfg, "pretrain_model_dir", ""))
    if "Instructions" not in prompt:
        prompt["Instructions"] = {}
    prompt["Instructions"]["Search score and final evaluation lifecycle"] = SEARCH_EVALUATION_GUIDANCE
    if "Implementation guideline" in prompt["Instructions"]:
        prompt["Instructions"]["Implementation guideline"].extend(internet_clarification)
    else:
        prompt["Instructions"]["⚠️ Internet Access Clarification"] = internet_clarification
    prompt["Instructions"]["Generated solution interface and safety preflight"] = [
        "Repair every deterministic preflight issue in this review while preserving the chosen method and evaluator.",
        f"Required interface: {interface}",
        "The corrected code must contain these exact leading parameter names and order:",
        interface_contract_text(interface),
        f"Current preflight: {preflight}",
        "Do not introduce eval(), exec(), os.system(), or subprocess shell=True.",
        output_language_instruction(configured_output_language(agent)),
    ]

    use_diff_for_review = agent.acfg.use_diff_mode
    retry_cfg = getattr(agent.acfg, "retries", None)
    max_retries = max(1, int(getattr(retry_cfg, "code_review_max_attempts", 2)))
    retry_delay = max(0.0, float(getattr(retry_cfg, "code_review_delay_seconds", 5.0)))
    primary_role = str(getattr(retry_cfg, "code_review_model_role", "feedback") or "feedback").strip().lower()
    if primary_role not in {"feedback", "code"}:
        logger.warning("Unknown code_review_model_role=%r; using feedback", primary_role)
        primary_role = "feedback"
    roles = [primary_role]
    if bool(getattr(retry_cfg, "code_review_escalate_to_code", True)) and primary_role != "code":
        roles.append("code")
    roles = roles[:max_retries]
    retry_evidence = None
    allowed_definitions = getattr(node, "review_allowed_definitions", None)

    for attempt, role in enumerate(roles):
        try:
            if attempt > 0:
                logger.info(
                    "Escalating code review to role=%s (attempt %s/%s) for node %s",
                    role,
                    attempt + 1,
                    len(roles),
                    node.id,
                )
                time.sleep(retry_delay)

            stage_cfg = agent.acfg.feedback if role == "feedback" else agent.acfg.code

            review_response = cast(
                dict,
                query(
                    system_message={
                        "Introduction": prompt.get("Introduction", ""),
                        "Instructions": prompt.get("Instructions", {}),
                    },
                    user_message=(
                        candidate_review_section(review_task_desc, review_context, node.code)
                        + "# Latest deterministic preflight evidence\n"
                        + json.dumps(
                            {
                                "task_family": task_family,
                                "required_interface": interface.__dict__,
                                "preflight": preflight.__dict__,
                                "actual_call_contract": {
                                    "default_search": "Only task training/development data may be fitted/predicted; never final holdout",
                                    "final_evaluate": "final_evaluate(Path(full_input_directory), Path(final_artifact_directory)); first argument is NOT a DataFrame or CSV filename. Resolve and parse the exact source files before dataframe validation. The output directory contains the frozen ./working/artifact contents at its root, or the legacy ./working contents when no canonical artifact directory exists. Load selected_config.json and recipe dependencies from final_artifact_directory (the supplied artifact_dir), never from __file__, cwd, another node, or an absolute workspace path. When status='completed', return metrics as finite numeric scalars in a non-empty object, inference_checks, and a relative model_path. When no independent evaluation applies, return status='not_applicable' and a concise reason without inventing business metrics.",
                                    "reusable_training": "train(task_training_data, Path(fresh_output_directory)); bind exported selected recipe relative to solution, fit supplied rows, do not enumerate or select candidates again",
                                    "independent_probe_evidence": getattr(node, "interface_probe_evidence", None),
                                    "allowed_function_edits": allowed_definitions,
                                    "focused_repair_rule": "When allowed_function_edits is present, change/add only those named functions. Preserve all other imports, constants, feature transformations, model grids, split logic and scoring exactly. No invented fallback recipe; missing actual selected configuration is an error.",
                                },
                                "previous_review_failure": retry_evidence,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        )
                        + "\n\n"
                        + output_language_instruction(configured_output_language(agent))
                    ),
                    func_spec=CODE_REVIEW_SPEC,
                    model=stage_cfg.model,
                    temperature=stage_cfg.temp,
                    stage_name=role,
                    stage_id="code_review",
                    structured_output_mode="json" if is_deepseek_model(stage_cfg.model) else None,
                    cfg=agent.cfg
                ),
            )

            if not isinstance(review_response, dict) or type(review_response.get("needs_revision")) is not bool:
                raise ValueError("Code review must return an explicit boolean needs_revision")
            needs_revision = review_response["needs_revision"]
            reasoning = review_response.get("reasoning", "")
            revised_code = review_response.get("revised_code")
            logger.info(
                "Code review for node %s using role=%s: needs_revision=%s",
                node.id,
                role,
                needs_revision,
            )
            logger.info(f"Reasoning: {reasoning}", extra={"verbose": True})

            if needs_revision:
                if revised_code and revised_code.strip():
                    if use_diff_for_review and (
                        "<<<<<<< SEARCH" in revised_code or "< SEARCH" in revised_code
                        ):
                        try:
                            logger.info("Code review returned diff format, applying patch")
                            patcher = SearchReplacePatcher()
                            patched_code, count = patcher.apply_patch(
                                revised_code, node.code, strict=True
                            )
                            if count > 0 and patched_code and patched_code != node.code:
                                logger.info(f"Successfully applied {count} review patch(es)")
                                return validate_review_edit_scope(node.code, patched_code.strip(), allowed_definitions)
                            raise ValueError(f"review diff did not apply (count={count})")
                        except Exception as e:
                            retry_evidence = str(e)
                            logger.warning(
                                "Failed to apply %s review patch: %s",
                                role,
                                e,
                            )
                            if attempt < len(roles) - 1:
                                continue
                            raise CodeReviewFailed("Required review patch could not be applied") from e
                    else:
                        # Full code revision (original behavior)
                        if use_diff_for_review:
                            retry_evidence = "Full code returned but exact SEARCH/REPLACE blocks are required"
                            logger.warning(
                                "%s reviewer returned full code while diff mode is enabled.",
                                role,
                            )
                            if attempt < len(roles) - 1:
                                continue
                            raise CodeReviewFailed("Review returned full code where a targeted patch is required")
                        else:
                            logger.info("Using revised code from reviewer")
                            return validate_review_edit_scope(node.code, revised_code.strip(), allowed_definitions)

                if attempt < len(roles) - 1:
                    retry_evidence = "Required revision has no applicable patch: " + str(reasoning)
                    logger.warning(
                        "Code review requested a revision without an applicable patch; escalating."
                    )
                    logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                    continue
                logger.error("Code review requested a revision without an applicable patch; blocking execution")
                logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                raise CodeReviewFailed("Critical review findings have no applicable repair")

            if revised_code is not None and revised_code.strip():
                logger.warning(
                    "Code review warning: needs_revision=False but revised_code was provided. "
                    "Ignoring revised_code and using original code."
                )
            logger.info("Code approved, using original code")
            return node.code

        except Exception as e:
            retry_evidence = str(e)
            error_msg = f"Code review failed with exception: {e}"
            if attempt < len(roles) - 1:
                logger.warning("%s - escalating to the next review role", error_msg)
                continue
            logger.error(f"{error_msg} - blocking execution")
            raise CodeReviewFailed(error_msg) from e

    raise CodeReviewFailed("Code review finished without approval or an applicable repair")


def regenerate_after_preflight_failure(
    agent,
    node: SearchNode,
    interface: SolutionInterface,
    preflight,
    *,
    attempt: int,
) -> tuple[str, str]:
    """Regenerate one candidate in place using exact deterministic failure evidence."""

    introduction = (
        "You are the corrective code-generation stage of an automated ML and decision search system. "
        "A complete candidate was rejected before execution by deterministic interface or safety checks. "
        "Regenerate the complete Python solution while preserving the candidate's method, evaluator, data contract, "
        "output schema, and useful implementation. The deterministic contract below is authoritative."
    )
    stable_context = task_section(
        str(getattr(agent, "task_desc", "") or ""),
        str(
            getattr(agent, "autorealize_context", "")
            or getattr(agent, "data_preview", "")
            or ""
        ),
    )
    exact_contract = interface_contract_text(interface)
    user_prompt = (
        f"{stable_context}\n"
        "# Stable regeneration rules\n"
        "- Preserve the selected algorithmic trajectory; this is a correctness regeneration, not a new search branch.\n"
        "- Return a complete executable Python program, not a patch and not partial snippets.\n"
        "- Preserve the official evaluator and validate the exact returned prediction or decision artifact.\n"
        "- Do not use eval(), exec(), os.system(), or subprocess shell=True.\n"
        f"- Required callable contract:\n{exact_contract}\n"
        f"- {output_language_instruction(configured_output_language(agent))}\n\n"
        "# Current rejected candidate\n"
        f"Plan:\n{str(getattr(node, 'plan', '') or '')}\n\n"
        f"Code:\n```python\n{str(getattr(node, 'code', '') or '')}\n```\n\n"
        "# Latest deterministic rejection evidence (authoritative)\n"
        f"Regeneration attempt: {attempt}\n"
        + json.dumps(
            {
                "required_interface": interface.__dict__,
                "exact_callable_contract": exact_contract,
                "preflight": preflight.__dict__,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )
    assistant_prefix = (
        "I will preserve the method and return a concise corrected plan followed by one complete fenced Python code block."
    )
    prompt = build_chat_prompt_for_model(
        agent.acfg.code.model,
        introduction,
        user_prompt,
        assistant_prefix,
    )
    plan, code = plan_and_code_query(
        agent,
        prompt,
        stage_id="preflight_code_regeneration",
    )
    if not str(code or "").strip():
        raise RuntimeError("Preflight regeneration returned no executable code")
    return plan, code
