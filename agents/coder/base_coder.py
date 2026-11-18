"""Base code generation mode (single-shot plan + code).

The simplest generation strategy: one LLM call produces a natural language
plan followed by a complete code block. Used as the default / fallback mode
when diff or stepwise generation is not enabled or fails.
"""

from __future__ import annotations

import logging
from typing import Tuple

from llm import generate
from utils.response import extract_plan_and_code, code_extraction_diagnostic
from agents.prompts import plan_and_code_response_format
from agents.prompts.shared import PLAN_AND_CODE_RESPONSE_FORMAT

logger = logging.getLogger("AlgoEvolve")


# ============ Response format prompt (rewrite mode specific) ============

RESPONSE_FORMAT = {
    "Response format": plan_and_code_response_format(
        "the complete runnable solution, including the required validation metric output"
    )
}


def complete_script_prompt(prompt):
    """Keep the task prefix intact while supplying the full-rewrite response contract."""
    suffix = "\n\n# Required response\n" + RESPONSE_FORMAT["Response format"]
    if isinstance(prompt, str):
        return prompt if PLAN_AND_CODE_RESPONSE_FORMAT in prompt else prompt + suffix
    if isinstance(prompt, dict) and isinstance(prompt.get("user"), str):
        if PLAN_AND_CODE_RESPONSE_FORMAT in prompt["user"]:
            return prompt
        return {**prompt, "user": prompt["user"] + suffix}
    raise TypeError("Full script generation requires text or a chat dictionary with a user string")


def plan_and_code_query(
    agent_instance,
    prompt,
    retries: int = 3,
    stage_id: str = "candidate_code_generation",
) -> Tuple[str, str]:
    """Generate a complete plan and script; never return prose as executable code."""
    retry_cfg = getattr(agent_instance.acfg, "retries", None)
    retries = max(
        1,
        int(getattr(retry_cfg, "code_generation_extract_max_attempts", retries)),
    )
    completion_text = None
    current_prompt = prompt
    for attempt in range(retries):
        completion_text = generate(
            prompt=current_prompt,
            temperature=agent_instance.acfg.code.temp,
            cfg=agent_instance.cfg,
            stage_id=stage_id,
        )
        nl_text, code = extract_plan_and_code(
            completion_text,
            default_plan="Implement the requested complete solution and report its validation metric.",
            allow_last_complete_block=True,
        )

        if code:
            if completion_text.lstrip().startswith("```"):
                logger.info("Accepted a valid code-first response without regenerating it.")
            return nl_text, code

        logger.warning("No complete Python script extracted from response (%s chars), attempt %s/%s",
                       len(completion_text or ""), attempt + 1, retries)
        if attempt + 1 < retries:
            diagnostic = code_extraction_diagnostic(completion_text or "")[:1200]
            logger.warning("Code extraction diagnostic: %s", diagnostic)
            instruction = ("The previous response was rejected before execution. " + diagnostic
                + "\nRegenerate the complete solution from the original task and data contract. "
                  "Preserve all constraints, interfaces, and evaluation boundaries. Return one short plan "
                  "and exactly one complete runnable Python code block. Do not include analysis or examples. "
                  "No previous code was executed.")
            if isinstance(prompt, list):
                current_prompt = [
                    *prompt,
                    {"role": "user", "content": "# Required correction\n" + instruction},
                ]
            elif isinstance(prompt, dict) and isinstance(prompt.get("user"), str):
                current_prompt = {
                    **prompt,
                    "user": prompt["user"] + "\n\n# Required correction\n" + instruction,
                }
            else:
                current_prompt = str(prompt) + "\n\n# Required correction\n" + instruction

    raise RuntimeError("Code extraction failed after retries; original responses remain in the verbose log")
