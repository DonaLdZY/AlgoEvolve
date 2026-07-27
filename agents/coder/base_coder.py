"""Base code generation mode (single-shot plan + code).

The simplest generation strategy: one LLM call produces a natural language
plan followed by a complete code block. Used as the default / fallback mode
when diff or stepwise generation is not enabled or fails.
"""

from __future__ import annotations

import logging
from typing import Tuple

from llm import generate
from utils.response import extract_plan_and_code
from agents.prompts import plan_and_code_response_format

logger = logging.getLogger("AlgoEvolve")


# ============ Response format prompt (rewrite mode specific) ============

RESPONSE_FORMAT = {
    "Response format": plan_and_code_response_format(
        "the complete runnable solution, including the required validation metric output"
    )
}


def plan_and_code_query(
    agent_instance,
    prompt,
    retries: int = 3,
) -> Tuple[str, str]:
    """Generate plan + code in one LLM call; returns (nl_text, code). On failure returns ("", raw_completion_text)."""
    retry_cfg = getattr(agent_instance.acfg, "retries", None)
    retries = max(
        1,
        int(getattr(retry_cfg, "code_generation_extract_max_attempts", retries)),
    )
    completion_text = None
    for _ in range(retries):
        completion_text = generate(
            prompt=prompt,
            temperature=agent_instance.acfg.code.temp,
            cfg=agent_instance.cfg,
        )
        nl_text, code = extract_plan_and_code(
            completion_text,
            default_plan="Implement the requested complete solution and report its validation metric.",
        )

        if code:
            if completion_text.lstrip().startswith("```"):
                logger.info("Accepted a valid code-first response without regenerating it.")
            return nl_text, code

        logger.debug("Extraction retry...")

    logger.warning("Code extraction failed after retries")
    return "", completion_text  # type: ignore
