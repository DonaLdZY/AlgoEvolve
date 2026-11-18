"""Prompt helpers for provider-side input-cache friendly layouts.

The main rule is simple: large stable task context should appear once and
before node-specific memory/code/output. Dynamic search state belongs later.
"""

from __future__ import annotations

import hashlib
from typing import Any


DATA_CONTEXT_REFERENCE = (
    "The stable task and data context is already provided in the Task description "
    "section above. Do not expect it to be repeated here."
)
TASK_CONTEXT_END_MARKER = "# End stable task/data context"
CANDIDATE_CODE_END_MARKER = "# End candidate implementation"


def _norm(text: Any) -> str:
    return str(text or "").strip()


def looks_same_context(task_desc: Any, data_preview: Any) -> bool:
    """Return True when data_preview is just a duplicate of task_desc."""
    task = _norm(task_desc)
    preview = _norm(data_preview)
    if not task or not preview:
        return False
    return task == preview


def stable_data_context(task_desc: Any, data_preview: Any) -> str:
    """Return the one large context block that should be placed early."""
    task = _norm(task_desc)
    preview = _norm(data_preview)
    if task and preview and not looks_same_context(task, preview):
        return f"{task}\n\n# Supplementary Data Preview\n{preview}"
    return task or preview


def dynamic_data_reference(task_desc: Any, data_preview: Any) -> str:
    """Return a short late-prompt reference instead of repeating large context."""
    preview = _norm(data_preview)
    task = _norm(task_desc)
    if not preview:
        return DATA_CONTEXT_REFERENCE
    if task:
        return DATA_CONTEXT_REFERENCE
    return preview


def task_section(task_desc: Any, data_preview: Any = "") -> str:
    context = stable_data_context(task_desc, data_preview)
    return f"\n# Task description\n{context}\n\n{TASK_CONTEXT_END_MARKER}\n"


def candidate_review_section(task_desc: Any, data_preview: Any, code: str) -> str:
    """Identical task + exact implementation prefix for both review roles."""
    from utils.response import wrap_code

    wrapped = wrap_code(code)
    return (task_section(task_desc, data_preview) + "\n# Candidate implementation\n"
            + f"<!-- candidate-code-chars: {len(wrapped)} -->\n" + wrapped
            + "\n\n" + CANDIDATE_CODE_END_MARKER + "\n\n")


def dataset_reference_sentence(task_desc: Any, data_preview: Any = "") -> str:
    ref = dynamic_data_reference(task_desc, data_preview)
    if ref == DATA_CONTEXT_REFERENCE:
        return f"First, I'll use the stable task/data context already provided above."
    return f"First, I'll examine the dataset:\n{ref}"


def context_fingerprint(text: Any) -> str:
    value = _norm(text)
    if not value:
        return "empty"
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:12]


def routed_data_context(agent: Any, stage: str) -> str:
    """Route whole sections while keeping the task contract lossless.

    Standalone previews and unknown stages deliberately keep their full input.
    Never slice a schema or a task requirement to meet a character budget.
    """
    from utils.autorealize_context import select_autorealize_context_for_stage

    aliases = {"draft": "fast_draft", "debug": "code_review", "improve": "merge",
               "fusion": "merge", "evolution": "merge", "result_parse": "result_parsing"}
    return select_autorealize_context_for_stage(
        _norm(getattr(agent, "autorealize_context", "") or getattr(agent, "data_preview", "")),
        aliases.get(stage, stage),
    )
