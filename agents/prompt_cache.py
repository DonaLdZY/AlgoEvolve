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


def _norm(text: Any) -> str:
    return str(text or "").strip()


def looks_same_context(task_desc: Any, data_preview: Any) -> bool:
    """Return True when data_preview is just a duplicate of task_desc."""
    task = _norm(task_desc)
    preview = _norm(data_preview)
    if not task or not preview:
        return False
    if task == preview:
        return True
    # AutoRealize context may be regenerated with tiny trailing differences; use
    # a conservative prefix/length check so we only suppress obvious duplicates.
    if abs(len(task) - len(preview)) <= 128 and task[:4096] == preview[:4096]:
        return True
    return False


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
    """Return the complete AutoRealize context for every search stage."""

    del stage
    return _norm(getattr(agent, "data_preview", ""))
