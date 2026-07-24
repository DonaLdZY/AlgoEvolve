"""Search-budget resolution for fresh runs and checkpoint resumes."""

from __future__ import annotations

import math


def resolve_search_budget(
    *,
    requested_steps: int,
    requested_time_limit: int,
    restored_completed: int,
    restored_elapsed: float,
    resume_run: bool,
    resume_budget_mode: str,
    preserved_target_steps: int = 0,
    preserved_time_limit: int = 0,
) -> tuple[int, int]:
    """Return cumulative step/time targets understood by the search loop."""

    steps = max(0, int(requested_steps))
    time_limit = max(0, int(requested_time_limit))
    if not resume_run:
        return steps, time_limit

    if str(resume_budget_mode).strip().lower() != "additional":
        return (
            max(steps, max(0, int(preserved_target_steps))),
            max(time_limit, max(0, int(preserved_time_limit))),
        )

    steps += max(0, int(restored_completed))
    if time_limit > 0:
        time_limit = int(math.ceil(max(0.0, float(restored_elapsed)) + time_limit))
    return steps, time_limit
