from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.search_budget import resolve_search_budget, search_termination_reason
from service_api import _is_search_budget_exhausted


def _scheduler(*, total: int, completed: int = 0, restored: int = 0, workers: int = 2, expose_after_first: bool = False):
    # Execute the production scheduling closure without model/runtime startup.
    source = ast.parse((Path(__file__).resolve().parents[1] / "run.py").read_text(encoding="utf-8"))
    fill = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "fill_worker_slots")
    factory = ast.parse("""
def build_scheduler(completed, submitted_actions):
    def count():
        return submitted_actions
    def commit(number):
        nonlocal completed
        completed += number
    return fill_worker_slots, count, commit
""")
    factory.body[0].body.insert(-1, fill)
    futures = {object() for _ in range(restored)}
    submissions = []
    agent = SimpleNamespace(search_integrity_failure=None)

    def schedule(_executor):
        future = object()
        submissions.append(future)
        if expose_after_first:
            agent.search_integrity_failure = "search_evaluation_exposure"
        return future, SimpleNamespace(action_id=str(len(submissions)), parent_node_id="root"), None, str(len(submissions))

    scope = {"agent": agent, "futures": futures, "future_metadata": {}, "max_workers": workers,
        "total_steps": total, "action_submission_budget": max(0, total - completed),
        "timed_out": False, "schedule_new_step": schedule, "executor": None,
        "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(factory), "scheduler-under-test", "exec"), scope)
    fill, count, commit = scope["build_scheduler"](completed, restored)
    return fill, count, commit, futures, submissions


def test_exposure_during_admission_blocks_additional_workers_and_refill():
    fill, count, _commit, futures, submissions = _scheduler(total=1000, expose_after_first=True)
    assert fill() == 1
    futures.clear()
    assert fill() == 0
    assert count() == len(submissions) == 1


def test_repeated_failed_generation_stops_even_without_time_limit():
    fill, count, _commit, futures, submissions = _scheduler(total=3)
    assert fill() == 2
    futures.clear()  # Both expensive generations failed without journal nodes.
    assert fill() == 1
    futures.clear()
    assert fill() == 0
    assert count() == len(submissions) == 3
    assert search_termination_reason(timed_out=False, completed_steps=0, total_steps=3, submitted_actions=count(), action_budget=3) == "action_budget_exhausted"


def test_restored_and_new_actions_share_remaining_budget():
    fill, count, _commit, futures, submissions = _scheduler(total=7, completed=4, restored=2, workers=8)
    assert fill() == 1
    futures.clear()
    assert fill() == 0
    assert count() == 3 and len(submissions) == 1


def test_failures_do_not_expand_the_budget_after_successful_nodes():
    fill, count, commit, futures, _submissions = _scheduler(total=3)
    assert fill() == 2
    futures.clear()
    commit(1)  # One node succeeds and one fails.
    assert fill() == 1
    futures.clear()
    commit(1)
    assert fill() == 0
    assert count() == 3
    assert search_termination_reason(timed_out=False, completed_steps=2, total_steps=3, submitted_actions=count(), action_budget=3) == "action_budget_exhausted"


@pytest.mark.parametrize("timed_out,completed,submitted,expected", [
    (True, 1, 3, "time_limit_exhausted"),
    (False, 3, 3, "steps_completed"),
    (False, 1, 3, "action_budget_exhausted"),
    (False, 1, 1, "search_exhausted"),
])
def test_terminal_reason_keeps_attempts_distinct_from_completed_nodes(timed_out, completed, submitted, expected):
    assert search_termination_reason(timed_out=timed_out, completed_steps=completed, total_steps=3, submitted_actions=submitted, action_budget=3) == expected


def test_action_exhaustion_is_recognized_and_continuation_adds_fresh_budget(tmp_path: Path):
    (tmp_path / "run_status.json").write_text(json.dumps({"status": "completed", "termination_reason": "action_budget_exhausted", "completed_steps": 1, "total_steps": 4, "submitted_actions": 4}), encoding="utf-8")
    assert _is_search_budget_exhausted(tmp_path)
    target, time_limit = resolve_search_budget(requested_steps=4, requested_time_limit=0, restored_completed=1, restored_elapsed=300, resume_run=True, resume_budget_mode="additional")
    assert target == 5 and time_limit == 0
    fill, count, _commit, _futures, _submissions = _scheduler(total=target, completed=1, workers=8)
    assert fill() == 4 and count() == 4
