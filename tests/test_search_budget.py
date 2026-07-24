from engine.search_budget import resolve_search_budget


def test_completed_resume_appends_fresh_step_and_time_budget() -> None:
    assert resolve_search_budget(
        requested_steps=20,
        requested_time_limit=3600,
        restored_completed=50,
        restored_elapsed=10800.25,
        resume_run=True,
        resume_budget_mode="additional",
    ) == (70, 14401)


def test_interrupted_resume_keeps_configured_cumulative_targets() -> None:
    assert resolve_search_budget(
        requested_steps=50,
        requested_time_limit=10800,
        restored_completed=31,
        restored_elapsed=7200,
        resume_run=True,
        resume_budget_mode="total",
    ) == (50, 10800)


def test_interrupted_additional_session_preserves_its_effective_targets() -> None:
    assert resolve_search_budget(
        requested_steps=20,
        requested_time_limit=3600,
        restored_completed=60,
        restored_elapsed=12000,
        resume_run=True,
        resume_budget_mode="total",
        preserved_target_steps=70,
        preserved_time_limit=14401,
    ) == (70, 14401)


def test_zero_time_limit_remains_unlimited_when_appending() -> None:
    assert resolve_search_budget(
        requested_steps=5,
        requested_time_limit=0,
        restored_completed=50,
        restored_elapsed=10800,
        resume_run=True,
        resume_budget_mode="additional",
    ) == (55, 0)
