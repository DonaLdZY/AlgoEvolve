from types import SimpleNamespace

from utils import data_preview


def test_task_description_cleanup_has_a_stable_usage_stage(monkeypatch):
    captured = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        return "Cleaned task description"

    monkeypatch.setattr("llm.query", fake_query)
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            code=SimpleNamespace(model="test-model"),
            generate_submission=False,
        )
    )

    cleaned = data_preview.clean_task_desc("Original task", cfg)

    assert cleaned == "Cleaned task description"
    assert captured["stage_id"] == "task_description_cleanup"
