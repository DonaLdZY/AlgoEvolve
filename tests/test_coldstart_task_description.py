from types import SimpleNamespace

from engine.coldstart import knowledge


def test_inline_goal_and_evaluation_are_normalized_for_cold_start(monkeypatch):
    captured = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        return {"category": "Others", "reasoning": "Tabular prediction."}

    monkeypatch.setattr("llm.query", fake_query)
    cfg = SimpleNamespace(
        exp_id="inline-task",
        agent=SimpleNamespace(
            code=SimpleNamespace(model="test-model"),
            feedback=SimpleNamespace(model="test-model", temp=0.0),
        ),
    )
    task_desc = {
        "Task goal": "Train a binary classifier from the supplied CSV.",
        "Task evaluation": "Report accuracy on a deterministic split.",
    }

    category = knowledge._classify_task_with_llm(cfg, task_desc)

    assert category == "Others"
    assert "# Task goal\nTrain a binary classifier" in captured["user_message"]
    assert "# Task evaluation\nReport accuracy" in captured["user_message"]
    assert captured["stage_id"] == "cold_start_classification"


def test_empty_inline_task_description_skips_the_classifier(monkeypatch):
    def fail_if_called(**_kwargs):
        raise AssertionError("empty task descriptions must not call the provider")

    monkeypatch.setattr("llm.query", fail_if_called)
    cfg = SimpleNamespace(agent=SimpleNamespace(code=SimpleNamespace(model="test-model")))

    assert knowledge._classify_task_with_llm(cfg, {"Task goal": "  "}) is None
