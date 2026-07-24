from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("dataclasses_json")
pytest.importorskip("humanize")
pytest.importorskip("coolname")
pytest.importorskip("omegaconf")
pytest.importorskip("llm")

from agents import result_parse_agent
from engine.executor import ExecutionResult
from engine.search_node import SearchNode
from utils.metric import MetricValue


def test_result_review_supplies_ui_insight_without_a_second_llm_call(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        result_parse_agent,
        "query",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("no second insight call")),
    )
    agent = SimpleNamespace(
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        acfg=SimpleNamespace(
            generate_submission=False,
            use_global_memory=False,
            check_data_leakage=False,
            output_language="chinese",
        ),
        metric_maximize=False,
        metric_maximize_reasoning="task contract",
        global_memory=None,
    )
    node = SearchNode(
        code="def solve(model_path, data): return data",
        plan="solve",
        stage="draft",
        _term_out=["Final Validation Score: 12\n"],
    )
    response = {
        "verdict": "accept",
        "is_bug": False,
        "summary": "结果可比较。",
        "reason_codes": ["accepted"],
        "debug_hint": "",
        "technical_summary": "共享评估器对返回方案计算了有限成本。",
        "human_insight": "该方案已通过结果可信度检查，当前成本为 12。",
        "confidence": 0.9,
        "metric": 12.0,
        "lower_is_better": True,
    }

    result_parse_agent._apply_review_response(agent, node, response)

    assert node.llm_insight == "该方案已通过结果可信度检查，当前成本为 12。"
    assert node.review_reason_codes == ["accepted"]


def test_metric_direction_uses_autorealize_contract_without_llm(monkeypatch) -> None:
    def fail_query(**kwargs):
        raise AssertionError("metric direction LLM should not be called")

    monkeypatch.setattr(result_parse_agent, "query", fail_query)
    agent = SimpleNamespace(
        autorealize_context=(
            "## AutoRealize Structured Context\n"
            "## Evaluation Contract Reference\n"
            "- metric_direction: minimize\n"
        ),
        task_desc="demo",
        acfg=SimpleNamespace(retries=SimpleNamespace()),
    )

    result_parse_agent.determine_metric_direction(agent)

    assert agent.metric_maximize is False
    assert "AutoRealize evaluation contract" in agent.metric_maximize_reasoning


def _parse_agent(tmp_path, contract: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_desc="decision optimization task",
        coldstart_description="",
        data_preview=contract,
        autorealize_context=contract,
        metric_maximize=False,
        metric_maximize_reasoning="contract says minimize",
        global_memory=None,
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        acfg=SimpleNamespace(
            generate_submission=False,
            use_global_memory=False,
            check_data_leakage=False,
            feedback=SimpleNamespace(model="fake-feedback", temp=0.0),
            retries=SimpleNamespace(result_parse_max_attempts=1),
        ),
    )


def test_final_score_always_routes_through_llm_reviewer(monkeypatch, tmp_path) -> None:
    contract = """## AutoRealize Structured Context
## Evaluation Contract Reference
- metric_direction: minimize
- Valid solutions are scored by score_solution; negative values are allowed.
"""
    agent = _parse_agent(tmp_path, contract)
    calls: list[dict] = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        return {
            "is_bug": False,
            "summary": "The returned solution is valid and the task evaluator produced a comparable score.",
            "metric": -2.5,
            "lower_is_better": True,
        }

    monkeypatch.setattr(result_parse_agent, "query", fake_query)
    node = SearchNode(
        parent=None,
        plan="solve",
        code="solution = solve(data)\nscore = score_solution(solution)\nprint(score)",
        stage="draft",
        _term_out=[],
    )
    exec_result = ExecutionResult(
        term_out=[
            'Decision Validation Summary: {"feasible": true, '
            '"final_score_source": "score_solution", "total_score": -2.5}\n'
            'Final Validation Score: -2.5\n'
        ],
        exec_time=1.25,
        exc_type=None,
    )

    parsed = result_parse_agent.run(agent, node, exec_result)

    assert parsed is node
    assert len(calls) == 1
    assert calls[0]["func_spec"].name == "submit_review"
    assert "evidence only, not a verdict" in calls[0]["user_message"]
    assert '"candidate_reported_final_score": -2.5' in calls[0]["user_message"]
    assert node.is_buggy is False
    assert node.metric.value == -2.5


def test_llm_accepted_decision_score_is_valid_without_submission(monkeypatch, tmp_path) -> None:
    contract = """## AutoRealize Structured Context
## Evaluation Contract Reference
- metric_direction: minimize
- The task evaluator returns one comparable scalar score.
"""
    agent = _parse_agent(tmp_path, contract)
    agent.acfg.generate_submission = True

    monkeypatch.setattr(
        result_parse_agent,
        "query",
        lambda **_: {
            "is_bug": False,
            "summary": "The returned decision is valid and its score is comparable.",
            "metric": 619406.0,
            "lower_is_better": True,
        },
    )
    node = SearchNode(
        parent=None,
        plan="solve",
        code="print('Final Validation Score: 619406.0')",
        stage="debug",
        _term_out=[],
    )
    exec_result = ExecutionResult(
        term_out=["Final Validation Score: 619406.0\n"],
        exec_time=1.0,
        exc_type=None,
    )

    parsed = result_parse_agent.run(agent, node, exec_result)

    assert parsed.is_buggy is False
    assert parsed.is_valid is True
    assert parsed.metric.value == 619406.0
    assert "separate delivery follow-up" in parsed.analysis


def test_llm_reviewer_can_reject_candidate_reported_score(monkeypatch, tmp_path) -> None:
    contract = """## AutoRealize Structured Context
## Evaluation Contract Reference
- metric_direction: minimize
- Hard constraints must pass. A constraint violation makes the submission invalid and not scored.
"""
    agent = _parse_agent(tmp_path, contract)
    calls: list[dict] = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        return {
            "is_bug": True,
            "summary": "The program scored a contract-invalid solution, so its printed number is not comparable.",
            "metric": None,
            "lower_is_better": True,
        }

    monkeypatch.setattr(result_parse_agent, "query", fake_query)
    node = SearchNode(
        parent=None,
        plan="solve",
        code="score = -100.0\nprint(score)",
        stage="draft",
        _term_out=[],
    )
    exec_result = ExecutionResult(
        term_out=[
            'Decision Validation Summary: {"feasible": false, "violations": ["capacity"], '
            '"total_penalty": -100.0}\nFinal Validation Score: -100.0\n'
        ],
        exec_time=0.5,
        exc_type=None,
    )

    parsed = result_parse_agent.run(agent, node, exec_result)

    assert parsed is node
    assert len(calls) == 1
    assert node.is_buggy is True
    assert node.metric.value is None
    assert "contract-invalid" in node.analysis
    assert node.decision_signals["reported_feasible"] is False


def test_execution_exception_still_routes_through_llm_reviewer(monkeypatch, tmp_path) -> None:
    contract = "## AutoRealize Structured Context\n## Evaluation Contract Reference\n- metric_direction: minimize\n"
    agent = _parse_agent(tmp_path, contract)
    calls: list[dict] = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        return {
            "is_bug": True,
            "summary": "Execution raised KeyError before producing a comparable result.",
            "metric": None,
            "lower_is_better": True,
        }

    monkeypatch.setattr(result_parse_agent, "query", fake_query)
    node = SearchNode(
        parent=None,
        plan="solve",
        code="raise KeyError('missing')",
        stage="draft",
        _term_out=[],
    )
    exec_result = ExecutionResult(
        term_out=["Traceback ... KeyError: missing\n"],
        exec_time=0.1,
        exc_type="KeyError",
        exc_info={"message": "missing"},
    )

    parsed = result_parse_agent.run(agent, node, exec_result)

    assert parsed is node
    assert len(calls) == 1
    assert '"exception_type": "KeyError"' in calls[0]["user_message"]
    assert node.is_buggy is True
    assert node.metric.value is None
