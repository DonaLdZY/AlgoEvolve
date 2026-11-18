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


@pytest.mark.parametrize("use_memory", [False, True])
@pytest.mark.parametrize("optimization_rl", [False, True])
def test_failed_review_null_metric_passes_provider_validation(use_memory, optimization_rl):
    """A crash verdict must survive JSON validation without another paid request."""
    import json
    import jsonschema
    from llm.openai import _validated_json_content, _build_json_mode_messages

    response = {
        "verdict": "reject", "is_bug": True, "summary": "Prediction parity check failed.",
        "reason_codes": ["runtime_exception"], "debug_hint": "Use identical feature windows.",
        "technical_summary": "Execution raised RuntimeError before producing a score.",
        "human_insight": "The candidate needs repair before it can be compared.",
        "confidence": 1.0, "metric": None, "lower_is_better": True,
        "output_scope": "unknown",
    }
    if use_memory:
        response["code_summary"] = "A regression model with trailing history features."
    spec = result_parse_agent.get_review_func_spec(use_memory, optimization_rl)
    messages = _build_json_mode_messages('Review the execution.', 'RuntimeError: parity', spec)
    assert '"metric":null' in '\n'.join(m['content'] for m in messages)
    assert _validated_json_content(json.dumps(response), spec, "stop") == response
    # Nullable does not allow prose or a fabricated numeric fallback as a metric.
    with pytest.raises(jsonschema.ValidationError):
        _validated_json_content(json.dumps({**response, "metric": "not available"}), spec, "stop")


@pytest.mark.parametrize("scope,context,artifact,phase,accepted", [
    ("development_validation_only", True, True, "search", True),
    ("provided_prediction_population", True, True, "search", False),
    ("unknown", True, True, "search", False),
    ("development_validation_only", False, True, "search", False),
    ("development_validation_only", True, False, "search", False),
    ("development_validation_only", True, True, "final", False),
])
def test_development_score_does_not_claim_missing_new_output_delivery(monkeypatch, tmp_path, scope, context, artifact, phase, accepted):
    from engine.execution import update_node_certification

    monkeypatch.setattr(result_parse_agent, "find_model_artifacts", lambda *args: [tmp_path / "model.pkl"] if artifact else [])
    agent = SimpleNamespace(cfg=SimpleNamespace(workspace_dir=tmp_path),
                            acfg=SimpleNamespace(generate_submission=True, use_global_memory=False, output_language="english"),
                            metric_maximize=True, has_autorealize_context=context, global_memory=None,
                            task_desc="Classify equipment state", data_preview="", coldstart_description="")
    node = SearchNode(code="def predict(model_path, data): return data", stage="draft", _term_out=[],
                      evaluation_protocol={
                          "evaluation_phase": phase,
                          "scoring_split": "development",
                          "final_evaluation_accessed": False,
                          "protocol_evidence_status": "read",
                      })
    response = {"verdict": "accept", "is_bug": False, "metric": 0.7, "lower_is_better": False,
                "summary": "Development probabilities were evaluated.", "reason_codes": ["accepted"],
                "output_scope": scope, "confidence": 0.9}
    result_parse_agent._apply_review_response(agent, node, response)
    update_node_certification(agent, node)
    assert node.search_eligible is accepted
    assert node.delivery_ready is False


@pytest.mark.parametrize("csv_text,expected_scope", [
    ("Id,prediction\n", "development_validation_only"),
    ("Id,prediction\n1,2.1\n2,3.2\n", "provided_prediction_population"),
])
def test_populated_submission_scope_still_runs_normal_validation(monkeypatch, tmp_path, csv_text, expected_scope):
    agent = SimpleNamespace(cfg=SimpleNamespace(workspace_dir=tmp_path),
                            acfg=SimpleNamespace(generate_submission=True, use_global_memory=False, output_language="english"),
                            metric_maximize=False, has_autorealize_context=True, global_memory=None,
                            task_desc="Regression", data_preview="", coldstart_description="")
    node = SearchNode(code="def predict(model_path, data): return data", stage="draft", _term_out=[],
                      evaluation_protocol={"evaluation_phase": "search", "final_evaluation_accessed": False})
    path = tmp_path / "submission" / f"submission_{node.id}.csv"
    path.parent.mkdir()
    path.write_text(csv_text, encoding="utf-8")
    monkeypatch.setattr(result_parse_agent, "find_model_artifacts", lambda *a: [tmp_path / "model.pkl"])
    validations = []
    monkeypatch.setattr(result_parse_agent, "_validate_format_with_retry", lambda *a: validations.append(True))
    response = {"verdict": "accept", "is_bug": False, "metric": .12, "lower_is_better": True,
                "summary": "Development score is correct.", "reason_codes": ["accepted"],
                "output_scope": "development_validation_only", "confidence": .9}
    result_parse_agent._apply_review_response(agent, node, response)
    assert node.output_scope == expected_scope
    assert len(validations) == (1 if expected_scope == "provided_prediction_population" else 0)


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


@pytest.mark.parametrize("response", [{"lower_is_better": "false"}, {}])
def test_metric_direction_never_defaults_to_maximize_on_invalid_response(monkeypatch, response):
    monkeypatch.setattr(result_parse_agent, "query", lambda **kwargs: response)
    agent = SimpleNamespace(
        task_desc="Use the stated task metric", autorealize_context="", cfg=SimpleNamespace(),
        acfg=SimpleNamespace(retries=SimpleNamespace(metric_direction_max_attempts=1),
                             feedback=SimpleNamespace(model="test", temp=0)),
    )
    with pytest.raises(RuntimeError, match="Metric direction is unresolved"):
        result_parse_agent.determine_metric_direction(agent)
    assert not hasattr(agent, "metric_maximize")


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


def test_standalone_result_review_includes_task_description(monkeypatch, tmp_path) -> None:
    agent = _parse_agent(tmp_path, "")
    agent.autorealize_context = ""
    agent.data_preview = ""
    agent.task_desc = "Choose a feasible production schedule with minimum total cost."
    calls: list[dict] = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        return {
            "is_bug": False,
            "summary": "The task evaluator produced a comparable score.",
            "metric": 12.0,
            "lower_is_better": True,
        }

    monkeypatch.setattr(result_parse_agent, "query", fake_query)
    node = SearchNode(
        code="print('Final Validation Score: 12')",
        stage="draft",
        _term_out=[],
    )

    result_parse_agent.run(
        agent,
        node,
        ExecutionResult(term_out=["Final Validation Score: 12\n"], exec_time=1, exc_type=None),
    )

    assert len(calls) == 1
    assert agent.task_desc in calls[0]["user_message"]
    assert "# Task description" in calls[0]["user_message"]


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


@pytest.mark.parametrize("exception_type", ["KeyError", "TimeoutError", "RuntimeError", "PermissionError"])
def test_executor_failure_skips_paid_review_and_preserves_debug_evidence(monkeypatch, tmp_path, exception_type) -> None:
    contract = "## AutoRealize Structured Context\n## Evaluation Contract Reference\n- metric_direction: minimize\n"
    agent = _parse_agent(tmp_path, contract)
    def fake_query(**kwargs):
        pytest.fail("An executor-confirmed failure must not call the result reviewer")

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
        exc_type=exception_type,
        exc_info={"message": "missing"},
    )

    parsed = result_parse_agent.run(agent, node, exec_result)

    assert parsed is node
    assert node.is_buggy is True
    assert node.is_valid is False
    assert node.search_eligible is False
    assert node.delivery_ready is False
    assert node.metric.value is None
    assert node.review_verdict == "reject"
    assert node.review_reason_codes == ["runtime_exception"]
    assert exception_type in node.analysis_for_prompt
    assert "missing" in node.analysis_for_prompt
    assert "Traceback ... KeyError: missing" in node.term_out
    assert node.exec_time == 0.1
    assert node.llm_insight


def test_printed_score_cannot_make_crashed_run_comparable(monkeypatch, tmp_path):
    agent = _parse_agent(tmp_path, "- metric_direction: minimize")
    monkeypatch.setattr(result_parse_agent, "query", lambda **_: pytest.fail("No paid review for a crash"))
    node = SearchNode(code="raise ValueError()", stage="draft", _term_out=[])
    result_parse_agent.run(agent, node, ExecutionResult(
        term_out=["Final Validation Score: 0.01\n"], exec_time=1, exc_type="ValueError",
    ))
    assert node.metric.value is None
    assert not node.search_eligible


def test_candidate_failure_file_without_score_skips_paid_review(monkeypatch, tmp_path):
    agent = _parse_agent(tmp_path, "- metric_direction: minimize")
    calls = []
    monkeypatch.setattr(result_parse_agent, "candidate_failure_evidence", lambda *_: {
        "path": "working/evaluation_failure.json", "content": {"status": "failed"},
    })
    def review(**kwargs):
        calls.append(kwargs)
        return {"is_bug": True, "summary": "Incomplete evaluation", "metric": None, "lower_is_better": True}
    monkeypatch.setattr(result_parse_agent, "query", review)
    node = SearchNode(code="print('evaluation incomplete')", stage="draft", _term_out=[])
    result_parse_agent.run(agent, node, ExecutionResult(term_out=["incomplete\n"], exec_time=1, exc_type=None))
    assert len(calls) == 0
    assert node.is_buggy is True
    assert node.search_eligible is False
    assert node.review_verdict == "reject"
    assert node.review_reason_codes == ["candidate_failure_artifact"]
    assert "evaluation_failure.json" in node.analysis_for_prompt


def test_missing_score_without_verifiable_result_skips_paid_review(monkeypatch, tmp_path):
    agent = _parse_agent(tmp_path, "- metric_direction: minimize")
    monkeypatch.setattr(
        result_parse_agent,
        "query",
        lambda **_: pytest.fail("An unscorable result must not call the reviewer"),
    )
    node = SearchNode(code="print('finished without result')", stage="draft", _term_out=[])
    result_parse_agent.run(
        agent,
        node,
        ExecutionResult(term_out=["finished without result\n"], exec_time=1, exc_type=None),
    )
    assert node.is_buggy is True
    assert node.review_reason_codes == ["metric_missing"]
    assert node.metric.value is None
    assert "no comparable" in node.analysis_for_prompt.lower()


def test_prediction_result_review_receives_complete_contract_and_final_score_boundary(monkeypatch, tmp_path):
    contract = "## AutoRealize Structured Context\nReserved evaluation uses the last 7008 rows. Only development_search may rank candidates.\n" + "Complete rule. " * 1000 + "\nLate exception: legal observed history crosses partitions."
    agent = _parse_agent(tmp_path, contract)
    agent.task_desc = "Forecast the next electricity usage observation."
    calls = []
    def fake_query(**kwargs):
        calls.append(kwargs)
        return {"is_bug": True, "summary": "Holdout score is invalid for search", "metric": None, "lower_is_better": True}
    monkeypatch.setattr(result_parse_agent, "query", fake_query)
    node = SearchNode(code="print('Final Validation Score: 3.9')", stage="draft", _term_out=[])
    result_parse_agent.run(agent, node, ExecutionResult(term_out=["Final Validation Score: 3.9\n"], exec_time=1, exc_type=None))
    assert len(calls) == 1
    assert contract in calls[0]["user_message"]
    assert "Trace the printed scalar to its prediction population" in calls[0]["user_message"]
    assert node.metric.value is None
