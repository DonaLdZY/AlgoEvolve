from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agents.memory.optimization_experience import (
    build_optimization_experience_for_agent,
    load_optimization_experience_cards,
    render_optimization_experience_context,
    retrieve_optimization_experiences,
)
from agents.coder.stepwise_coder import StepwiseContext, create_default_step_agents
from agents.result_parse_agent import (
    _decision_signals_for_node,
    _extract_optimization_solver_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def _agent(task_desc: str, data_preview: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        task_desc=task_desc,
        data_preview=data_preview,
        coldstart_description="",
        acfg=SimpleNamespace(
            use_optimization_experience_library=True,
            optimization_experience_library_path="",
            optimization_experience_max_cards=2,
            optimization_experience_min_score=3.0,
            optimization_experience_max_chars=6000,
        ),
    )


def test_bounded_state_assignment_retrieves_exact_lns_trajectory() -> None:
    cards = load_optimization_experience_cards()
    query = (
        "Assign discrete entities to days. Each day has an integer occupancy between 125 and 300. "
        "The nonlinear accounting penalty depends on the occupancy of adjacent days."
    )

    matches = retrieve_optimization_experiences(query, cards=cards, min_score=3.0)

    assert matches
    assert matches[0].card.experience_id == "bounded_aggregate_state_milp_lns"
    assert {
        "discrete_decisions",
        "bounded_aggregate_state",
        "local_state_transition_cost",
        "nonlinear_aggregate_cost",
    }.issubset(set(matches[0].matched_signals))
    rendered = render_optimization_experience_context(cards, matches)
    assert "lower bound" in rendered
    assert "upper bound" in rendered
    assert "exact large neighborhoods" in rendered
    assert "not mandatory algorithms" in rendered


def test_unmatched_continuous_task_receives_only_compact_index() -> None:
    agent = _agent("Tune continuous process temperature with Bayesian optimization.")

    context = build_optimization_experience_for_agent(agent, task_mode="optimization")

    assert "Optimization Method Experience Index" in context
    assert "Retrieved Optimization Experience" not in context
    assert len(context) < 1200


def test_experience_library_is_generic_not_task_specific() -> None:
    text = (ROOT / "agents/memory/optimization_experiences.yaml").read_text(encoding="utf-8")

    assert "Santa" not in text
    assert "family_id" not in text
    assert "assigned_day" not in text
    assert "Gurobi" not in text


def test_solver_summary_becomes_compact_node_facts() -> None:
    output = (
        'Optimization Solver Summary: {"solver_family":"MILP","solve_status":"feasible",'
        '"incumbent":70134,"best_bound":69500,"relative_gap":0.009,'
        '"warm_start_used":true,"variable_count":12000,"unknown_blob":"omit"}\n'
        "Final Validation Score: 70134\n"
    )

    summary = _extract_optimization_solver_summary(output)
    signals = _decision_signals_for_node(None, 70134.0, summary)

    assert summary == {
        "solver_family": "MILP",
        "solve_status": "feasible",
        "incumbent": 70134,
        "best_bound": 69500,
        "relative_gap": 0.009,
        "variable_count": 12000,
        "warm_start_used": True,
    }
    assert signals == {
        "final_score": 70134.0,
        "optimization_solver": summary,
    }


def test_optimization_stepwise_prompt_contains_conditional_reformulation_guidance() -> None:
    stepwise = (ROOT / "agents/coder/stepwise_coder.py").read_text(encoding="utf-8")
    shared = (ROOT / "agents/prompts/shared.py").read_text(encoding="utf-8")
    review = (ROOT / "agents/prompts/validation_template_prompts.py").read_text(encoding="utf-8")

    assert "Optimization Structure Assessment" in stepwise
    assert "Optimization Solver Summary" in stepwise
    assert "exact large-neighborhood search" in stepwise
    assert "commercial solver" in stepwise
    assert "small bounded integer aggregate state" in shared
    assert "Never equate an incumbent with a proof" in shared
    assert "unconditional optimality claims" in review
    assert "relaxation objective alone is not a final solution score" in review


def test_full_experience_card_is_routed_only_to_solver_design_step() -> None:
    steps = create_default_step_agents("optimization")
    fake_agent = SimpleNamespace(
        use_coldstart=False,
        coldstart_description="",
        acfg=SimpleNamespace(
            generate_submission=False,
            draft=SimpleNamespace(stepwise_stage_context=False),
            code=SimpleNamespace(model="deepseek-v4-pro"),
        ),
    )
    context = StepwiseContext(
        task_mode="optimization",
        optimization_experience="EXACT EXPERIENCE CARD",
    )
    prompt_base = {
        "Introduction": "Optimization system",
        "Task description": "Optimize a bounded assignment.",
        "Instructions": {"Implementation guideline": []},
        "Memory": "",
    }

    prompts = {
        step.name: step._build_prompt(
            task_desc=prompt_base["Task description"],
            data_preview_str="schema",
            previous_steps=[],
            prompt_base=prompt_base,
            agent_instance=fake_agent,
            context=context,
        )
        for step in steps
    }

    assert "EXACT EXPERIENCE CARD" in prompts["decision_method"]["user"]
    assert "EXACT EXPERIENCE CARD" not in prompts["problem_and_evaluator"]["user"]
    assert "EXACT EXPERIENCE CARD" not in prompts["solve_rollout_and_artifact"]["user"]
