from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "autorealize_context.py"
_SPEC = importlib.util.spec_from_file_location("mlevolve_autorealize_context_for_test", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_CTX = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CTX)


def test_short_autorealize_stub_is_not_prompt_ready(tmp_path: Path) -> None:
    (tmp_path / "autorealize_context.md").write_text(
        "## AutoRealize Structured Context\n\n- problem_paradigm: unknown_but_executable\n",
        encoding="utf-8",
    )

    assert _CTX.has_autorealize_context(tmp_path) is False
    assert _CTX.build_autorealize_context_md(tmp_path, write_context_file=False) == ""


def test_autorealize_markdown_context_is_used_directly(tmp_path: Path) -> None:
    md = "\n".join(
        [
            "# AutoRealize Context For AutoML",
            "",
            "## Exact Source Schema Contract",
            "- Use exact physical source columns.",
            "",
            "## Evaluation Contract Reference",
            "- final_validation_score: use one scalar score.",
            "",
            "## Output Contract Reference",
            "- output_filename: submission.csv",
            "",
            "## Supplemental Data Facts",
            "- orders.xlsx contains the authoritative order table.",
            "",
            "Extra details. " * 120,
        ]
    )
    (tmp_path / "automl_context.md").write_text(md, encoding="utf-8")

    assert _CTX.has_autorealize_context(tmp_path) is True
    text = _CTX.build_autorealize_context_md(tmp_path, write_context_file=False)
    assert text.startswith("## AutoRealize Structured Context")
    assert "# AutoRealize Context For AutoML" in text
    assert "Exact Source Schema Contract" in text


def test_complete_automl_context_pack_is_prompt_ready(tmp_path: Path) -> None:
    pack = {
        "problem_paradigm": "static_optimization",
        "task_goal": "Minimize transport cost.",
        "evaluation_contract": {
            "primary_metric": "total_cost",
            "metric_direction": "minimize",
            "final_score_formula": "total_cost + hard_violation_penalty",
        },
        "output_contract": {
            "output_kind": "solution_table",
            "output_filename": "submission.csv",
            "columns": ["order_id", "vehicle_id"],
        },
        "method_strategy": {
            "explicit_rl_requested": True,
            "rl_as_required_paradigm": False,
            "recommended_solver_families": ["greedy_baseline", "rl_candidate"],
            "first_draft_policy": "Build deterministic baseline first.",
            "rl_branch_policy": "Compare RL later with the same evaluator.",
        },
        "data_access": [
            {
                "path": "orders.xlsx",
                "kind": "excel",
                "read_method": "pandas.read_excel",
            }
        ],
        "data_schema_contract": {
            "rules": ["Use exact physical column names."],
            "workbooks": [
                {
                    "source_file": "orders.xlsx",
                    "valid_sheet_names_exact": ["订单表信息"],
                }
            ],
            "tables": [
                {
                    "table_id": "orders.xlsx::订单表信息",
                    "source_file": "orders.xlsx",
                    "sheet_name": "订单表信息",
                    "table_kind": "excel_sheet",
                    "column_count": 2,
                    "physical_columns_exact": ["订单号", "要求交付时间"],
                    "field_summaries": [
                        {"name": "订单号", "meaning": "order id", "logical_type": "text"},
                    ],
                }
            ],
        },
    }
    (tmp_path / "automl_context_pack.json").write_text(json.dumps(pack), encoding="utf-8")

    assert _CTX.has_autorealize_context(tmp_path) is True
    text = _CTX.build_autorealize_context_md(tmp_path, write_context_file=False)
    assert "AutoRealize Structured Context" in text
    assert "Exact Source Schema Contract" in text
    assert "`订单号`" in text
    assert "valid_sheet_names_exact" in text
    assert "Method Strategy" in text
    assert "explicit_rl_requested" in text
    assert "Build deterministic baseline first" in text
    assert "Data Access" in text
    assert "final_validation_score" in text


def test_pack_renders_executable_read_recipes_as_advisory_context(tmp_path: Path) -> None:
    pack = {
        "problem_paradigm": "static_optimization",
        "task_goal": "Minimize transport cost.",
        "evaluation_contract": {"primary_metric": "cost", "metric_direction": "minimize"},
        "output_contract": {"output_kind": "solution_table", "columns": ["id"]},
        "data_access": [
            {
                "path": "input_items.csv",
                "read_method": "pandas.read_csv",
                "read_contract": {
                    "schema_version": "autorealize.csv_read_contract.v1",
                    "verified": True,
                    "path": "input_items.csv",
                    "reader": "pandas.read_csv",
                    "pandas_kwargs": {"sep": ";", "decimal": ",", "encoding": "utf-8-sig"},
                    "validated_shape": [203, 19],
                    "validated_columns_exact": ["Item ident", "Weight"],
                },
            }
        ],
    }
    (tmp_path / "automl_context_pack.json").write_text(json.dumps(pack), encoding="utf-8")

    text = _CTX.build_autorealize_context_md(tmp_path, write_context_file=False)
    fast_context = _CTX.select_autorealize_context_for_stage(text, "fast_draft")

    assert "Executable Data Read Recipes" in text
    assert '"decimal": ","' in text
    assert "not an immutable host-level gate" in text
    assert "Executable Data Read Recipes" in fast_context


def test_stage_context_routes_complete_sections_without_character_truncation() -> None:
    text = """## AutoRealize Structured Context
Stable introduction.

## Priority Rules
- authoritative

## Exact Source Schema Contract
- physical_columns_exact: `exact_field`

## Minimal Task Reference
- problem_paradigm: `static_optimization`

## Method Strategy
- use task-appropriate methods

## Evaluation Contract Reference
- metric_direction: minimize

## Output Contract Reference
- output_filename: submission.csv

## Supplemental Data Facts
- very verbose table profile

## Constraints Reference
- obey task constraints

## Leakage Guards
- no leakage

## Pitfalls
- avoid guessed fields
"""

    data_context = _CTX.select_autorealize_context_for_stage(
        text, "data_processing_and_feature_engineering"
    )
    model_context = _CTX.select_autorealize_context_for_stage(text, "model_design")
    training_context = _CTX.select_autorealize_context_for_stage(text, "training_evaluation")
    evaluator_context = _CTX.select_autorealize_context_for_stage(
        text, "evaluator_and_constraint_checker"
    )

    assert "very verbose table profile" in data_context
    assert "very verbose table profile" not in model_context
    assert "very verbose table profile" not in training_context
    assert "physical_columns_exact: `exact_field`" in evaluator_context
    assert "metric_direction: minimize" in model_context
    assert "output_filename: submission.csv" in training_context
    assert _CTX.select_autorealize_context_for_stage("plain preview", "model_design") == "plain preview"
