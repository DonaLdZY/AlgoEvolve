"""AutoRealize context adapter for MLEvolve.

MLEvolve can still run standalone with only a data folder and description.md.
When AutoRealize artifacts are present, this module turns the structured task,
data-access, output and evaluation contracts into concise prompt context.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("MLEvolve")

CONTEXT_MARKER = "AutoRealize Structured Context"
MIN_PROMPT_READY_CONTEXT_CHARS = 900

_STAGE_CONTEXT_SECTIONS: dict[str, set[str] | None] = {
    "fast_draft": {
        "autorealize structured context",
        "priority rules",
        "exact source schema contract",
        "executable data read recipes",
        "source alias guard",
        "entity alias candidates",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "output contract reference",
        "relation cards",
        "problem boundary reference",
        "constraints reference",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "data access",
        "modeling boundary",
        "constraints",
    },
    # Data loading owns the full physical schema and all table-level caveats.
    "data_processing_and_feature_engineering": None,
    "decision_data_and_contract": None,
    "evaluator_and_constraint_checker": {
        "autorealize structured context",
        "priority rules",
        "exact source schema contract",
        "executable data read recipes",
        "source alias guard",
        "entity alias candidates",
        "minimal task reference",
        "evaluation contract reference",
        "output contract reference",
        "relation cards",
        "problem boundary reference",
        "constraints reference",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "data access",
        "modeling boundary",
        "constraints",
    },
    "solver_design": {
        "autorealize structured context",
        "priority rules",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "output contract reference",
        "relation cards",
        "problem boundary reference",
        "constraints reference",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "modeling boundary",
        "constraints",
    },
    "solve_evaluate_and_artifact": {
        "autorealize structured context",
        "priority rules",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "output contract reference",
        "problem boundary reference",
        "constraints reference",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "modeling boundary",
        "constraints",
    },
    "decision_process_design": {
        "autorealize structured context",
        "priority rules",
        "entity alias candidates",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "output contract reference",
        "relation cards",
        "problem boundary reference",
        "constraints reference",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "modeling boundary",
        "constraints",
    },
    "environment_implementation": {
        "autorealize structured context",
        "priority rules",
        "entity alias candidates",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "output contract reference",
        "relation cards",
        "problem boundary reference",
        "constraints reference",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "modeling boundary",
        "constraints",
    },
    "policy_and_algorithm_design": {
        "autorealize structured context",
        "priority rules",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "output contract reference",
        "problem boundary reference",
        "constraints reference",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "modeling boundary",
        "constraints",
    },
    "training_strategy": {
        "autorealize structured context",
        "priority rules",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "problem boundary reference",
        "constraints reference",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "modeling boundary",
        "constraints",
    },
    "rollout_and_artifact": {
        "autorealize structured context",
        "priority rules",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "output contract reference",
        "problem boundary reference",
        "constraints reference",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "modeling boundary",
        "constraints",
    },
    "model_design": {
        "autorealize structured context",
        "priority rules",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "output contract reference",
        "problem boundary reference",
        "constraints reference",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "modeling boundary",
        "constraints",
    },
    "training_evaluation": {
        "autorealize structured context",
        "priority rules",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "output contract reference",
        "problem boundary reference",
        "constraints reference",
        "leakage guards",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "modeling boundary",
        "constraints",
    },
    "merge": {
        "autorealize structured context",
        "priority rules",
        "minimal task reference",
        "method strategy",
        "evaluation contract reference",
        "output contract reference",
        "problem boundary reference",
        "constraints reference",
        "leakage guards",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "modeling boundary",
        "constraints",
    },
    "code_review": {
        "autorealize structured context",
        "priority rules",
        "exact source schema contract",
        "executable data read recipes",
        "source alias guard",
        "minimal task reference",
        "evaluation contract reference",
        "output contract reference",
        "problem boundary reference",
        "constraints reference",
        "leakage guards",
        "pitfalls",
        "problem and goal",
        "evaluation contract",
        "output contract",
        "data access",
        "modeling boundary",
        "constraints",
    },
}


def select_autorealize_context_for_stage(text: str, stage: str) -> str:
    """Select complete AutoRealize contract sections needed by one code stage.

    This is section routing, not character truncation. Unknown/non-AutoRealize
    inputs are returned unchanged so standalone MLEvolve behavior is preserved.
    """

    source = str(text or "").strip()
    selected = _STAGE_CONTEXT_SECTIONS.get(str(stage or "").strip())
    if not source or CONTEXT_MARKER.lower() not in source.lower() or selected is None:
        return source
    if str(stage or "").strip() not in _STAGE_CONTEXT_SECTIONS:
        return source

    all_headings = list(re.finditer(r"(?m)^(#{2,3})\s+([^\r\n]+?)\s*$", source))
    marker_index = next(
        (
            index
            for index, match in enumerate(all_headings)
            if match.group(2).strip().lower() == CONTEXT_MARKER.lower()
        ),
        None,
    )
    if marker_index is None or marker_index + 1 >= len(all_headings):
        return source
    section_level = len(all_headings[marker_index + 1].group(1))
    matches = [
        match
        for match in all_headings[marker_index:]
        if match is all_headings[marker_index] or len(match.group(1)) == section_level
    ]
    if not matches:
        return source
    parts: list[str] = []
    prefix = source[: matches[0].start()].strip()
    if prefix:
        parts.append(prefix)
    for index, match in enumerate(matches):
        title = match.group(2).strip().lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        if title in selected:
            parts.append(source[match.start() : end].strip())
    return "\n\n".join(parts).strip() or source


def _safe_read_json(path: Path) -> Any:
    try:
        if path.exists() and path.is_file():
            return json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to read AutoRealize json %s: %s", path, exc)
    return None


def _safe_read_text(path: Path, limit: int | None = 60000) -> str:
    try:
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            return text if limit is None else text[:limit]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to read AutoRealize text %s: %s", path, exc)
    return ""


def _first_existing(base: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def _report_dir(input_dir: Path) -> Path:
    direct = input_dir / "realize_report"
    if direct.exists():
        return direct
    nested = input_dir / "autorealize" / "realize_report"
    if nested.exists():
        return nested
    return direct


def _nonempty(values: Any, limit: int = 12) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(values, dict):
        values = values.values()
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    for value in values:
        if isinstance(value, dict):
            text = value.get("description") or value.get("name") or value.get("evidence") or json.dumps(value, ensure_ascii=False)
        else:
            text = str(value or "")
        text = re.sub(r"\s+", " ", text).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _truncate(text: str, limit: int = 1200) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _load_contracts(input_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    report_dir = _report_dir(input_dir)
    pack_path = _first_existing(
        input_dir,
        [
            "automl_context_pack.json",
        ],
    )
    if pack_path is None:
        pack_path = _first_existing(report_dir, ["automl_context_pack.json"])
    agent_pack_path = _first_existing(input_dir, ["agent_context_pack.json"])
    if agent_pack_path is None:
        agent_pack_path = _first_existing(report_dir, ["agent_context_pack.json"])
    markdown_path = _first_existing(
        input_dir,
        [
            "automl_context.md",
        ],
    )
    if markdown_path is None:
        markdown_path = _first_existing(report_dir, ["automl_context.md"])
    return {
        "input_dir": input_dir,
        "report_dir": report_dir,
        "automl_pack": _safe_read_json(pack_path) if pack_path else None,
        "automl_md": _safe_read_text(markdown_path, limit=None) if markdown_path else "",
        "automl_pack_path": str(pack_path) if pack_path else "",
        "automl_md_path": str(markdown_path) if markdown_path else "",
        "problem": _safe_read_json(report_dir / "problem_paradigm_report.json") or {},
        "data_access": _safe_read_json(report_dir / "data_access_protocol.json") or {},
        "bundle": _safe_read_json(report_dir / "description_protocol_bundle.json") or {},
        "evaluation_report": _safe_read_json(report_dir / "evaluation_contract_report.json") or {},
        "task_report": _safe_read_json(report_dir / "task_definition_report.json") or {},
        "agent_pack": _safe_read_json(agent_pack_path) if agent_pack_path else {},
        "description": _safe_read_text(input_dir / "description.md", limit=30000),
    }


def _pack_is_prompt_ready(pack: Any) -> bool:
    """Return True only for the final AutoRealize AutoML context pack.

    Partial artifacts such as agent_context_pack/problem_paradigm_report are useful
    audit material, but they are not enough to replace MLEvolve's standalone data
    preview. Accept only packs that carry the downstream contracts MLEvolve needs.
    """

    if not isinstance(pack, dict) or not pack:
        return False
    has_goal = bool(pack.get("task_goal") or pack.get("problem_paradigm"))
    has_data = bool(pack.get("data_access"))
    has_eval = isinstance(pack.get("evaluation_contract"), dict) and bool(pack.get("evaluation_contract"))
    has_output = isinstance(pack.get("output_contract"), dict) and bool(pack.get("output_contract"))
    return has_goal and has_data and (has_eval or has_output)


def _markdown_is_prompt_ready(text: str) -> bool:
    text = str(text or "").strip()
    if not text or len(text) < MIN_PROMPT_READY_CONTEXT_CHARS:
        return False
    lowered = text.lower()
    has_marker = (
        CONTEXT_MARKER.lower() in lowered
        or "autorealize context for automl" in lowered
        or "automl context" in lowered
    )
    has_data = (
        "data access" in lowered
        or "data access and orchestration" in lowered
        or "exact source schema contract" in lowered
        or "supplemental data facts" in lowered
        or "data facts" in lowered
        or "鏁版嵁璁块棶" in text
    )
    has_eval = "evaluation contract" in lowered or "final_validation_score" in lowered or "评估" in text
    has_output = "output contract" in lowered or "submission" in lowered or "输出" in text or "提交" in text
    return has_marker and has_data and (has_eval or has_output)


def has_autorealize_context(input_dir: Path) -> bool:
    contracts = _load_contracts(Path(input_dir))
    return _pack_is_prompt_ready(contracts.get("automl_pack")) or _markdown_is_prompt_ready(
        str(contracts.get("automl_md") or "")
    )


def infer_autorealize_task_mode(text: str) -> str | None:
    """Read the task-method route authored by AutoRealize.

    Returns ``None`` for standalone previews so callers can use conservative
    keyword fallback without overriding a structured method contract.
    """

    source = str(text or "")
    lowered = source.lower()
    if CONTEXT_MARKER.lower() not in lowered:
        return None

    def _value(key: str) -> str:
        match = re.search(rf"(?im)^\s*-\s*{re.escape(key)}\s*:\s*`?([^`\r\n]+)", source)
        return match.group(1).strip().lower() if match else ""

    rl_required = _value("rl_required") in {"true", "yes", "1"}
    first_method = _value("first_draft_method")
    required_block = ""
    required_match = re.search(
        r"(?ims)^\s*-\s*required_method_families\s*:\s*(.*?)(?=^\s*-\s*[a-z_]+\s*:|^#{2,3}\s|\Z)",
        source,
    )
    if required_match:
        required_block = required_match.group(1).lower()
    if rl_required or "reinforcement_learning" in first_method or "reinforcement" in required_block:
        return "rl"

    structure = _value("problem_structure")
    paradigm = _value("problem_paradigm")
    if structure == "native_sequential_control":
        return "rl"
    if structure == "decision_optimization":
        return "optimization"
    if paradigm in {"static_optimization", "hybrid_ml_optimization"}:
        return "optimization"
    if paradigm == "reinforcement_learning":
        return "rl"
    if structure == "prediction" or paradigm == "ml_dl_prediction":
        return "prediction"
    return None


def _render_from_pack(pack: dict[str, Any]) -> str:
    lines = [
        f"## {CONTEXT_MARKER}",
        "",
        "This section was generated by AutoRealize and has higher priority than generic Kaggle templates, cold-start recommendations, and lightweight data previews.",
        "When this context is present, MLEvolve must not regenerate or append a separate data preview; AutoRealize has already authored the stable task/data context.",
        "",
    ]
    priority_rules = _nonempty(pack.get("priority_rules"), limit=8)
    if priority_rules:
        lines.append("### Priority Rules")
        lines.extend(f"- {x}" for x in priority_rules)
        lines.append("")

    schema_contract = pack.get("data_schema_contract") if isinstance(pack.get("data_schema_contract"), dict) else {}
    if schema_contract:
        lines.append("### Exact Source Schema Contract")
        for item in _nonempty(schema_contract.get("rules"), limit=10):
            lines.append(f"- {item}")
        snippet = str(schema_contract.get("runtime_inspection_snippet", "") or "").strip()
        if snippet:
            lines.append(f"- runtime_inspection_snippet: {_truncate(snippet, 1200)}")
        workbooks = schema_contract.get("workbooks") if isinstance(schema_contract.get("workbooks"), list) else []
        for workbook in workbooks[:20]:
            if not isinstance(workbook, dict):
                continue
            sheets = [str(x) for x in (workbook.get("valid_sheet_names_exact") or [])[:20]]
            lines.append(f"- workbook `{workbook.get('source_file')}` valid_sheet_names_exact: {sheets}")
        tables = schema_contract.get("tables") if isinstance(schema_contract.get("tables"), list) else []
        for table in tables[:32]:
            if not isinstance(table, dict):
                continue
            table_id = table.get("table_id") or table.get("source_file") or "table"
            lines.append(
                f"- table `{table_id}`: kind={table.get('table_kind')}; "
                f"sheet_name={table.get('sheet_name')}; shape={table.get('shape')}; "
                f"column_count={table.get('column_count')}"
            )
            group = table.get("schema_group") if isinstance(table.get("schema_group"), dict) else {}
            if group:
                lines.append(
                    f"  - schema_group: file_count={group.get('file_count')}; "
                    f"representative_files={group.get('representative_files')}"
                )
            columns = [str(x) for x in (table.get("physical_columns_exact") or [])[:120]]
            if columns:
                lines.append("  - physical_columns_exact: " + ", ".join(f"`{x}`" for x in columns))
            omitted = int(table.get("physical_columns_omitted") or 0)
            if omitted:
                lines.append(f"  - physical_columns_omitted: {omitted}")
            fields = table.get("field_summaries") if isinstance(table.get("field_summaries"), list) else []
            if fields:
                rendered = []
                for field in fields[:10]:
                    if not isinstance(field, dict):
                        continue
                    rendered.append(
                        "; ".join(
                            str(x)
                            for x in [
                                f"name={field.get('name')}",
                                f"meaning={_truncate(field.get('meaning'), 180)}" if field.get("meaning") else "",
                                f"type={field.get('logical_type')}" if field.get("logical_type") else "",
                            ]
                            if str(x).strip()
                        )
                    )
                if rendered:
                    lines.append("  - key_field_summaries: " + " | ".join(rendered))
        lines.append("")
    lines.append("### Problem And Goal")
    lines.append(f"- problem_paradigm: `{pack.get('problem_paradigm') or 'unknown_but_executable'}`")
    for key in ["problem_structure", "decision_temporality", "environment_source"]:
        value = pack.get(key)
        if value not in (None, ""):
            lines.append(f"- {key}: `{value}`")
    if pack.get("task_goal"):
        lines.append(f"- task_goal: {_truncate(pack.get('task_goal'), 1600)}")
    lines.append("")

    method = pack.get("method_strategy") if isinstance(pack.get("method_strategy"), dict) else {}
    if method:
        lines.append("### Method Strategy")
        for key in [
            "problem_paradigm",
            "problem_structure",
            "decision_temporality",
            "environment_source",
            "explicit_rl_requested",
            "rl_as_required_paradigm",
            "rl_required",
            "first_draft_method",
        ]:
            if key in method:
                lines.append(f"- {key}: `{method.get(key)}`")
        for key in ["required_method_families", "allowed_method_families", "recommended_solver_families"]:
            families = _nonempty(method.get(key), limit=16)
            if families:
                lines.append(f"- {key}: " + ", ".join(f"`{x}`" for x in families))
        for key in ["first_draft_policy"]:
            value = str(method.get(key, "") or "").strip()
            if value:
                lines.append(f"- {key}: {_truncate(value, 1000)}")
        candidates = method.get("rl_formulation_candidates")
        if isinstance(candidates, list) and candidates:
            lines.append("- rl_formulation_candidates:")
            for index, candidate in enumerate(candidates[:8], 1):
                if not isinstance(candidate, dict):
                    continue
                lines.append(
                    f"  - candidate_{index}: formulation_type={candidate.get('formulation_type')}; "
                    f"generalization_target={candidate.get('generalization_target')}"
                )
                for candidate_key in [
                    "state_candidates",
                    "action_candidates",
                    "transition_description",
                    "action_mask_basis",
                    "reward_alignment",
                    "terminal_condition",
                    "episode_generation",
                    "evidence_refs",
                    "unresolved_points",
                ]:
                    candidate_value = candidate.get(candidate_key)
                    if candidate_value not in (None, "", [], {}):
                        lines.append(f"    - {candidate_key}: {_truncate(candidate_value, 1000)}")
        notes = _nonempty(method.get("method_routing_notes"), limit=8)
        if notes:
            lines.append("- method_routing_notes:")
            lines.extend(f"  - {_truncate(x, 800)}" for x in notes)
        lines.append("")

    evaluation = pack.get("evaluation_contract") if isinstance(pack.get("evaluation_contract"), dict) else {}
    if evaluation:
        lines.append("### Evaluation Contract")
        final_formula = str(evaluation.get("final_score_formula") or evaluation.get("metric_formula") or "").strip()
        for key in [
            "passed",
            "executable",
            "authority_status",
            "primary_metric",
            "metric_direction",
            "prediction_unit",
            "computation_scope",
            "aggregation_rule",
            "validation_protocol",
        ]:
            value = str(evaluation.get(key, "") or "").strip()
            if value:
                lines.append(f"- {key}: {_truncate(value, 1800)}")
        for key in ["assumptions", "residual_issues"]:
            values = _nonempty(evaluation.get(key), limit=12)
            if values:
                lines.append(f"- {key}:")
                lines.extend(f"  - {_truncate(x, 800)}" for x in values)
        if final_formula:
            lines.append(f"- final_score_formula: {_truncate(final_formula, 1800)}")
        lines.append("- final_validation_score: print exactly one numeric `Final Validation Score` using `final_score_formula`; do not create or optimize another metric.")
        for key in ["submission_checks", "invalid_solution_rules", "tie_break_rules"]:
            values = _nonempty(evaluation.get(key), limit=8)
            if values:
                lines.append(f"- {key}:")
                lines.extend(f"  - {_truncate(x, 800)}" for x in values)
        lines.append("")

    output = pack.get("output_contract") if isinstance(pack.get("output_contract"), dict) else {}
    if output:
        lines.append("### Output Contract")
        for key in ["output_kind", "output_filename", "sample_submission_required", "row_unit", "no_sample_submission_reason"]:
            value = output.get(key)
            if value not in (None, "", []):
                lines.append(f"- {key}: {value}")
        columns = _nonempty(output.get("columns"), limit=60)
        if columns:
            lines.append("- columns: " + ", ".join(f"`{x}`" for x in columns))
        rules = _nonempty(output.get("format_rules"), limit=10)
        if rules:
            lines.append("- format_rules:")
            lines.extend(f"  - {_truncate(x, 700)}" for x in rules)
        lines.append("")

    data_access = pack.get("data_access") if isinstance(pack.get("data_access"), list) else []
    if data_access:
        lines.append("### Data Access And Orchestration")
        for item in _nonempty(pack.get("data_orchestration"), limit=8):
            lines.append(f"- {item}")
        for entry in data_access[:20]:
            if not isinstance(entry, dict):
                continue
            name = entry.get("pattern") or entry.get("path") or "data file"
            lines.append(f"- `{name}`")
            for key in ["kind", "file_count", "read_method", "row_grain", "orchestration_note"]:
                value = entry.get(key)
                if value not in (None, "", []):
                    lines.append(f"  - {key}: {_truncate(value, 700)}")
            read_example = str(entry.get("read_example", "") or "").strip()
            if read_example:
                lines.append("  - read_example:")
                lines.append("```python")
                lines.append(read_example.replace("```python", "").replace("```", "").strip())
                lines.append("```")
            notes = _nonempty(entry.get("parsing_notes"), limit=5)
            if notes:
                lines.append("  - parsing_notes: " + "; ".join(_truncate(x, 280) for x in notes))
        lines.append("")

        read_contracts = [
            entry.get("read_contract")
            for entry in data_access[:40]
            if isinstance(entry, dict)
            and isinstance(entry.get("read_contract"), dict)
            and entry.get("read_contract")
        ]
        if read_contracts:
            lines.append("### Executable Data Read Recipes")
            lines.append(
                "- These recipes were produced from successful AutoRealize reads. Prefer reusing them. "
                "They are not an immutable host-level gate: if runtime evidence conflicts, adjust explicitly "
                "and report the observed shape and columns."
            )
            lines.append("```json")
            lines.append(json.dumps(read_contracts, ensure_ascii=False, sort_keys=True, indent=2))
            lines.append("```")
            lines.append("")

    for title, key in [
        ("Modeling Boundary", "modeling_boundary"),
        ("Constraints", "constraints"),
        ("Leakage Guards", "leakage_guards"),
        ("Pitfalls", "pitfalls"),
    ]:
        values = _nonempty(pack.get(key), limit=16)
        if values:
            lines.append(f"### {title}")
            lines.extend(f"- {_truncate(x, 900)}" for x in values)
            lines.append("")
    return "\n".join(lines).strip()


def _render_fallback(contracts: dict[str, Any]) -> str:
    problem = contracts.get("problem") or {}
    bundle = contracts.get("bundle") or {}
    data_access = contracts.get("data_access") or {}
    evaluation_report = contracts.get("evaluation_report") or {}
    evaluation = evaluation_report.get("final") if isinstance(evaluation_report, dict) else {}
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    lines = [
        f"## {CONTEXT_MARKER}",
        "",
        "AutoRealize structured artifacts were detected. Treat them as high-priority task facts and do not regenerate a raw MLEvolve data preview.",
        "",
        "### Problem And Goal",
        f"- problem_paradigm: `{problem.get('problem_paradigm') or bundle.get('problem_paradigm') or 'unknown_but_executable'}`",
    ]
    if bundle.get("task_goal") or bundle.get("overview"):
        lines.append(f"- task_goal: {_truncate(bundle.get('task_goal') or bundle.get('overview'), 1600)}")
    if evaluation:
        lines.extend(["", "### Evaluation Contract"])
        final_formula = str(evaluation.get("final_score_formula") or evaluation.get("metric_formula") or "").strip()
        for key in ["primary_metric", "metric_direction", "aggregation_rule", "validation_protocol"]:
            value = str(evaluation.get(key, "") or "").strip()
            if value:
                lines.append(f"- {key}: {_truncate(value, 1800)}")
        if final_formula:
            lines.append(f"- final_score_formula: {_truncate(final_formula, 1800)}")
        lines.append("- final_validation_score: print exactly one numeric scalar using `final_score_formula`.")
    output = bundle.get("output") if isinstance(bundle.get("output"), dict) else {}
    if output:
        lines.extend(["", "### Output Contract"])
        for key in ["output_kind", "output_filename", "sample_submission_required", "row_unit", "no_sample_submission_reason"]:
            value = output.get(key)
            if value not in (None, "", []):
                lines.append(f"- {key}: {value}")
        columns = _nonempty(output.get("columns"), limit=60)
        if columns:
            lines.append("- columns: " + ", ".join(f"`{x}`" for x in columns))
    files = data_access.get("files") if isinstance(data_access, dict) else []
    if files:
        lines.extend(["", "### Data Access"])
        for item in files[:20]:
            if not isinstance(item, dict):
                continue
            name = item.get("path") or "data file"
            lines.append(f"- `{name}`: {item.get('read_method') or 'pandas'}")
            if item.get("read_example"):
                lines.append("```python")
                lines.append(str(item.get("read_example")).strip())
                lines.append("```")
            notes = _nonempty(item.get("parsing_notes"), limit=4)
            if notes:
                lines.append("  - parsing_notes: " + "; ".join(_truncate(x, 260) for x in notes))
    return "\n".join(lines).strip()


def build_autorealize_context_md(input_dir: Path, *, write_context_file: bool = True) -> str:
    """Return the prompt context authored by AutoRealize, and optionally materialize it.

    MLEvolve must not recreate data cognition when an AutoRealize package is
    available. AutoRealize owns context construction; this adapter only copies
    the already-authored prompt into the MLEvolve input directory.
    """
    contracts = _load_contracts(Path(input_dir))
    existing_md = str(contracts.get("automl_md") or "").strip()
    text = ""
    if existing_md:
        text = existing_md if CONTEXT_MARKER in existing_md else f"## {CONTEXT_MARKER}\n\n{existing_md}"
        logger.info("Using AutoRealize automl_context.md directly (%s chars).", len(existing_md))
    if not text and _pack_is_prompt_ready(contracts.get("automl_pack")):
        text = _render_from_pack(contracts["automl_pack"])
    if not text:
        fallback = _render_fallback(contracts)
        if _markdown_is_prompt_ready(fallback):
            text = fallback
        elif any(contracts.get(k) for k in ["bundle", "evaluation_report", "data_access", "problem", "agent_pack"]):
            logger.warning(
                "AutoRealize artifacts exist but are not prompt-ready for MLEvolve; "
                "standalone data preview remains enabled."
            )
    if text and write_context_file:
        try:
            (Path(input_dir) / "autorealize_context.md").write_text(text + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to write autorealize_context.md: %s", exc)
    return text


def load_autorealize_description_md(input_dir: Path, *, fallback: str = "") -> str:
    """Return reader-facing description.md for MLEvolve task_desc."""
    contracts = _load_contracts(Path(input_dir))
    text = str(contracts.get("description") or "").strip()
    return text or str(fallback or "").strip()


def submission_required_from_context(input_dir: Path) -> bool | None:
    """Infer whether MLEvolve should require ./submission/submission.csv."""
    contracts = _load_contracts(Path(input_dir))
    pack = contracts.get("automl_pack")
    output = {}
    if isinstance(pack, dict):
        output = pack.get("output_contract") if isinstance(pack.get("output_contract"), dict) else {}
    if not output:
        bundle = contracts.get("bundle") or {}
        output = bundle.get("output") if isinstance(bundle.get("output"), dict) else {}
    if output:
        if bool(output.get("sample_submission_required")):
            return True
        kind = str(output.get("output_kind") or "").lower()
        columns = _nonempty(output.get("columns"), limit=5)
        reason = str(output.get("no_sample_submission_reason") or "").strip()
        if kind in {"policy", "solution", "solution_table", "artifact", "report"} and reason:
            return False
        if not columns and reason:
            return False
    problem = str((contracts.get("problem") or {}).get("problem_paradigm") or "").strip()
    if problem in {"static_optimization", "reinforcement_learning"}:
        sample = Path(input_dir) / "sample_submission.csv"
        return True if sample.exists() else False
    return None
