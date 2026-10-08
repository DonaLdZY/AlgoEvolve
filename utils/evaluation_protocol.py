"""Read candidate-reported evaluation metadata without treating it as certification."""

from __future__ import annotations

import json
import re
from typing import Any

from utils.decision_validation import extract_decision_validation_summary


PROTOCOL_FIELDS = {
    "contract_id", "evaluation_id", "metric_name", "maximize",
    "split_fingerprint", "evaluation_count", "cutoff", "horizon",
    "evaluation_phase", "scoring_split", "final_evaluation_accessed",
}


def extract_contract_identity(context: str) -> dict[str, str]:
    section = re.search(r"(?ms)^#{2,3} Execution Contract[ \t]*\r?\n(.*?)(?=^#{1,3} |\Z)", str(context or ""))
    if section is None:
        return {}
    block = re.search(r"```json\s*(\{.*?\})\s*```", section.group(1), re.DOTALL)
    if block is None:
        return {}
    try:
        value = json.loads(block.group(1))
    except (ValueError, TypeError):
        return {}
    return {
        f"expected_{key}": value[key]
        for key in ("contract_id", "evaluation_id")
        if isinstance(value, dict) and isinstance(value.get(key), str) and value[key]
    }


def extract_evaluation_protocol(output: Any) -> dict[str, Any]:
    text = "".join(str(part) for part in output) if isinstance(output, list) else str(output or "")
    protocol: dict[str, Any] = {}
    for line in text.splitlines():
        prefix, separator, value = line.strip().partition(":")
        if prefix != "Evaluation Protocol Summary" or not separator:
            continue
        try:
            parsed = json.loads(value.strip())
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            protocol = {
                key: item for key, item in parsed.items()
                if key in PROTOCOL_FIELDS and (
                    isinstance(item, (str, int, float, bool, type(None)))
                    or (key in {"cutoff", "horizon"} and isinstance(item, (dict, list)))
                )
            }
    summary = extract_decision_validation_summary(text)
    nested = summary.get("evaluation_protocol") if isinstance(summary, dict) else None
    if isinstance(nested, dict):
        nested = {
            key: item for key, item in nested.items()
            if key in PROTOCOL_FIELDS and (
                isinstance(item, (str, int, float, bool, type(None)))
                or (key in {"cutoff", "horizon"} and isinstance(item, (dict, list)))
            )
        }
        conflicts = {
            key for key in protocol.keys() & nested.keys()
            if json.dumps(protocol[key], sort_keys=True) != json.dumps(nested[key], sort_keys=True)
        }
        if conflicts:
            # Conflicting reports must not hide exposure or qualify for ranking.
            exposed = protocol.get("final_evaluation_accessed") is True or nested.get("final_evaluation_accessed") is True
            return {"final_evaluation_accessed": exposed}
        protocol = {**nested, **protocol}
    return protocol
