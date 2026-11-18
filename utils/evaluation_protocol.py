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
            protocol = normalize_evaluation_protocol(parsed)
    summary = extract_decision_validation_summary(text)
    nested = summary.get("evaluation_protocol") if isinstance(summary, dict) else None
    if isinstance(nested, dict):
        return merge_evaluation_protocols(protocol, nested)
    return protocol


def normalize_evaluation_protocol(value: Any) -> dict[str, Any]:
    """Keep only bounded protocol fields from an untrusted candidate document."""

    if not isinstance(value, dict):
        return {}
    return {
        key: item for key, item in value.items()
        if key in PROTOCOL_FIELDS and (
            isinstance(item, (str, int, float, bool, type(None)))
            or (key in {"cutoff", "horizon"} and isinstance(item, (dict, list)))
        )
    }


def merge_evaluation_protocols(*values: Any) -> dict[str, Any]:
    """Merge protocol receipts, marking disagreement instead of picking a winner."""

    trusted_base = values[0] if values and isinstance(values[0], dict) else {}
    expected = {
        key: trusted_base[key]
        for key in ("expected_contract_id", "expected_evaluation_id")
        if isinstance(trusted_base.get(key), str) and trusted_base[key]
    }
    protocols = [normalize_evaluation_protocol(value) for value in values]
    merged: dict[str, Any] = {}
    conflicts: set[str] = set()
    for protocol in protocols:
        for key, value in protocol.items():
            if key in merged and json.dumps(merged[key], sort_keys=True) != json.dumps(value, sort_keys=True):
                conflicts.add(key)
            else:
                merged[key] = value

    if conflicts or any(
        isinstance(value, dict) and value.get("protocol_conflict") is True
        for value in values
    ):
        result = {**expected, "protocol_conflict": True}
        if any(protocol.get("final_evaluation_accessed") is True for protocol in protocols):
            result["final_evaluation_accessed"] = True
        elif any(protocol.get("final_evaluation_accessed") is False for protocol in protocols):
            result["final_evaluation_accessed"] = False
        return result

    return {**expected, **merged}
