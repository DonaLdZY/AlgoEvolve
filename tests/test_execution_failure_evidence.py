import json
import pytest

from engine.execution_evidence import (
    candidate_evaluation_protocol,
    candidate_failure_evidence,
    failure_csv_header_evidence,
    failure_context,
)


@pytest.mark.parametrize("filename", ["evaluation_failure.json", "run_failure.json"])
def test_failure_evidence_reads_only_the_current_node(tmp_path, filename):
    directory = tmp_path / "node_runs/candidate/working"
    directory.mkdir(parents=True)
    failure = {"error_type": "ContractError", "error": "Expected nested Cooling_rates_S-pattern.csv", "evaluation_count": 0}
    (directory / filename).write_text(json.dumps(failure), encoding="utf-8")
    assert candidate_failure_evidence(tmp_path, "candidate")["content"] == failure
    assert candidate_failure_evidence(tmp_path, "another") is None


def test_failure_evidence_does_not_follow_links_outside_candidate(tmp_path):
    other = tmp_path / "other.json"
    other.write_text('{"error":"unrelated"}', encoding="utf-8")
    directory = tmp_path / "node_runs/candidate/working"
    directory.mkdir(parents=True)
    (directory / "evaluation_failure.json").symlink_to(other)
    assert candidate_failure_evidence(tmp_path, "candidate") is None


def test_failure_evidence_handles_large_or_malformed_payload(tmp_path):
    directory = tmp_path / "node_runs/candidate/working"
    directory.mkdir(parents=True)
    path = directory / "evaluation_failure.json"
    path.write_text('{"error":', encoding="utf-8")
    assert candidate_failure_evidence(tmp_path, "candidate")["content_status"] == "unreadable_json"
    path.write_bytes(b"x" * 16385)
    assert candidate_failure_evidence(tmp_path, "candidate")["content_status"] == "exceeds_16KiB_diagnostic_limit"


def test_protocol_evidence_reads_only_current_node_allowlisted_paths(tmp_path):
    evidence = tmp_path / "node_runs/candidate/working/evidence"
    evidence.mkdir(parents=True)
    protocol = {
        "evaluation_phase": "search",
        "scoring_split": "development",
        "final_evaluation_accessed": False,
        "private_extra": "ignored by protocol normalizer",
    }
    (evidence / "evaluation_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")

    assert candidate_evaluation_protocol(tmp_path, "candidate") == (protocol, "read")
    assert candidate_evaluation_protocol(tmp_path, "another") == (None, "missing")


def test_protocol_evidence_rejects_symlinks_malformed_and_oversized_files(tmp_path):
    evidence = tmp_path / "node_runs/candidate/working/evidence"
    evidence.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{"evaluation_phase":"search"}', encoding="utf-8")
    (evidence / "evaluation_protocol.json").symlink_to(outside)
    assert candidate_evaluation_protocol(tmp_path, "candidate") == (None, "unsafe_path")

    (evidence / "evaluation_protocol.json").unlink()
    (evidence / "evaluation_protocol.json").write_text("{", encoding="utf-8")
    assert candidate_evaluation_protocol(tmp_path, "candidate") == (None, "invalid")

    (evidence / "evaluation_protocol.json").write_bytes(b"x" * (16 * 1024 + 1))
    assert candidate_evaluation_protocol(tmp_path, "candidate") == (None, "too_large")


def test_conflicting_protocol_receipt_files_fail_closed(tmp_path):
    evidence = tmp_path / "node_runs/candidate/working/evidence"
    evidence.mkdir(parents=True)
    (evidence / "evaluation_protocol.json").write_text(
        '{"evaluation_phase":"search","scoring_split":"development","final_evaluation_accessed":false}',
        encoding="utf-8",
    )
    (evidence / "evaluation_protocol_summary.json").write_text(
        '{"evaluation_phase":"search","scoring_split":"holdout","final_evaluation_accessed":false}',
        encoding="utf-8",
    )

    assert candidate_evaluation_protocol(tmp_path, "candidate") == (None, "conflict")


def test_header_probe_preserves_actual_encoding_and_only_reads_named_failure(tmp_path):
    root = tmp_path / "input/nested"
    root.mkdir(parents=True)
    (root / "Cooling_rates_S-pattern.csv").write_bytes(
        b"A1;;\nTIME [s];TIME IN SIMULATION [s];TEMPERATURE [\xb0C]\n0;29.5;2749\n")
    (root / "unrelated.csv").write_text("unrelated", encoding="utf-8")
    rows = failure_csv_header_evidence(tmp_path, "ContractError: Cooling_rates_S-pattern.csv has no verified header")
    assert len(rows) == 1
    alternatives = rows[0]["encoding_candidates_not_an_authority_override"]
    cp1252 = next(item for item in alternatives if item["encoding"] == "cp1252")
    assert cp1252["raw_header_rows"][1][2] == "TEMPERATURE [\u00b0C]"
    assert cp1252["column_counts"] == [3, 3]
    assert all("2749" not in json.dumps(item) for item in alternatives)
    assert failure_csv_header_evidence(tmp_path, "Model training failed") == []
    original = "ContractError: Cooling_rates_S-pattern.csv has no verified header"
    enriched = failure_context(tmp_path, "candidate", original)
    assert enriched.startswith(original)
    assert "TEMPERATURE" in enriched
    assert failure_context(tmp_path, "candidate", enriched) == enriched


@pytest.mark.parametrize("record", ["holdout-17,2026-09-08,91827", "secret_label,private_text,never_expose"])
def test_header_probe_does_not_publish_first_data_row(tmp_path, record):
    root = tmp_path / "input"
    root.mkdir()
    (root / "heldout.csv").write_text("id,date,target\n" + record + "\n", encoding="utf-8")
    rows = failure_csv_header_evidence(tmp_path, "heldout.csv failed")
    assert rows[0]["encoding_candidates_not_an_authority_override"]
    assert all(item["raw_header_rows"] == [["id", "date", "target"]]
               for item in rows[0]["encoding_candidates_not_an_authority_override"])
    assert record.split(",")[0] not in json.dumps(rows)


def test_header_probe_does_not_follow_input_file_link(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    other = tmp_path / "private.csv"
    other.write_text("private,header\n1,2\n", encoding="utf-8")
    (root / "failed.csv").symlink_to(other)
    rows = failure_csv_header_evidence(tmp_path, "failed.csv failed")
    assert rows == [{"filename": "failed.csv", "read_status": "outside_input"}]
