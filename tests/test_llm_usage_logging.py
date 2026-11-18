from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_USAGE_PATH = Path(__file__).resolve().parents[1] / "llm" / "usage.py"
_SPEC = importlib.util.spec_from_file_location("algoevolve_llm_usage_for_test", _USAGE_PATH)
assert _SPEC and _SPEC.loader
_USAGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_USAGE)
log_llm_usage = _USAGE.log_llm_usage


def test_resumed_usage_replays_events_without_reset_or_double_count(tmp_path):
    cfg = SimpleNamespace(log_dir=tmp_path)
    args = dict(cfg=cfg, prompt_name='generation', mode='generate', provider='openai_compatible',
        model='deepseek-flash', usage={'prompt_tokens':100,'completion_tokens':10,
        'prompt_cache_hit_tokens':80,'prompt_cache_miss_tokens':20}, seconds=2)
    log_llm_usage(**args)
    # Simulate a fresh process, and a stale summary that must not be trusted.
    _USAGE._SUMMARIES.clear()
    (tmp_path/'llm_usage_summary.json').write_text('{"calls":999}',encoding='utf-8')
    log_llm_usage(**args)
    summary=json.loads((tmp_path/'llm_usage_summary.json').read_text(encoding='utf-8'))
    assert summary['calls']==2 and summary['prompt_tokens']==200
    assert summary['completion_tokens']==20 and summary['seconds']==4
    assert summary['provider_cache_hit_ratio']==.8
    assert summary['by_prompt']['generation']['calls']==2
    # A partial tail from a process interruption remains auditable. A new
    # complete event must not be joined to it and lost on the next replay.
    with (tmp_path/'llm_usage.jsonl').open('a',encoding='utf-8') as stream:
        stream.write('{"partial":')
    _USAGE._SUMMARIES.clear()
    log_llm_usage(**args)
    _USAGE._SUMMARIES.clear()
    rebuilt=_USAGE._summary_for(tmp_path/'llm_usage_summary.json')
    assert rebuilt['calls']==3 and rebuilt['prompt_tokens']==300
    assert rebuilt['usage_replay_skipped_rows']==1


def test_log_llm_usage_writes_jsonl_summary_and_prompt_parts(tmp_path):
    cfg = SimpleNamespace(log_dir=tmp_path)
    response = SimpleNamespace(
        id="chatcmpl-deepseek-test",
        model="deepseek-v4-flash",
        system_fingerprint="fp_deepseek_v4_test",
        usage={
            "prompt_tokens": 120,
            "completion_tokens": 12,
            "total_tokens": 132,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 7},
        }
    )

    row = log_llm_usage(
        cfg=cfg,
        prompt_name="draft_agent",
        mode="generate_stream",
        provider="openai_compatible",
        model="deepseek-chat",
        response=response,
        seconds=1.25,
        max_tokens=4096,
        parsed_ok=True,
        prompt_parts=[
            {"name": "system_prompt", "role": "system", "content": "stable"},
            {"name": "task_context", "role": "user", "content": "x" * 400},
            {"name": "dynamic_feedback", "role": "user", "content": "fix this"},
        ],
        estimated_completion_text="answer",
    )

    assert row["prompt_tokens"] == 120
    assert row["completion_tokens"] == 12
    assert row["reasoning_tokens"] == 7
    assert row["response_model"] == "deepseek-v4-flash"
    assert row["system_fingerprint"] == "fp_deepseek_v4_test"
    assert row["prompt_cache_hit_tokens"] == 80
    usage_rows = [json.loads(line) for line in (tmp_path / "llm_usage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert usage_rows[-1]["prompt_parts"][1]["name"] == "task_context"
    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    assert summary["calls"] == 1
    assert summary["prompt_tokens"] == 120
    assert summary["provider_cache_hit_ratio"] == round(80 / 120, 6)
    assert summary["reasoning_tokens"] == 7
    assert summary["provider_response_models"] == {"deepseek-v4-flash": 1}
    assert summary["system_fingerprints"] == {"fp_deepseek_v4_test": 1}
    assert summary["deepseek_cost_breakdown_usd"]["cache_miss_input_tokens"] == 40
    assert summary["deepseek_cost_breakdown_usd"]["output_tokens"] == 12
    assert summary["by_prompt"]["draft_agent"]["by_part"]["task_context"]["estimated_tokens"] > 0
    assert summary["by_prompt_part_ranked"][0]["estimated_tokens"] >= summary["by_prompt_part_ranked"][-1]["estimated_tokens"]
    brief = json.loads((tmp_path / "llm_usage_brief.json").read_text(encoding="utf-8"))
    assert brief["schema_version"] == "algoevolve.llm_usage_brief.v4"
    assert brief["deepseek_pricing_usd_per_1m"]["cache_miss_input"] == 0.14
    assert brief["deepseek_cost_breakdown_usd"]["output_tokens"] == 12
    assert brief["reasoning_tokens"] == 7
    assert brief["top_prompts_by_estimated_cost"][0]["stage"] == "unknown_stage"
    assert brief["top_prompts_by_estimated_cost"][0]["legacy_prompt_stage_guess"] == "draft_or_code_generation"
    assert brief["by_stage"][0]["stage"] == "unknown_stage"


def test_log_llm_usage_marks_missing_provider_usage(tmp_path):
    cfg = SimpleNamespace(log_dir=tmp_path)

    row = log_llm_usage(
        cfg=cfg,
        prompt_name="stream_without_usage",
        mode="generate_stream",
        provider="gemini",
        model="gemini-test",
        usage={},
        prompt_parts=[{"name": "prompt", "role": "user", "content": "hello world"}],
        estimated_completion_text="ok",
    )

    assert row["usage_available"] is False
    assert row["estimated_prompt_tokens"] > 0
    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    assert summary["provider_usage_missing_calls"] == 1
    assert summary["estimated_completion_tokens"] > 0


def test_mixed_deepseek_models_are_priced_per_call_model(tmp_path):
    cfg = SimpleNamespace(log_dir=tmp_path)
    for model in ["deepseek-v4-pro", "deepseek-v4-flash"]:
        response = SimpleNamespace(
            model=model,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 100,
            },
        )
        log_llm_usage(
            cfg=cfg,
            prompt_name="query:agents.code_review_agent.run",
            mode="query_function",
            provider="openai_compatible",
            model=model,
            response=response,
        )

    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    brief = json.loads((tmp_path / "llm_usage_brief.json").read_text(encoding="utf-8"))

    assert summary["models"] == {"deepseek-v4-pro": 1, "deepseek-v4-flash": 1}
    assert summary["estimated_deepseek_usd_unknown_prompt_as_miss"] == 0.000069
    assert summary["deepseek_cost_by_model_usd"]["deepseek-v4-pro"][
        "estimated_deepseek_usd_unknown_prompt_as_miss"
    ] == 0.0000522
    assert summary["deepseek_cost_by_model_usd"]["deepseek-v4-flash"][
        "estimated_deepseek_usd_unknown_prompt_as_miss"
    ] == 0.0000168
    prompt = summary["by_prompt"]["query:agents.code_review_agent.run"]
    assert prompt["models"] == summary["models"]
    assert brief["model"] == "mixed"
    assert brief["models"] == summary["models"]
    assert brief["top_prompts_by_estimated_cost"][0]["models"] == summary["models"]


def test_zero_reported_input_is_unknown_and_does_not_pollute_cache_ratio(tmp_path):
    cfg = SimpleNamespace(log_dir=tmp_path)
    for tokens, cached in [(100, 80), (0, 10)]:
        row = log_llm_usage(
            cfg=cfg, prompt_name="generation", mode="generate_stream", provider="openai_compatible",
            model="gpt-5.6-luna", usage={"prompt_tokens": tokens, "completion_tokens": 12,
                                       "prompt_tokens_details": {"cached_tokens": cached}},
            prompt_parts=[{"name": "task", "role": "user", "content": "A real nonempty task context"}],
        )
    assert row["prompt_tokens"] == 0
    assert row["estimated_prompt_tokens"] > 0
    assert row["prompt_usage_available"] is False
    assert row["provider_cache_tokens_known"] is False
    assert row["raw_usage"]["prompt_tokens_details"]["cached_tokens"] == 10
    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    assert summary["provider_usage_missing_calls"] == 1
    assert summary["completion_tokens"] == 24
    assert summary["prompt_cache_hit_tokens"] == 80
    assert summary["provider_cache_hit_ratio"] is None
    assert summary["known_provider_cache_hit_ratio"] == .8
    brief = json.loads((tmp_path / "llm_usage_brief.json").read_text(encoding="utf-8"))
    assert len(brief["by_stage"]) == 1
    stage = brief["by_stage"][0]
    assert stage["provider_usage_missing_calls"] == 1
    assert stage["estimated_input_tokens"] > 0


def test_usage_brief_keeps_unknown_cache_telemetry_distinct_from_zero(tmp_path):
    cfg = SimpleNamespace(log_dir=tmp_path)
    common = dict(
        cfg=cfg,
        prompt_name="query:agents.result_parse_agent.run",
        mode="query_function",
        provider="openai_compatible",
        model="third-party-model",
        prompt_parts=[{"name": "task", "role": "user", "content": "Review the candidate output."}],
    )
    log_llm_usage(**common, usage={"prompt_tokens": 100, "completion_tokens": 8})
    log_llm_usage(
        **common,
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 8,
            "prompt_cache_hit_tokens": 40,
            "prompt_cache_miss_tokens": 60,
        },
    )
    log_llm_usage(
        **common,
        usage={
            "prompt_tokens": 50,
            "completion_tokens": 8,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 50,
        },
    )

    brief = json.loads((tmp_path / "llm_usage_brief.json").read_text(encoding="utf-8"))
    prompt = brief["top_prompts_by_estimated_cost"][0]
    stage = next(row for row in brief["by_stage"] if row["stage"] == prompt["stage"])

    assert brief["schema_version"] == "algoevolve.llm_usage_brief.v4"
    assert brief["input_tokens"] == 250
    assert brief["cache_telemetry_known_input_tokens"] == 150
    assert brief["unknown_input_tokens"] == 100
    assert brief["cache_telemetry_coverage"] == 0.6
    assert brief["cache_observed_calls"] == 2
    assert brief["cache_hit_calls"] == 1
    assert brief["cache_unknown_calls"] == 1
    assert brief["cache_hit_request_ratio"] == 0.5
    assert prompt["unknown_input_tokens"] == 100
    assert prompt["cache_hit_request_ratio"] == 0.5
    assert prompt["known_input_cache_hit_ratio"] == round(40 / 150, 6)
    assert prompt["provider_cache_hit_ratio"] is None
    assert stage["unknown_input_tokens"] == 100
    assert stage["cache_telemetry_coverage"] == 0.6
    assert stage["cache_hit_request_ratio"] == 0.5
    assert stage["provider_cache_hit_ratio"] is None


def test_stage_usage_is_explicit_and_old_rows_replay_as_unknown(tmp_path):
    cfg = SimpleNamespace(log_dir=tmp_path)
    log_llm_usage(
        cfg=cfg,
        prompt_name="query:agents.planner.run",
        stage_id="planner",
        mode="query_function",
        provider="openai_compatible",
        model="test-model",
        usage={"prompt_tokens": 100, "completion_tokens": 20},
    )

    usage_path = tmp_path / "llm_usage.jsonl"
    rows = [
        json.loads(line)
        for line in usage_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0].pop("stage_id")
    rows[0].pop("stage_id_source")
    usage_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    _USAGE._SUMMARIES.clear()

    row = log_llm_usage(
        cfg=cfg,
        prompt_name="query:agents.planner.run",
        stage_id="planner",
        mode="query_function",
        provider="openai_compatible",
        model="test-model",
        usage={"prompt_tokens": 80, "completion_tokens": 10},
    )

    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    brief = json.loads((tmp_path / "llm_usage_brief.json").read_text(encoding="utf-8"))
    assert row["stage_id"] == "planner"
    assert summary["calls"] == 2
    assert summary["by_stage_id"]["unknown_stage"]["calls"] == 1
    assert summary["by_stage_id"]["planner"]["calls"] == 1
    assert summary["by_stage_id"]["planner"]["stage_id_source_counts"] == {"explicit": 1}
    assert summary["by_stage_id"]["unknown_stage"]["stage_id_source_counts"] == {"missing": 1}
    assert {item["stage_id"] for item in brief["by_stage"]} == {"unknown_stage", "planner"}
    assert brief["schema_version"] == "algoevolve.llm_usage_brief.v4"
