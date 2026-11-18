from types import SimpleNamespace as NS
import pytest
from llm import openai as backend
from llm import FunctionSpec


SPEC = FunctionSpec(name='review', description='review', json_schema={
    'type': 'object', 'properties': {'valid': {'type': 'boolean'}}, 'required': ['valid']})


def prepare_backend(monkeypatch, tmp_path, *, mode='auto', content='{"valid":true}', finish='stop'):
    calls, logs = [], []
    stage = NS(api_key='test-only', base_url='https://api.deepseek.com', model='deepseek-flash',
               enable_thinking=True, reasoning_effort='high', max_tokens=1024, minimum_output_tokens=0)
    cfg = NS(agent=NS(code=stage, feedback=stage), log_dir=tmp_path)
    response = NS(choices=[NS(message=NS(content=content, tool_calls=None), finish_reason=finish)],
                  usage=NS(prompt_tokens=100, completion_tokens=10), model='deepseek-flash', created=0)
    monkeypatch.setenv('ALGOEVOLVE_STRUCTURED_OUTPUT_MODE', mode)
    monkeypatch.setattr(backend, 'OpenAI', lambda **kwargs: object())
    def create(_client, params, **kwargs):
        calls.append(params)
        return response
    monkeypatch.setattr(backend, '_create_with_retry', create)
    monkeypatch.setattr(backend, 'log_llm_usage', lambda **kwargs: logs.append(kwargs))
    return cfg, calls, logs, response


def test_official_deepseek_uses_one_json_request_and_full_schema(monkeypatch, tmp_path):
    cfg, calls, logs, _ = prepare_backend(monkeypatch, tmp_path)
    result = backend.query('Review', '# Task description\ncontract', SPEC, cfg, model='deepseek-flash')
    assert result[0] == {'valid': True}
    assert len(calls) == len(logs) == 1
    assert calls[0]['response_format'] == {'type': 'json_object'}
    assert 'tools' not in calls[0]
    assert '"required":["valid"]' in calls[0]['messages'][1]['content']
    assert calls[0]['extra_body']['thinking']['type'] == 'enabled'


def test_explicit_json_override_prevents_tool_then_json_double_request(monkeypatch, tmp_path):
    cfg, calls, logs, _ = prepare_backend(monkeypatch, tmp_path, mode="tools")
    cfg.agent.code.base_url = "https://compatible.example/v1"

    result = backend.query(
        "Review",
        "contract",
        SPEC,
        cfg,
        model="deepseek-flash",
        structured_output_mode="json",
    )

    assert result[0] == {"valid": True}
    assert len(calls) == len(logs) == 1
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "tools" not in calls[0]


def test_valid_tool_mode_content_does_not_trigger_second_paid_call(monkeypatch, tmp_path):
    cfg, calls, logs, _ = prepare_backend(monkeypatch, tmp_path, mode='tools')
    assert backend.query('Review', 'contract', SPEC, cfg, model='deepseek-flash')[0] == {'valid': True}
    assert len(calls) == len(logs) == 1
    assert 'tools' in calls[0]


@pytest.mark.parametrize('content,finish', [('{"valid":"true"}', 'stop'), ('{}', 'stop'), ('{"valid":true}', 'length'), ('', 'stop')])
def test_invalid_json_fails_and_records_paid_usage(monkeypatch, tmp_path, content, finish):
    cfg, calls, logs, _ = prepare_backend(monkeypatch, tmp_path, content=content, finish=finish)
    with pytest.raises(Exception):
        backend.query('Review', 'contract', SPEC, cfg, model='deepseek-flash')
    assert len(calls) == len(logs) == 1
    assert logs[0]['parsed_ok'] is False


def test_third_party_retains_tools_by_default(monkeypatch):
    monkeypatch.delenv('ALGOEVOLVE_STRUCTURED_OUTPUT_MODE', raising=False)
    assert not backend._prefer_structured_json(NS(base_url='https://compatible.example/v1'), 'deepseek-flash')


def test_invalid_tool_content_and_failed_fallback_both_record_usage(monkeypatch, tmp_path):
    cfg, calls, logs, _ = prepare_backend(monkeypatch, tmp_path, mode='tools', content='{"valid":"true"}')
    with pytest.raises(Exception):
        backend.query('Review', 'contract', SPEC, cfg, model='deepseek-flash')
    assert len(calls) == len(logs) == 2
    assert all(row['parsed_ok'] is False for row in logs)


def test_tool_call_schema_is_validated_and_paid_usage_is_logged(monkeypatch, tmp_path):
    cfg, calls, logs, response = prepare_backend(monkeypatch, tmp_path, mode="tools")
    response.choices[0].message.tool_calls = [
        NS(function=NS(name="review", arguments='{"valid":"true"}'))
    ]

    with pytest.raises(Exception):
        backend.query("Review", "contract", SPEC, cfg, model="deepseek-flash")

    assert len(calls) == len(logs) == 1
    assert logs[0]["parsed_ok"] is False


def test_valid_tool_call_passes_schema_and_logs_usage(monkeypatch, tmp_path):
    cfg, calls, logs, response = prepare_backend(monkeypatch, tmp_path, mode="tools")
    response.choices[0].message.tool_calls = [
        NS(function=NS(name="review", arguments='{"valid":true}'))
    ]

    result = backend.query("Review", "contract", SPEC, cfg, model="deepseek-flash")

    assert result[0] == {"valid": True}
    assert len(calls) == len(logs) == 1
    assert logs[0]["parsed_ok"] is True
