from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import pytest

from agents.coder import base_coder
from utils.response import extract_plan_and_code
from utils.response import extract_code


def test_embedded_documentation_fences_preserve_the_complete_script() -> None:
    source = 'readme = """Example:\n```bash\npython solution.py\n```\n```json\n{"ok": true}\n```\n"""\nprint(readme)'
    response = "Short plan.```python\n" + source + "\n```"
    code = extract_code(response)
    namespace = {}
    exec(compile(code, "<test>", "exec"), namespace)
    assert namespace["readme"] == 'Example:\n```bash\npython solution.py\n```\n```json\n{"ok": true}\n```\n'
    assert "print(readme)" in code


def test_invalid_outer_script_cannot_be_replaced_with_valid_nested_example() -> None:
    assert extract_code('```python\nreadme = """\n```python\nprint(1)\n```\n"""\nbroken =\n```') == ""


def test_crlf_and_long_outer_fence() -> None:
    code = extract_code('````python\r\ntext = "```"\r\nprint(text)\r\n````')
    assert 'text = "```"' in code
    assert 'print("```")' in extract_code('print("```")')


def test_failed_generation_does_not_pass_prose_to_the_code_reviewer(monkeypatch) -> None:
    monkeypatch.setattr(base_coder, "generate", lambda **kwargs: "No complete code was generated.")
    agent = SimpleNamespace(acfg=SimpleNamespace(code=SimpleNamespace(temp=0), retries=None), cfg=None)
    with pytest.raises(RuntimeError, match="Code extraction failed"):
        base_coder.plan_and_code_query(agent, "task", retries=1)


def test_full_rewrite_contract_keeps_complete_prompt_prefix_and_is_idempotent():
    original = {"system": "role", "user": "All task constraints and data", "assistant": "Complete original code"}
    patched = base_coder.complete_script_prompt(original)
    assert patched["user"].startswith(original["user"])
    assert patched["system"] == original["system"] and patched["assistant"] == original["assistant"]
    assert original["user"] == "All task constraints and data"
    assert base_coder.complete_script_prompt(patched) == patched
    assert base_coder.complete_script_prompt("task").startswith("task")


def test_multiple_code_blocks_are_combined_but_not_silently_dropped() -> None:
    assert "print(value)" in extract_code('```python\nvalue = 1\n```\nThen:\n```python\nprint(value)\n```')
    assert extract_code('```python\nvalue =\n```\n```python\nprint(1)\n```') == ""


def test_code_first_response_is_valid_without_prose_plan() -> None:
    plan, code = extract_plan_and_code(
        "```python\nprint('Final Validation Score: 1.0')\n```",
        default_plan="Default implementation plan.",
    )

    assert plan == "Default implementation plan."
    assert "Final Validation Score: 1.0" in code


def test_base_coder_does_not_regenerate_valid_code_first_response(monkeypatch) -> None:
    calls = 0

    def fake_generate(**kwargs):
        nonlocal calls
        calls += 1
        return "```python\nscore = 1.0\nprint(f'Final Validation Score: {score}')\n```"

    monkeypatch.setattr(base_coder, "generate", fake_generate)
    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            code=SimpleNamespace(temp=0.0),
            retries=SimpleNamespace(code_generation_extract_max_attempts=2),
        ),
        cfg=SimpleNamespace(),
    )

    plan, code = base_coder.plan_and_code_query(agent, {"user": "demo"})

    assert calls == 1
    assert plan
    assert "Final Validation Score" in code


def test_full_code_prompts_do_not_request_conflicting_long_or_fenced_plans() -> None:
    root = Path(__file__).resolve().parents[1]
    prompt_sources = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in [
            "agents/draft_agent.py",
            "agents/improve_agent.py",
            "agents/evolution_agent.py",
            "agents/fusion_agent.py",
        ]
    )

    assert "Natural length: around 8-12 sentences" not in prompt_sources
    assert "You MUST structure your plan using the following EXACT format" not in prompt_sources
