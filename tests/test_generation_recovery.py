import json
from types import SimpleNamespace

import pytest

from agents.coder.recovery import recover_first_draft, text_sha256
from llm.openai import _prompt_to_messages


def fixture(tmp_path):
    prompt = {"system": "Implement the task.", "user": "Task rules and current draft profile."}
    stage = SimpleNamespace(model="gpt-5.6-luna", base_url="https://example.invalid/v1")
    agent = SimpleNamespace(
        cfg=SimpleNamespace(log_dir=tmp_path, runtime=SimpleNamespace(resume_run=True)),
        acfg=SimpleNamespace(code=stage), journal=[object()], next_branch_id=1,
        virtual_root=SimpleNamespace(id="root"), task_desc="Full task", data_preview="Full context",
    )
    response = 'Plan.```python\nreadme = """\n```bash\npython train.py\n```\n"""\nprint(readme)\n```'
    (tmp_path / "original.log").write_bytes(("prefix" + response + "\nnext").encode("utf-8"))
    messages = _prompt_to_messages(prompt, model=stage.model, base_url=stage.base_url)
    record = {
        "schema": "algoevolve.first_draft_response_recovery.v1", "parent_id": "root",
        "task_description_sha256": text_sha256(agent.task_desc),
        "data_context_sha256": text_sha256(agent.data_preview),
        "provider_prompt_signatures": [{"role": m["role"], "sha256_16": text_sha256(m["content"])[:16]} for m in messages],
        "source_log": "original.log", "response_offset": len("prefix"),
        "response_bytes": len(response.encode("utf-8")), "response_sha256": text_sha256(response),
    }
    (tmp_path / "draft_response_recovery.json").write_text(json.dumps(record), encoding="utf-8")
    return agent, prompt


def test_saved_provider_response_recovers_without_execution_or_generation(tmp_path):
    agent, prompt = fixture(tmp_path)
    plan, code = recover_first_draft(agent, prompt)
    assert plan == "Plan."
    assert "print(readme)" in code
    assert (tmp_path / "draft_response_recovery.applied.json").is_file()
    assert len(agent.journal) == 1
    # A further interruption before node materialization can still recover.
    assert recover_first_draft(agent, prompt) == (plan, code)
    agent.journal.append(object())
    assert recover_first_draft(agent, prompt) is None


@pytest.mark.parametrize("change", ["task", "prompt", "source", "root"])
def test_recovery_rejects_changed_inputs_and_provenance(tmp_path, change):
    agent, prompt = fixture(tmp_path)
    if change == "task":
        agent.task_desc += "New rule"
    elif change == "prompt":
        prompt["user"] += "Changed profile"
    elif change == "source":
        (tmp_path / "original.log").write_text("changed", encoding="utf-8")
    else:
        agent.virtual_root.id = "other-root"
    with pytest.raises(RuntimeError):
        recover_first_draft(agent, prompt)
    assert (tmp_path / "draft_response_recovery.json").is_file()


def test_new_search_does_not_consume_an_old_recovery(tmp_path):
    agent, prompt = fixture(tmp_path)
    agent.cfg.runtime.resume_run = False
    assert recover_first_draft(agent, prompt) is None


def test_recovery_allows_only_package_order_changes(tmp_path):
    agent, prompt = fixture(tmp_path)
    source = "Your solution can use any relevant machine learning packages such as: `pandas`, `numpy`. Feel free to use them."
    prompt["user"] = source
    messages = _prompt_to_messages(prompt, model=agent.acfg.code.model, base_url=agent.acfg.code.base_url)
    path = tmp_path / "draft_response_recovery.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["source_prompt_messages"] = messages
    record["provider_prompt_signatures"] = [{"role": m["role"], "sha256_16": text_sha256(m["content"])[:16]} for m in messages]
    path.write_text(json.dumps(record), encoding="utf-8")
    prompt["user"] = source.replace("`pandas`, `numpy`", "`numpy`, `pandas`")
    assert recover_first_draft(agent, prompt)[1]
    prompt["user"] += " New constraint."
    with pytest.raises(RuntimeError, match="Recovery prompt differs"):
        recover_first_draft(agent, prompt)


def test_environment_prompt_does_not_depend_on_or_advance_search_randomness():
    import random
    from agents.prompts.environment import get_prompt_environment

    state = random.getstate()
    first = get_prompt_environment()
    assert random.getstate() == state
    random.random()
    assert get_prompt_environment() == first


def test_environment_inventory_does_not_claim_missing_packages_are_installed(monkeypatch):
    from agents.prompts import environment

    def available(name):
        if name == "numpy":
            return "test-version"
        raise environment.PackageNotFoundError(name)

    environment._installed_packages.cache_clear()
    monkeypatch.setattr(environment, "version", available)
    try:
        prompt = environment.get_prompt_environment()["Installed Packages"]
        assert "numpy==test-version" in prompt
        assert "xgboost" not in prompt
        assert "all packages are already installed" not in prompt
        assert "does not grant GPU access" in prompt
    finally:
        environment._installed_packages.cache_clear()
