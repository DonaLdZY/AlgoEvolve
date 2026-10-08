"""Explicit recovery of a paid first-draft response after a parser failure."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from engine.expansion_profile import ExpansionProfile

from utils.response import extract_plan_and_code

logger = logging.getLogger("AlgoEvolve")


class GenerationRecoveryError(RuntimeError):
    """An explicit recovery mismatch requires intervention, not another search attempt."""


def _recovery_path(agent) -> Path | None:
    if not getattr(getattr(agent.cfg, "runtime", None), "resume_run", False):
        return None
    path = Path(agent.cfg.log_dir) / "draft_response_recovery.json"
    if path.is_file():
        return path
    applied = path.with_suffix(".applied.json")
    if applied.is_file() and len(agent.journal) == 1 and agent.next_branch_id == 1:
        return applied
    return None


def first_draft_recovery_profile(agent) -> ExpansionProfile | None:
    path = _recovery_path(agent)
    if path is None:
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return ExpansionProfile.from_payload(record.get("expansion_profile"))


def _stable_package_order(text: str) -> str:
    pattern = r"(Your solution can use any relevant machine learning packages such as: )([^\n]*?)(\. Feel free)"
    return re.sub(pattern, lambda m: m[1] + ", ".join(sorted(m[2].split(", "), key=lambda name: name.strip("`"))) + m[3], text)


def _normalize_environment_update(text: str) -> str:
    from agents.prompts.environment import get_prompt_environment

    legacy_names = ["numpy", "pandas", "scikit-learn", "statsmodels", "xgboost", "lightGBM", "torch", "torchvision", "torch-geometric", "bayesian-optimization", "timm", "transformers", "sentence-transformers", "opencv-python", "Pillow"]
    legacy = (
        "Your solution can use any relevant machine learning packages such as: "
        + ", ".join(f"`{name}`" for name in sorted(legacy_names))
        + ". Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
    )
    text = _stable_package_order(text)
    return text.replace(legacy, get_prompt_environment()["Installed Packages"])


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recover_first_draft(agent, prompt) -> tuple[str, str] | None:
    path = _recovery_path(agent)
    if path is None:
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if len(agent.journal) != 1 or agent.next_branch_id != 1:
        raise GenerationRecoveryError("First-draft recovery requires an empty search tree")
    expected = {
        "schema": "algoevolve.first_draft_response_recovery.v1",
        "parent_id": agent.virtual_root.id,
        "task_description_sha256": text_sha256(agent.task_desc),
        "data_context_sha256": text_sha256(agent.data_preview),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise GenerationRecoveryError("Recovered draft does not match this task and search root")
    from llm.openai import _prompt_to_messages

    stage = agent.acfg.code
    messages = _prompt_to_messages(prompt, model=stage.model, base_url=stage.base_url)
    signatures = [{"role": item["role"], "sha256_16": text_sha256(item["content"])[:16]}
                  for item in messages]
    if signatures != record.get("provider_prompt_signatures"):
        original = record.get("source_prompt_messages", [])
        original_signatures = [{"role": item["role"], "sha256_16": text_sha256(item["content"])[:16]} for item in original]
        canonical = lambda items: [{**item, "content": _normalize_environment_update(item["content"])} for item in items]
        if original_signatures != record.get("provider_prompt_signatures") or canonical(original) != canonical(messages):
            raise GenerationRecoveryError("Recovery prompt differs from the actual original provider request")
    source = (path.parent / record["source_log"]).resolve()
    if not source.is_relative_to(path.parent.resolve()):
        raise GenerationRecoveryError("Recovery source must be in the current search log directory")
    with source.open("rb") as stream:
        stream.seek(int(record["response_offset"]))
        response_bytes = stream.read(int(record["response_bytes"]))
    if hashlib.sha256(response_bytes).hexdigest() != record["response_sha256"]:
        raise GenerationRecoveryError("The saved provider response has changed")
    plan, code = extract_plan_and_code(
        response_bytes.decode("utf-8"), "Recover the original provider-generated draft."
    )
    if not code:
        raise GenerationRecoveryError("The saved provider draft still cannot be parsed")
    logger.info(
        "Recovered original first-draft provider response from %s at byte %s; "
        "normal preflight, review, execution and validation still apply.",
        source.name, record["response_offset"],
    )
    if not path.name.endswith(".applied.json"):
        path.replace(path.with_suffix(".applied.json"))
    return plan, code
