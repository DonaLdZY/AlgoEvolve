"""OpenAI-compatible API backend (query + generate). Supports Qwen API and any OpenAI-compatible endpoint."""

import hashlib
import json
import logging
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI
import jsonschema

from config import Config
from .gemini import FunctionSpec, compile_prompt_to_md
from .model_profiles import (
    get_profile,
    is_deepseek_model,
    supports_json_schema,
    thinking_json_incompatible,
)
from .usage import (
    estimate_text_tokens,
    infer_prompt_name,
    log_llm_usage,
    prompt_parts_from_messages,
    usage_to_dict,
)

logger = logging.getLogger("AlgoEvolve")
NETWORK_RETRY_MAX_ATTEMPTS = 5
NETWORK_RETRY_BASE_SLEEP_SECONDS = 5.0
NETWORK_RETRY_MAX_SLEEP_SECONDS = 30.0
MAX_CONTINUATION_ROUNDS = 2
CONTINUATION_OVERLAP_SCAN_CHARS = 4096
CACHE_FRIENDLY_SYSTEM = (
    "You are AlgoEvolve, an automated ML/RL coding agent. Follow the task/data "
    "context and the agent-specific instructions in the user messages. "
    "When a JSON/tool schema is supplied, return data that satisfies it exactly."
)
AGENT_INSTRUCTIONS_TITLE = "# Agent/System Instructions"
TASK_CONTEXT_END_MARKER = "# End stable task/data context"


def _strip_visible_thinking(text: str) -> str:
    """Remove a leading provider thinking block without touching literals in code."""
    thinking = re.match(r"\s*<think>.*?</think>", text, re.S)
    if thinking:
        return text[thinking.end():]
    return text


def _finish_reason_is_length(finish_reason: str | None) -> bool:
    return str(finish_reason or "").strip().lower() in {"length", "max_tokens", "max_output_tokens"}


def _normalize_stream_completion(text: str, finish_reason: str) -> tuple[str, str]:
    # Some gateways append an error as answer text and incorrectly report stop.
    marker = re.search(
        r"(?m)^>[ \t]*(?:\u274c[ \t]*)?(?:\u7f51\u7edc\u51fa\u73b0\u5f02\u5e38|Network error occurred)[ \t\r\n]*\Z",
        text,
    )
    if marker:
        return text[:marker.start()], "provider_interrupted"
    return text, finish_reason


def _needs_continuation(finish_reason: str) -> bool:
    return _finish_reason_is_length(finish_reason) or finish_reason == "provider_interrupted"


def _finish_reason_is_resource_limited(finish_reason: str | None) -> bool:
    return str(finish_reason or "").strip().lower() == "insufficient_system_resource"


def _append_with_overlap(
    base: str,
    addition: str,
    max_scan_chars: int = CONTINUATION_OVERLAP_SCAN_CHARS,
) -> str:
    """Append continuation text while removing exact repeated overlap at the join."""
    if not base or not addition:
        return f"{base}{addition}"
    max_scan = min(len(base), len(addition), max(0, int(max_scan_chars)))
    minimum_overlap = 6 if (
        (base[-1].isalnum() or base[-1] == "_")
        and (addition[0].isalnum() or addition[0] == "_")
    ) else 16
    for size in range(max_scan, minimum_overlap - 1, -1):
        if base[-size:] == addition[:size]:
            return base + addition[size:]
    return base + addition


def _build_continuation_instruction(max_tokens: int | None, round_index: int, reason: str = "length") -> str:
    token_text = f"{max_tokens}" if max_tokens else "the provider/API default"
    return (
        ("Your previous assistant response was interrupted by the provider connection. "
         if reason == "provider_interrupted" else
         "Your previous assistant response was cut off because it reached the output token limit. ")
        + "Continue exactly from the point where the previous assistant message stopped.\n"
        "- Do not repeat any previous text.\n"
        "- Do not restart the answer.\n"
        "- Start with the next character or token that should follow the previous assistant message.\n"
        "- Preserve all required formatting, markdown fences, JSON/text/code structure, and indentation.\n"
        f"- This is continuation round {round_index}; the output cap for this continuation is {token_text} tokens."
    )


def _build_continuation_messages(
    original_messages: list[dict[str, Any]],
    raw_assistant_so_far: str,
    *,
    max_tokens: int | None,
    round_index: int,
    reason: str = "length",
) -> list[dict[str, Any]]:
    """Replay the original prompt plus the partial assistant output, then ask for only the tail."""
    messages = [dict(m) for m in original_messages]
    if messages and messages[-1].get("role") == "assistant":
        messages[-1]["content"] = str(messages[-1].get("content", "") or "") + raw_assistant_so_far
        messages[-1].pop("prefix", None)
        messages[-1].pop("reasoning_content", None)
    else:
        messages.append({"role": "assistant", "content": raw_assistant_so_far})
    messages.append(
        {
            "role": "user",
            "content": _build_continuation_instruction(max_tokens=max_tokens, round_index=round_index, reason=reason),
        }
    )
    return messages


def _strip_markdown_fences(args: str) -> str:
    """Remove markdown code fences that LLMs sometimes append inside JSON string values."""
    cleaned = re.sub(r'\\n```[a-z]*\s*("?\s*\}?\s*)$', r'\1', args.rstrip())
    cleaned = cleaned.rstrip()
    if not cleaned.endswith('}'):
        if not cleaned.endswith('"'):
            cleaned += '"'
        cleaned += '}'
    return cleaned


def _parse_json_args(args: str) -> dict:
    """Parse function call arguments, tolerating Python literals and markdown fences."""
    # 1. Fast path: valid JSON as-is
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        pass

    # 2. Try stripping markdown fences
    try:
        cleaned = _strip_markdown_fences(args)
        if cleaned != args:
            result = json.loads(cleaned)
            logger.warning("Fixed malformed function args by stripping markdown code fences")
            return result
    except json.JSONDecodeError:
        pass

    # 3. Normalize Python literals (None/True/False) outside quoted strings
    parts = re.split(r'("(?:[^"\\]|\\.)*")', args)
    normalized = []
    for part in parts:
        if part.startswith('"'):
            normalized.append(part)
        else:
            part = re.sub(r'\bNone\b', 'null', part)
            part = re.sub(r'\bTrue\b', 'true', part)
            part = re.sub(r'\bFalse\b', 'false', part)
            normalized.append(part)
    normalized_str = ''.join(normalized)

    try:
        return json.loads(normalized_str)
    except json.JSONDecodeError:
        pass

    # 4. Normalized + strip markdown fences
    cleaned = _strip_markdown_fences(normalized_str)
    return json.loads(cleaned)


def _strip_json_fences(text: str) -> str:
    """Extract JSON text from markdown code fences when present."""
    stripped = (text or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    return fenced.group(1).strip() if fenced else stripped


def _example_from_schema(schema: dict | None) -> Any:
    """Build a minimal example JSON object from a schema for JSON-mode prompting."""
    schema = schema or {}
    if "anyOf" in schema and schema["anyOf"]:
        return _example_from_schema(schema["anyOf"][0])

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if "null" in schema_type:
            return None
        schema_type = schema_type[0] if schema_type else None
    if schema_type == "null":
        return None
    if schema_type == "object":
        properties = schema.get("properties", {})
        return {key: _example_from_schema(value) for key, value in properties.items()}
    if schema_type == "array":
        items = schema.get("items", {})
        return [_example_from_schema(items)]
    if schema_type == "boolean":
        return False
    if schema_type in {"number", "integer"}:
        return 0
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    return ""


def _looks_like_task_prefixed_user(user_message: str | None) -> bool:
    return bool(str(user_message or "").lstrip().startswith("# Task description"))


def _stable_task_section_end(user: str) -> int:
    marker_index = user.find(TASK_CONTEXT_END_MARKER)
    if marker_index >= 0:
        return marker_index + len(TASK_CONTEXT_END_MARKER)
    stripped_offset = len(user) - len(user.lstrip())
    search_start = stripped_offset + len("# Task description")
    match = re.search(r"\n# (?!#)", user[search_start:])
    return search_start + match.start() if match else len(user)


def _insert_after_task_section(user_message: str, section: str) -> str:
    """Put changing stage instructions after the stable task section.

    Provider-side caches reuse only exact request prefixes. If every agent uses
    a different system message before the task context, the large task context
    cannot be shared across agents. This keeps a short stable system message and
    makes the first large user-message prefix the same across calls.
    """
    user = str(user_message or "")
    inserted = f"{AGENT_INSTRUCTIONS_TITLE}\n{str(section or '').strip()}"
    if not inserted.strip():
        return user

    # The AutoRealize context uses mostly "##" headings. The next top-level
    # section ("# Instructions", "# Implementation", etc.) is the safe point
    # where the stage-specific instructions can diverge.
    split_at = _stable_task_section_end(user)
    # Use a generated length, not a marker search through untrusted code/logs.
    candidate = re.match(r"\s*# Candidate implementation\n<!-- candidate-code-chars: (\d+) -->\n", user[split_at:])
    if candidate:
        code_end = split_at + candidate.end() + int(candidate.group(1))
        boundary = "\n\n# End candidate implementation"
        if user[code_end:].startswith(boundary + "\n"):
            split_at = code_end + len(boundary)
    if split_at == len(user):
        return f"{user.rstrip()}\n\n{inserted}"
    return f"{user[:split_at].rstrip()}\n\n{inserted}\n\n{user[split_at:].lstrip()}"


def _cache_friendly_messages(system_message: str | None, user_message: str | None) -> list[dict[str, str]] | None:
    if not system_message or not user_message or not _looks_like_task_prefixed_user(user_message):
        return None
    return [
        {"role": "system", "content": CACHE_FRIENDLY_SYSTEM},
        {"role": "user", "content": _insert_after_task_section(user_message, system_message)},
    ]


def _build_json_mode_messages(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec,
) -> list[dict[str, str]]:
    """Build a JSON-output fallback prompt for providers with weak tool-calling support."""
    example = json.dumps(_example_from_schema(func_spec.json_schema), ensure_ascii=False, separators=(",", ":"))
    schema = json.dumps(func_spec.json_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    instruction = (
        f"You must respond with a valid JSON object for `{func_spec.name}`.\n"
        f"Return JSON only, with no markdown fences and no extra commentary.\n"
        f"JSON schema (all constraints and descriptions apply):\n{schema}\n"
        f"JSON example (shape only, not factual values):\n{example}"
    )
    combined_system = f"{system_message}\n\n{instruction}" if system_message else instruction
    cache_messages = _cache_friendly_messages(combined_system, user_message)
    if cache_messages is not None:
        return cache_messages
    return _build_messages(combined_system, user_message)


def _parse_message_json_content(content: str) -> dict:
    """Parse JSON object content from a model response."""
    text = _strip_json_fences(content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _prefer_structured_json(stage, model: str, mode_override: str | None = None) -> bool:
    mode = str(
        mode_override
        if mode_override is not None
        else os.environ.get("ALGOEVOLVE_STRUCTURED_OUTPUT_MODE", "auto")
    ).strip().lower()
    if mode not in {"auto", "json", "tools"}:
        raise ValueError("ALGOEVOLVE_STRUCTURED_OUTPUT_MODE must be auto, json, or tools")
    official = (urlparse(str(getattr(stage, "base_url", "") or "")).hostname or "").lower() == "api.deepseek.com"
    return mode == "json" or (mode == "auto" and official and is_deepseek_model(model))


def _validated_json_content(content: str, func_spec: FunctionSpec, finish_reason: str) -> dict:
    if finish_reason != "stop":
        raise ValueError(f"Structured content did not finish normally: {finish_reason}")
    output = _parse_message_json_content(content)
    jsonschema.validate(output, func_spec.json_schema)
    return output

# Return type aligned with gemini.query
OutputType = str | dict


def _stage_config_for_model(cfg: Config, model: str, stage_name: str | None = None):
    """Return code or feedback config depending on which model is being used."""
    stage_name = str(stage_name or "").strip().lower()
    if stage_name == "code":
        return cfg.agent.code
    if stage_name == "feedback":
        return cfg.agent.feedback
    if cfg.agent.code.model == model:
        return cfg.agent.code
    return cfg.agent.feedback


def _stage_max_tokens(stage) -> int | None:
    try:
        tokens = int(getattr(stage, "max_tokens", 0) or 0)
    except (TypeError, ValueError):
        return None
    return tokens if tokens > 0 else None


def _stage_minimum_output_tokens(stage) -> int:
    try:
        tokens = int(getattr(stage, "minimum_output_tokens", 0) or 0)
    except (TypeError, ValueError):
        tokens = 0
    return max(0, tokens)


def _resolve_max_tokens(explicit_value: Any, stage) -> int | None:
    try:
        explicit = int(explicit_value) if explicit_value is not None else 0
    except (TypeError, ValueError):
        explicit = 0
    if explicit > 0:
        return explicit
    effective = max(int(_stage_max_tokens(stage) or 0), _stage_minimum_output_tokens(stage))
    return effective or None


def _resolve_use_thinking(stage, func_spec: FunctionSpec | None, json_schema: dict | None = None) -> bool | None:
    """Resolve thinking override for this request.

    None means provider default: do not inject thinking/enable_thinking controls.
    """
    forced = getattr(stage, "enable_thinking", None)
    if forced is not None:
        return bool(forced)
    return None


def _build_messages(system_message: str | None, user_message: str | None) -> list[dict[str, str]]:
    cache_messages = _cache_friendly_messages(system_message, user_message)
    if cache_messages is not None:
        return cache_messages
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def _build_tool_dict(model: str, func_spec: FunctionSpec) -> dict[str, Any]:
    """Return a provider-compatible tool schema."""
    if is_deepseek_model(model):
        return func_spec.to_openai_tool_dict(strict=False)

    tool_dict = func_spec.to_openai_tool_dict(strict=True)
    if not supports_json_schema(model):
        tool_dict.get("function", {}).pop("strict", None)
    return tool_dict


def _normalize_deepseek_reasoning_effort(effort: str | None) -> str | None:
    """Map config effort names to the values DeepSeek actually applies."""
    if not effort:
        return None
    normalized = str(effort).strip().lower()
    mapping = {
        "low": "high",
        "medium": "high",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    }
    return mapping.get(normalized, normalized)


def _apply_deepseek_request_options(
    params: dict[str, Any],
    stage,
    model: str,
    use_thinking: bool | None,
) -> None:
    """Apply configured effort and compatible provider routing options."""
    cache_mode = os.environ.get("ALGOEVOLVE_PROMPT_CACHE_KEY_MODE", "auto").strip().lower()
    official = (urlparse(str(getattr(stage, "base_url", "") or "")).hostname or "").lower() == "api.openai.com"
    messages = params.get("messages", [])
    if (cache_mode == "enabled" or (cache_mode == "auto" and official)) and _is_cache_friendly_message_layout(messages):
        user = str(messages[1]["content"])
        prefix = user[:_stable_task_section_end(user)].rstrip()
        params["prompt_cache_key"] = "algoevolve:" + hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:32]
    if not is_deepseek_model(model):
        effort = str(getattr(stage, "reasoning_effort", None) or "").strip().lower()
        if effort and effort not in {"default", "none", "null"}:
            params["reasoning_effort"] = effort
            if str(model).lower().startswith(("gpt-5", "o1", "o3", "o4")):
                params.pop("temperature", None)
        return
    thinking_active = use_thinking is not False
    if thinking_active:
        for ignored_parameter in ["temperature", "top_p", "presence_penalty", "frequency_penalty"]:
            params.pop(ignored_parameter, None)
        effort = getattr(stage, "reasoning_effort", None)
        if effort:
            params["reasoning_effort"] = _normalize_deepseek_reasoning_effort(effort)
    if use_thinking is not None:
        body = dict(params.get("extra_body") or {})
        body["thinking"] = {"type": "enabled" if use_thinking else "disabled"}
        params["extra_body"] = body


def _is_retryable_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if any(
        key in name
        for key in [
            "timeout",
            "connection",
            "ratelimit",
            "internalserver",
            "apierror",
            "apiconnection",
            "badgateway",
            "serviceunavailable",
        ]
    ):
        return True
    if any(
        key in msg
        for key in [
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
                "connection refused",
                "10061",
                "actively refused",
                "积极拒绝",
                "temporary failure",
                "temporarily unavailable",
                "insufficient_system_resource",
                "bad gateway",
            "502",
            "503",
            "504",
                "rate limit",
                "too many requests",
                "getaddrinfo",
                "11001",
                "name resolution",
                "name or service not known",
            ]
        ):
        return True
    return False


def _create_with_retry(
    client: OpenAI,
    params: dict[str, Any],
    *,
    label: str,
    stage=None,
):
    window = int(getattr(stage, "context_window_tokens", 0) or 0)
    if window:
        input_tokens = estimate_text_tokens(json.dumps(
            {key: params[key] for key in ("messages", "tools", "response_format") if key in params},
            ensure_ascii=False, default=str,
        ))
        remaining = window - input_tokens - int(window * 0.18)
        requested = int(params.get("max_tokens") or 32768)
        minimum_reply = max(1, int(getattr(stage, "context_minimum_reply_tokens", 8192) or 8192))
        if remaining < min(requested, minimum_reply):
            raise ValueError(f"Complete request exceeds context budget: input={input_tokens}, window={window}")
        if remaining < requested:
            logger.info("Keeping complete request (%s estimated input tokens); reply budget %s -> %s with 18%% headroom",
                        input_tokens, requested, remaining)
        params["max_tokens"] = min(requested, remaining)
    max_attempts = max(
        1,
        int(getattr(stage, "network_retry_max_attempts", NETWORK_RETRY_MAX_ATTEMPTS)),
    )
    base_sleep = max(
        0.0,
        float(getattr(stage, "network_retry_base_sleep_seconds", NETWORK_RETRY_BASE_SLEEP_SECONDS)),
    )
    max_sleep = max(
        0.0,
        float(getattr(stage, "network_retry_max_sleep_seconds", NETWORK_RETRY_MAX_SLEEP_SECONDS)),
    )
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            try:
                completion = client.chat.completions.create(**params)
            except Exception as exc:
                message = str(exc).lower()
                if "prompt_cache_key" not in params or "prompt_cache_key" not in message or not any(
                    word in message for word in ("unsupported", "unknown", "unexpected", "unrecognized", "not permitted")
                ):
                    raise
                params = {key: value for key, value in params.items() if key != "prompt_cache_key"}
                logger.warning("%s provider rejected cache routing key; retrying without it", label)
                completion = client.chat.completions.create(**params)
            choices = getattr(completion, "choices", None) or []
            finish_reason = str(getattr(choices[0], "finish_reason", "") or "") if choices else ""
            if is_deepseek_model(str(params.get("model", ""))) and _finish_reason_is_resource_limited(
                finish_reason
            ):
                raise RuntimeError("DeepSeek returned finish_reason=insufficient_system_resource")
            return completion
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_error(exc)
            logger.warning(
                "%s call failed (attempt %s/%s, retryable=%s): %s",
                label,
                attempt,
                max_attempts,
                retryable,
                exc,
            )
            if (not retryable) or attempt >= max_attempts:
                raise
            sleep_secs = min(max_sleep, base_sleep * attempt)
            logger.warning(
                "%s network error; reconnecting after %.1fs (attempt %s/%s)",
                label,
                sleep_secs,
                attempt,
                max_attempts,
            )
            time.sleep(sleep_secs)
    if last_exc is not None:
        raise last_exc


def _stream_create_with_usage_fallback(
    client: OpenAI,
    params: dict[str, Any],
    *,
    label: str,
    stage=None,
):
    """Request streaming usage when supported; retry safely without it otherwise."""
    stream_params = dict(params)
    stream_params.setdefault("stream_options", {"include_usage": True})
    try:
        return _create_with_retry(client, stream_params, label=label, stage=stage)
    except Exception as exc:
        msg = str(exc).lower()
        unsupported = any(
            key in msg
            for key in [
                "stream_options",
                "include_usage",
                "extra inputs are not permitted",
                "unknown parameter",
                "unrecognized request argument",
                "unsupported parameter",
            ]
        )
        if not unsupported:
            raise
        logger.warning("%s provider rejected stream_options.include_usage; retrying stream without usage metadata", label)
        return _create_with_retry(
            client,
            params,
            label=f"{label}_without_usage",
            stage=stage,
        )


def _collect_stream_response(
    client: OpenAI,
    params: dict[str, Any],
    *,
    label: str,
    stage=None,
) -> tuple[str, str, Any, float]:
    """Run one streaming request and return raw text, finish reason, usage, and elapsed seconds."""
    t0 = time.time()
    stream = _stream_create_with_usage_fallback(
        client,
        params,
        label=label,
        stage=stage,
    )
    full_text = ""
    final_usage: Any = None
    final_finish_reason = ""
    chunks = 0
    last_progress = time.monotonic()
    for chunk in stream:
        chunks += 1
        if chunk.choices and chunk.choices[0].delta.content:
            full_text += chunk.choices[0].delta.content
        if getattr(chunk, "usage", None) is not None:
            final_usage = getattr(chunk, "usage", None)
        if chunk.choices and getattr(chunk.choices[0], "finish_reason", None):
            final_finish_reason = str(chunk.choices[0].finish_reason or "")
        if time.monotonic() - last_progress >= 30:
            logger.info("%s stream progress: chunks=%s output_chars=%s elapsed=%.1fs", label, chunks, len(full_text), time.time() - t0)
            last_progress = time.monotonic()
    if is_deepseek_model(str(params.get("model", ""))) and _finish_reason_is_resource_limited(
        final_finish_reason
    ):
        raise RuntimeError("DeepSeek returned finish_reason=insufficient_system_resource")
    return full_text, final_finish_reason, final_usage, time.time() - t0


def _message_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def _is_cache_friendly_message_layout(messages: list[dict[str, str]]) -> bool:
    return (
        len(messages) >= 2
        and messages[0].get("role") == "system"
        and messages[0].get("content") == CACHE_FRIENDLY_SYSTEM
        and _looks_like_task_prefixed_user(messages[1].get("content", ""))
    )


def _log_usage(label: str, completion: Any) -> None:
    usage = usage_to_dict(getattr(completion, "usage", None))
    if not usage:
        return
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    cached = next(
        (
            pool[key]
            for pool in (usage, details)
            for key in ("prompt_cache_hit_tokens", "cached_tokens")
            if pool.get(key) is not None
        ),
        None,
    )
    missed = next(
        (
            pool[key]
            for pool in (usage, details)
            for key in ("prompt_cache_miss_tokens", "cache_miss_input_tokens")
            if pool.get(key) is not None
        ),
        None,
    )
    logger.info(
        "[llm_usage] %s prompt=%s cached=%s miss=%s completion=%s total=%s",
        label,
        usage.get("prompt_tokens"),
        cached if cached is not None else "unknown",
        missed if missed is not None else "unknown",
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """OpenAI-compatible query (chat completions, optional function calling). Same return shape as gemini.query."""
    if cfg is None:
        raise ValueError("cfg is required for OpenAI backend")
    filtered = {k: v for k, v in model_kwargs.items() if v is not None}
    model = filtered.get("model", "")
    stage = _stage_config_for_model(cfg, model, filtered.get("stage_name"))
    client = OpenAI(
        api_key=stage.api_key,
        base_url=stage.base_url or None,
        max_retries=0,
        timeout=max(1.0, float(getattr(stage, "request_timeout_seconds", 1200.0))),
    )
    json_mode = func_spec is not None and _prefer_structured_json(
        stage,
        model,
        filtered.get("structured_output_mode"),
    )
    messages = (_build_json_mode_messages(system_message, user_message, func_spec)
                if json_mode else _build_messages(system_message, user_message))
    if not messages:
        raise ValueError("Either system_message or user_message must be provided")
    logger.info(
        "query messages: %s turns, chars=%s, cache_friendly=%s",
        len(messages),
        _message_chars(messages),
        _is_cache_friendly_message_layout(messages),
        extra={"verbose": True},
    )

    # Function calling requires non_thinking mode for some providers, otherwise they may
    # reject required/object tool_choice in thinking mode.
    use_thinking = _resolve_use_thinking(stage, func_spec=func_spec)
    profile = get_profile(model, use_thinking=bool(use_thinking)) if use_thinking is not None else {}
    deepseek_model = is_deepseek_model(model)

    extra_body: dict[str, Any] = {}
    if "top_k" in profile:
        extra_body["top_k"] = profile["top_k"]
    if use_thinking is not None and "enable_thinking" in profile:
        extra_body["enable_thinking"] = profile["enable_thinking"]
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": profile.get("temperature", filtered.get("temperature", 1.0)),
    }
    resolved_max_tokens = _resolve_max_tokens(filtered.get("max_tokens"), stage)
    if resolved_max_tokens is not None:
        params["max_tokens"] = resolved_max_tokens
    if "top_p" in profile:
        params["top_p"] = profile["top_p"]
    if "presence_penalty" in profile:
        params["presence_penalty"] = profile["presence_penalty"]
    if extra_body:
        params["extra_body"] = extra_body
    _apply_deepseek_request_options(params, stage, model, use_thinking)
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    elif func_spec is not None:
        tool_dict = _build_tool_dict(model, func_spec)
        params["tools"] = [tool_dict]
        if not deepseek_model:
            params["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    prompt_name = str(filtered.get("prompt_name") or infer_prompt_name("query"))
    stage_id = str(filtered.get("stage_id") or "").strip() or None
    prompt_parts = prompt_parts_from_messages(messages)
    if func_spec is not None and not json_mode:
        prompt_parts.append(
            {
                "name": "function_schema",
                "role": "tool_schema",
                "content": json.dumps(func_spec.to_dict(), ensure_ascii=False, default=str),
            }
        )
    logger.info(f"Querying OpenAI-compatible API with model: {model}")
    try:
        completion = _create_with_retry(client, params, label="query", stage=stage)
    except Exception as e:
        logger.error(f"Error calling OpenAI-compatible API: {e}")
        raise
    req_time = time.time() - t0
    choice = completion.choices[0]
    message = choice.message
    finish_reason = str(getattr(choice, "finish_reason", "") or "")

    if getattr(choice, "finish_reason", None) == "length":
        logger.warning(f"Response truncated by max_tokens ({params.get('max_tokens')}), consider increasing it")

    if json_mode:
        output = None
        try:
            output = _validated_json_content(message.content or "", func_spec, finish_reason)
        finally:
            # Even malformed/empty/truncated paid responses must enter the ledger.
            log_llm_usage(cfg=cfg, prompt_name=prompt_name, stage_id=stage_id, mode="query_json",
                provider="openai_compatible", model=model, response=completion, seconds=req_time,
                finish_reason=finish_reason, max_tokens=params.get("max_tokens"), parsed_ok=output is not None,
                prompt_parts=prompt_parts, estimated_completion_text=message.content or "")
    elif func_spec is None:
        output = message.content or ""
        logger.info(f"OpenAI response: {output}", extra={"verbose": True})
        log_llm_usage(
            cfg=cfg,
            prompt_name=prompt_name,
            stage_id=stage_id,
            mode="query_text",
            provider="openai_compatible",
            model=model,
            response=completion,
            seconds=req_time,
            finish_reason=finish_reason,
            max_tokens=params.get("max_tokens"),
            parsed_ok=True,
            prompt_parts=prompt_parts,
            estimated_completion_text=output,
        )
    else:
        if message.tool_calls:
            tc = message.tool_calls[0]
            parse_error = None
            output = None
            try:
                if tc.function.name != func_spec.name:
                    raise ValueError(
                        f"Function name mismatch: expected {func_spec.name}, got {tc.function.name}"
                    )
                output = _parse_json_args(tc.function.arguments or "{}")
                jsonschema.validate(output, func_spec.json_schema)
            except Exception as exc:
                parse_error = exc
            log_llm_usage(
                cfg=cfg,
                prompt_name=prompt_name,
                stage_id=stage_id,
                mode="query_function",
                provider="openai_compatible",
                model=model,
                response=completion,
                seconds=req_time,
                finish_reason=finish_reason,
                max_tokens=params.get("max_tokens"),
                parsed_ok=parse_error is None,
                prompt_parts=prompt_parts,
                estimated_completion_text=(
                    json.dumps(output, ensure_ascii=False, default=str)
                    if output is not None else str(getattr(tc.function, "arguments", "") or "")
                ),
            )
            if parse_error is not None:
                logger.error("Invalid function response: %s", parse_error)
                raise parse_error
            logger.info(f"OpenAI function call response: {output}", extra={"verbose": True})
        elif deepseek_model and message.content:
            # Compatible endpoints may return a complete structured answer as
            # content. Validate it locally before paying for a second request.
            try:
                output = _validated_json_content(message.content, func_spec, finish_reason)
            except (ValueError, jsonschema.ValidationError):
                output = None
            if output is not None:
                log_llm_usage(cfg=cfg, prompt_name=prompt_name, stage_id=stage_id, mode="query_function_content",
                    provider="openai_compatible", model=model, response=completion, seconds=req_time,
                    finish_reason=finish_reason, max_tokens=params.get("max_tokens"), parsed_ok=True,
                    prompt_parts=prompt_parts, estimated_completion_text=message.content)
                return (output, req_time, getattr(completion.usage, "prompt_tokens", 0) or 0,
                        getattr(completion.usage, "completion_tokens", 0) or 0,
                        {"model": getattr(completion, "model", model),
                         "created": getattr(completion, "created", int(time.time())),
                         "usage": usage_to_dict(getattr(completion, "usage", None))})
            logger.warning(
                "DeepSeek returned message content instead of tool_calls; falling back to JSON output parsing"
            )
            log_llm_usage(
                cfg=cfg,
                prompt_name=f"{prompt_name}:tool_call_failed_content",
                stage_id=stage_id,
                mode="query_function_unparsed",
                provider="openai_compatible",
                model=model,
                response=completion,
                seconds=req_time,
                finish_reason=finish_reason,
                max_tokens=params.get("max_tokens"),
                parsed_ok=False,
                prompt_parts=prompt_parts,
                estimated_completion_text=message.content or "",
            )
            json_params: dict[str, Any] = {
                "model": model,
                "messages": _build_json_mode_messages(system_message, user_message, func_spec),
                "response_format": {"type": "json_object"},
            }
            for key in [
                "temperature",
                "max_tokens",
                "top_p",
                "presence_penalty",
                "reasoning_effort",
                "extra_body",
            ]:
                if key in params:
                    json_params[key] = params[key]

            fallback_t0 = time.time()
            completion = _create_with_retry(
                client,
                json_params,
                label="query_json_fallback",
                stage=stage,
            )
            fallback_seconds = time.time() - fallback_t0
            message = completion.choices[0].message
            fallback_choice = completion.choices[0]
            fallback_finish = str(getattr(fallback_choice, "finish_reason", "") or "")
            output = None
            try:
                output = _validated_json_content(message.content or "", func_spec, fallback_finish)
            finally:
                log_llm_usage(cfg=cfg, prompt_name=f"{prompt_name}:json_fallback", stage_id=stage_id, mode="query_json_fallback",
                    provider="openai_compatible", model=model, response=completion, seconds=fallback_seconds,
                    finish_reason=fallback_finish, max_tokens=json_params.get("max_tokens"), parsed_ok=output is not None,
                    prompt_parts=prompt_parts_from_messages(json_params.get("messages", [])),
                    estimated_completion_text=message.content or "")
            in_tok = getattr(completion.usage, "prompt_tokens", 0) or 0
            out_tok = getattr(completion.usage, "completion_tokens", 0) or 0
            info = {
                "model": getattr(completion, "model", model),
                "created": getattr(completion, "created", int(time.time())),
                "usage": usage_to_dict(getattr(completion, "usage", None)),
            }
            return output, req_time, in_tok, out_tok, info
        else:
            raise ValueError("Expected function call, got no tool_calls")

    in_tok = getattr(completion.usage, "prompt_tokens", 0) or 0
    out_tok = getattr(completion.usage, "completion_tokens", 0) or 0
    info = {
        "model": getattr(completion, "model", model),
        "created": getattr(completion, "created", int(time.time())),
        "usage": usage_to_dict(getattr(completion, "usage", None)),
    }
    return output, req_time, in_tok, out_tok, info


def _is_deepseek_beta_base_url(base_url: str) -> bool:
    try:
        return urlparse(str(base_url or "").strip()).path.rstrip("/").endswith("/beta")
    except ValueError:
        return False


def _prompt_to_messages(
    prompt: str | dict | list,
    model: str = "",
    base_url: str = "",
    *,
    structured_output: bool = False,
) -> list[dict[str, Any]]:
    """Convert prompt to chat messages. Supports Qwen/OpenAI chat format: {system, user, assistant}.

    For GPT models, assistant content is appended to the user message instead of
    being sent as a separate assistant message, because GPT models may return
    empty responses when they see a trailing assistant prefill.
    """
    if isinstance(prompt, list) and all(
        isinstance(message, dict) and message.get("role") in {"system", "user", "assistant"}
        for message in prompt
    ):
        messages: list[dict[str, Any]] = [
            {
                "role": str(message["role"]),
                "content": str(message.get("content", "") or ""),
            }
            for message in prompt
            if str(message.get("content", "") or "").strip()
        ]
        if not messages:
            raise ValueError("Chat message list must contain at least one non-empty message")
        if (
            not structured_output
            and is_deepseek_model(model)
            and _is_deepseek_beta_base_url(base_url)
            and messages[-1].get("role") == "assistant"
        ):
            messages[-1]["prefix"] = True
        return messages

    if isinstance(prompt, dict) and ("system" in prompt or "user" in prompt or "assistant" in prompt):
        messages = []
        is_gpt = (model or "").lower().startswith("gpt")
        use_deepseek_prefix = is_deepseek_model(model) and _is_deepseek_beta_base_url(base_url)
        system_content = str(prompt["system"]) if prompt.get("system") else ""
        user_content = str(prompt["user"]) if prompt.get("user") else ""
        assistant_content = str(prompt["assistant"]) if prompt.get("assistant") else ""

        if system_content and _looks_like_task_prefixed_user(user_content):
            messages.append({"role": "system", "content": CACHE_FRIENDLY_SYSTEM})
            user_content = _insert_after_task_section(user_content, system_content)
        else:
            if system_content:
                messages.append({"role": "system", "content": system_content})

        if is_gpt and assistant_content:
            # GPT: merge assistant prefill into user message
            user_content = f"{user_content}\n\n{assistant_content}" if user_content else assistant_content

        if user_content:
            messages.append({"role": "user", "content": user_content})
        if assistant_content and not is_gpt:
            assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant_content}
            if use_deepseek_prefix and not structured_output:
                assistant_message["prefix"] = True
            messages.append(assistant_message)

        if not messages:
            raise ValueError("Chat dict must have at least one of: system, user, assistant")
        return messages
    content = prompt if isinstance(prompt, str) else compile_prompt_to_md(prompt)
    return [{"role": "user", "content": content}]


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int | None = None,
    retry_delay: float | None = None,
    stage_id: str | None = None,
) -> str:
    """Streaming text generation via OpenAI-compatible Chat API. Supports chat format {system, user, assistant} for Qwen."""
    stage = cfg.agent.code
    model = stage.model
    messages = _prompt_to_messages(
        prompt,
        model=model,
        base_url=stage.base_url,
        structured_output=json_schema is not None,
    )
    client = OpenAI(
        api_key=stage.api_key,
        base_url=stage.base_url or None,
        max_retries=0,
        timeout=max(1.0, float(getattr(stage, "request_timeout_seconds", 1200.0))),
    )
    max_retries = max(
        1,
        int(
            max_retries
            if max_retries is not None
            else getattr(stage, "generation_max_retries", 5)
        ),
    )
    retry_delay = max(
        0.0,
        float(
            retry_delay
            if retry_delay is not None
            else getattr(stage, "generation_retry_delay_seconds", 3.0)
        ),
    )
    max_continuation_rounds = max(
        0,
        int(getattr(stage, "continuation_max_rounds", MAX_CONTINUATION_ROUNDS)),
    )
    continuation_overlap_scan_chars = max(
        0,
        int(
            getattr(
                stage,
                "continuation_overlap_scan_chars",
                CONTINUATION_OVERLAP_SCAN_CHARS,
            )
        ),
    )
    network_retry_base_sleep = max(
        0.0,
        float(
            getattr(
                stage,
                "network_retry_base_sleep_seconds",
                NETWORK_RETRY_BASE_SLEEP_SECONDS,
            )
        ),
    )
    network_retry_max_sleep = max(
        0.0,
        float(
            getattr(
                stage,
                "network_retry_max_sleep_seconds",
                NETWORK_RETRY_MAX_SLEEP_SECONDS,
            )
        ),
    )
    # Qwen: thinking + json_schema are mutually exclusive — drop schema, keep thinking.
    if json_schema is not None and thinking_json_incompatible(model):
        json_schema = None
    use_thinking = _resolve_use_thinking(stage, func_spec=None, json_schema=json_schema)
    profile = get_profile(model, use_thinking=bool(use_thinking)) if use_thinking is not None else {}

    extra_body: dict[str, Any] = {}
    if "top_k" in profile:
        extra_body["top_k"] = profile["top_k"]
    if use_thinking is not None and "enable_thinking" in profile:
        extra_body["enable_thinking"] = profile["enable_thinking"]
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": profile.get("temperature", temperature if temperature is not None else 1.0),
        "stream": True,
    }
    resolved_max_tokens = _resolve_max_tokens(max_tokens, stage)
    if resolved_max_tokens is not None:
        params["max_tokens"] = resolved_max_tokens
    if "top_p" in profile:
        params["top_p"] = profile["top_p"]
    if "presence_penalty" in profile:
        params["presence_penalty"] = profile["presence_penalty"]
    if extra_body:
        params["extra_body"] = extra_body
    _apply_deepseek_request_options(params, stage, model, use_thinking)
    if stop_tokens:
        params["stop"] = stop_tokens
    if json_schema is not None:
        if supports_json_schema(model):
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "strict": False, "schema": json_schema},
            }
        else:
            params["response_format"] = {"type": "json_object"}

    logger.info(f"generate messages: {len(messages)} turns", extra={"verbose": True})
    logger.info(
        "generate message chars=%s, cache_friendly=%s",
        _message_chars(messages),
        _is_cache_friendly_message_layout(messages),
        extra={"verbose": True},
    )
    for attempt in range(max_retries):
        try:
            raw_full_text, final_finish_reason, final_usage, elapsed = _collect_stream_response(
                client,
                params,
                label="generate_stream",
                stage=stage,
            )
            logger.info(f"generate response: {_strip_visible_thinking(raw_full_text)}", extra={"verbose": True})
            provider_finish_reason = final_finish_reason
            raw_full_text, final_finish_reason = _normalize_stream_completion(raw_full_text, final_finish_reason)
            full_text = _strip_visible_thinking(raw_full_text)
            prompt_name = infer_prompt_name("generate")
            log_llm_usage(
                cfg=cfg,
                prompt_name=prompt_name,
                stage_id=stage_id,
                mode="generate_stream",
                provider="openai_compatible",
                model=model,
                usage=final_usage,
                seconds=elapsed,
                finish_reason=final_finish_reason,
                max_tokens=params.get("max_tokens"),
                parsed_ok=not _needs_continuation(final_finish_reason),
                prompt_parts=prompt_parts_from_messages(messages)
                + (
                    [
                        {
                            "name": "json_schema",
                            "role": "schema",
                            "content": json.dumps(json_schema, ensure_ascii=False, default=str),
                        }
                    ]
                    if json_schema is not None
                    else []
                ),
                estimated_completion_text=full_text,
                extra={"usage_estimated_when_missing": not bool(usage_to_dict(final_usage)),
                       "provider_finish_reason": provider_finish_reason},
            )
            if _finish_reason_is_length(final_finish_reason) and json_schema is not None:
                logger.warning(
                    "generate response hit max_tokens with json_schema enabled; automatic continuation is disabled for structured output"
                )

            allow_continuation = json_schema is None
            if not allow_continuation and final_finish_reason == "provider_interrupted":
                raise RuntimeError("Provider network error interrupted structured output")
            continuation_round = 0
            while (
                allow_continuation
                and _needs_continuation(final_finish_reason)
                and continuation_round < max_continuation_rounds
            ):
                continuation_round += 1
                logger.warning(
                    "generate response incomplete (%s); requesting continuation round %s/%s",
                    final_finish_reason,
                    continuation_round,
                    max_continuation_rounds,
                )
                continuation_messages = _build_continuation_messages(
                    messages,
                    raw_full_text,
                    max_tokens=params.get("max_tokens"),
                    round_index=continuation_round,
                    reason=final_finish_reason,
                )
                continuation_params = dict(params)
                continuation_params["messages"] = continuation_messages
                continuation_text, final_finish_reason, continuation_usage, continuation_elapsed = _collect_stream_response(
                    client,
                    continuation_params,
                    label=f"generate_stream_continuation_{continuation_round}",
                    stage=stage,
                )
                provider_finish_reason = final_finish_reason
                continuation_text, final_finish_reason = _normalize_stream_completion(continuation_text, final_finish_reason)
                continuation_text = _strip_visible_thinking(continuation_text)
                raw_full_text = _append_with_overlap(
                    raw_full_text,
                    continuation_text,
                    max_scan_chars=continuation_overlap_scan_chars,
                )
                full_text = _strip_visible_thinking(raw_full_text)
                log_llm_usage(
                    cfg=cfg,
                    prompt_name=f"{prompt_name}:continuation_{continuation_round}",
                    stage_id=stage_id,
                    mode="generate_stream_continuation",
                    provider="openai_compatible",
                    model=model,
                    usage=continuation_usage,
                    seconds=continuation_elapsed,
                    finish_reason=final_finish_reason,
                    max_tokens=params.get("max_tokens"),
                    parsed_ok=not _needs_continuation(final_finish_reason),
                    prompt_parts=prompt_parts_from_messages(continuation_messages),
                    estimated_completion_text=_strip_visible_thinking(continuation_text),
                    extra={
                        "usage_estimated_when_missing": not bool(usage_to_dict(continuation_usage)),
                        "continuation_round": continuation_round,
                        "provider_finish_reason": provider_finish_reason,
                    },
                )
            if allow_continuation and _needs_continuation(final_finish_reason):
                if final_finish_reason == "provider_interrupted":
                    raise RuntimeError("Provider interruption persisted after the bounded continuation attempts")
                logger.warning(
                    "generate response still truncated after %s continuation round(s); returning partial output",
                    continuation_round,
                )
            logger.info(f"generate final response after continuations: {full_text}", extra={"verbose": True})
            return full_text
        except Exception as e:
            retryable = _is_retryable_error(e)
            logger.warning(f"generate failed, retryable={retryable}, attempt {attempt + 1}/{max_retries}: {e}")
            if attempt >= max_retries - 1:
                logger.error("generate retry limit reached")
                raise
            if not retryable:
                raise
            time.sleep(
                min(
                    network_retry_max_sleep,
                    max(float(retry_delay), network_retry_base_sleep * (attempt + 1)),
                )
            )
    return ""
