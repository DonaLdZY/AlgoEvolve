"""Gemini API backend: function calling (query), streaming generation (generate),
   prompt compilation, retry logic, and function-calling specs."""

import json
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import backoff
import jsonschema
from dataclasses_json import DataClassJsonMixin
from funcy import notnone, select_values
from google import genai
from google.genai import types
from config import Config
from .usage import infer_prompt_name, log_llm_usage, estimate_text_tokens

logger = logging.getLogger("AlgoEvolve")
NETWORK_RETRY_MAX_ATTEMPTS = 5
NETWORK_RETRY_BASE_SLEEP_SECONDS = 5.0
NETWORK_RETRY_MAX_SLEEP_SECONDS = 30.0
MAX_CONTINUATION_ROUNDS = 2
CONTINUATION_OVERLAP_SCAN_CHARS = 4096


def _strip_visible_thinking(text: str) -> str:
    if "</think>" in text:
        return text[text.find("</think>") + 8:]
    return text


def _finish_reason_text(value) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value)


def _finish_reason_is_length(finish_reason: str | None) -> bool:
    normalized = str(finish_reason or "").strip().lower()
    return any(key in normalized for key in ["max_tokens", "max_output_tokens", "length"])


def _append_with_overlap(
    base: str,
    addition: str,
    max_scan_chars: int = CONTINUATION_OVERLAP_SCAN_CHARS,
) -> str:
    if not base or not addition:
        return f"{base}{addition}"
    max_scan = min(len(base), len(addition), max(0, int(max_scan_chars)))
    for size in range(max_scan, 15, -1):
        if base[-size:] == addition[:size]:
            return base + addition[size:]
    return base + addition


def _build_continuation_instruction(max_tokens: int | None, round_index: int) -> str:
    token_text = f"{max_tokens}" if max_tokens else "the provider/API default"
    return (
        "Your previous assistant response was cut off because it reached the output token limit. "
        "Continue exactly from the point where the previous assistant response stopped.\n"
        "- Do not repeat any previous text.\n"
        "- Do not restart the answer.\n"
        "- Start with the next character or token that should follow the previous assistant response.\n"
        "- Preserve all required formatting, markdown fences, JSON/text/code structure, and indentation.\n"
        f"- This is continuation round {round_index}; the output cap for this continuation is {token_text} tokens."
    )


def _build_continuation_prompt(prompt: str, raw_assistant_so_far: str, *, max_tokens: int | None, round_index: int) -> str:
    return (
        f"{prompt.rstrip()}\n\n"
        "# Previous Assistant Response Already Produced\n"
        "Do not repeat this text; it is provided only so you can continue from the cutoff point.\n"
        "<already_produced_assistant_response>\n"
        f"{raw_assistant_so_far}\n"
        "</already_produced_assistant_response>\n\n"
        "# Continuation Instruction\n"
        f"{_build_continuation_instruction(max_tokens=max_tokens, round_index=round_index)}"
    )


def _chunk_finish_reason(chunk) -> str:
    candidates = getattr(chunk, "candidates", None) or []
    if candidates:
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            reason = getattr(candidates[0], "finishReason", None)
        return _finish_reason_text(reason)
    return ""

# ---------------------------------------------------------------------------
#  Type aliases
# ---------------------------------------------------------------------------
PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

# ---------------------------------------------------------------------------
#  Prompt & message helpers
# ---------------------------------------------------------------------------

@backoff.on_predicate(
    wait_gen=backoff.constant,
    interval=5,
    max_time=300,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    """Call *create_fn* with automatic retry on transient errors."""
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.warning(f"Retryable error: {e}\n{traceback.format_exc()}")
        return False


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    def to_openai_tool_dict(self, strict: bool = True):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
                "strict": strict,
            },
        }

    @property
    def as_openai_tool_dict(self):
        return self.to_openai_tool_dict(strict=True)

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }

# ---------------------------------------------------------------------------
#  Gemini client
# ---------------------------------------------------------------------------

GEMINI_TIMEOUT_EXCEPTIONS = (
    Exception,  # Gemini SDK may throw various exceptions
)


def _setup_gemini_client(stage):
    timeout_ms = max(
        1000,
        int(float(getattr(stage, "request_timeout_seconds", 1200.0)) * 1000),
    )
    return genai.Client(
        api_key=stage.api_key,
        http_options={"base_url": stage.base_url, "timeout": timeout_ms},
    )


def _convert_func_spec_to_gemini_tool(func_spec: FunctionSpec) -> types.Tool:
    """Convert FunctionSpec to Gemini Tool format."""
    function_declaration = types.FunctionDeclaration(
        name=func_spec.name,
        description=func_spec.description,
        parameters=func_spec.json_schema
    )
    return types.Tool(function_declarations=[function_declaration])


def _gemini_thinking_level(stage, default_level: str) -> str | None:
    """Resolve Gemini thinking level from config override."""
    forced = getattr(stage, "enable_thinking", None)
    effort = getattr(stage, "reasoning_effort", None)
    if isinstance(effort, str) and effort.strip().lower() in {"", "default", "none", "null"}:
        effort = None
    if forced is None:
        return effort
    return default_level if forced else None


def _stage_max_tokens(stage) -> int | None:
    try:
        tokens = int(getattr(stage, "max_tokens", 0) or 0)
    except (TypeError, ValueError):
        return None
    return tokens if tokens > 0 else None


def _stage_minimum_output_tokens(stage) -> int:
    try:
        tokens = int(getattr(stage, "minimum_output_tokens", 32768) or 0)
    except (TypeError, ValueError):
        tokens = 32768
    return max(0, tokens)


def _resolve_max_tokens(explicit_value, stage) -> int | None:
    try:
        explicit = int(explicit_value) if explicit_value is not None else 0
    except (TypeError, ValueError):
        explicit = 0
    requested = explicit if explicit > 0 else int(_stage_max_tokens(stage) or 0)
    effective = max(requested, _stage_minimum_output_tokens(stage))
    return effective or None


def _stage_config_for_model(cfg: Config, model: str, stage_name: str | None = None):
    stage_name = str(stage_name or "").strip().lower()
    if stage_name == "code":
        return cfg.agent.code
    if stage_name == "feedback":
        return cfg.agent.feedback
    return cfg.agent.code if cfg.agent.code.model == model else cfg.agent.feedback


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


def _collect_stream_response(
    *,
    client: genai.Client,
    model_name: str,
    contents: str,
    generation_config: types.GenerateContentConfig,
) -> tuple[str, str, dict, float]:
    """Run one Gemini streaming request and collect raw text, finish reason, usage, and seconds."""
    t0 = time.time()
    response = client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=generation_config,
    )
    full_text = ""
    final_usage: dict = {}
    final_finish_reason = ""
    for chunk in response:
        if chunk.text:
            full_text += chunk.text
        reason = _chunk_finish_reason(chunk)
        if reason:
            final_finish_reason = reason
        usage_meta = getattr(chunk, "usage_metadata", None)
        if usage_meta is not None:
            in_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            out_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
            if in_tokens or out_tokens:
                final_usage = {
                    "prompt_tokens": in_tokens,
                    "completion_tokens": out_tokens,
                    "total_tokens": int(in_tokens) + int(out_tokens),
                }
    return full_text, final_finish_reason, final_usage, time.time() - t0


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    stage = _stage_config_for_model(cfg, filtered_kwargs.get("model", ""), filtered_kwargs.get("stage_name"))
    client = _setup_gemini_client(stage)
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

    # Construct contents for Gemini
    contents = []
    if system_message:
        if user_message:
            contents = f"{system_message}\n\n{user_message}"
        else:
            contents = system_message
    elif user_message:
        contents = user_message
    else:
        raise ValueError("Either system_message or user_message must be provided")

    # Build generation config with tools if func_spec is provided
    config_params = {
        "temperature": filtered_kwargs.get("temperature", 1.0),
    }
    resolved_max_tokens = _resolve_max_tokens(filtered_kwargs.get("max_tokens"), stage)
    if resolved_max_tokens is not None:
        config_params["max_output_tokens"] = resolved_max_tokens
    thinking_level = _gemini_thinking_level(stage, "low")
    if thinking_level is not None:
        config_params["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

    if func_spec is not None:
        config_params["response_mime_type"] = "application/json"
        config_params["response_json_schema"] = func_spec.json_schema

    generation_config = types.GenerateContentConfig(**config_params)

    t0 = time.time()
    logger.info(f"Querying Gemini with model: {filtered_kwargs.get('model')}")
    prompt_name = str(filtered_kwargs.get("prompt_name") or infer_prompt_name("query"))

    try:
        last_exc: Exception | None = None
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=filtered_kwargs.get("model", "gemini-3-pro-preview"),
                    contents=contents,
                    config=generation_config,
                )
                break
            except Exception as e:
                last_exc = e
                retryable = _is_retryable_error(e)
                logger.warning(
                    "Gemini query failed (attempt %s/%s, retryable=%s): %s",
                    attempt,
                    max_attempts,
                    retryable,
                    e,
                )
                if (not retryable) or attempt >= max_attempts:
                    raise
                sleep_secs = min(max_sleep, base_sleep * attempt)
                logger.warning(
                    "Gemini network error; reconnecting after %.1fs (attempt %s/%s)",
                    sleep_secs,
                    attempt,
                    max_attempts,
                )
                time.sleep(sleep_secs)
        if response is None and last_exc is not None:
            raise last_exc
        req_time = time.time() - t0

        # Parse response
        if func_spec is None:
            output = response.text
            logger.info(f"Gemini response: {output}", extra={"verbose": True})
        else:
            text = response.text
            if not text:
                raise ValueError("No response text from Gemini for structured output")
            output = json.loads(text)
            if isinstance(output, list):
                if len(output) > 0:
                    output = output[0]
                else:
                    raise ValueError("Gemini returned empty array for structured output")
            logger.info(f"Gemini structured output response: {output}", extra={"verbose": True})

        in_tokens = 0
        out_tokens = 0
        if hasattr(response, 'usage_metadata'):
            in_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            out_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        usage_payload = {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": int(in_tokens or 0) + int(out_tokens or 0),
        } if (in_tokens or out_tokens) else {}
        log_llm_usage(
            cfg=cfg,
            prompt_name=prompt_name,
            mode="query_function" if func_spec is not None else "query_text",
            provider="gemini",
            model=filtered_kwargs.get("model", "gemini-3-pro-preview"),
            usage=usage_payload,
            seconds=req_time,
            max_tokens=config_params.get("max_output_tokens"),
            parsed_ok=True,
            prompt_parts=[
                {"name": "contents", "role": "user", "content": str(contents)},
                *(
                    [
                        {
                            "name": "function_schema",
                            "role": "schema",
                            "content": json.dumps(func_spec.json_schema, ensure_ascii=False, default=str),
                        }
                    ]
                    if func_spec is not None
                    else []
                ),
            ],
            estimated_completion_text=json.dumps(output, ensure_ascii=False, default=str) if not isinstance(output, str) else output,
            extra={"usage_estimated_when_missing": not bool(usage_payload)},
        )

        info = {
            "model": filtered_kwargs.get("model", "gemini-3-pro-preview"),
            "created": int(time.time()),
        }

        return output, req_time, in_tokens, out_tokens, info

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise e


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int | None = None,
    retry_delay: float | None = None,
) -> str:
    """Streaming text generation via Gemini API.

    Args:
        prompt: The text prompt to complete.
        cfg: Config instance (provides model name and initializes client).
        temperature: Sampling temperature (default 1.0).
        max_tokens: Max output tokens. If omitted, the provider/API default is used.
        stop_tokens: Optional stop sequences.
        json_schema: Optional JSON schema for structured output.
        max_retries: Max retry attempts on failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        The generated text (with <think> blocks stripped).
    """
    stage = cfg.agent.code
    client = _setup_gemini_client(stage)
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

    # Convert dict/list prompts to markdown string
    if prompt is not None and not isinstance(prompt, str):
        prompt = compile_prompt_to_md(prompt)

    logger.info(f"generate prompt: {prompt}", extra={"verbose": True})

    config_params = {
        "temperature": temperature if temperature is not None else 1.0,
        "stop_sequences": stop_tokens,
    }
    resolved_max_tokens = _resolve_max_tokens(max_tokens, stage)
    if resolved_max_tokens is not None:
        config_params["max_output_tokens"] = resolved_max_tokens
    thinking_level = _gemini_thinking_level(stage, "high")
    if thinking_level is not None:
        config_params["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

    if json_schema is not None:
        config_params["response_mime_type"] = "application/json"
        config_params["response_json_schema"] = json_schema
        logger.info("Enforcing JSON output with schema", extra={"verbose": True})

    generation_config = types.GenerateContentConfig(**config_params)
    model_name = cfg.agent.code.model

    for attempt in range(max_retries):
        try:
            raw_full_text, final_finish_reason, final_usage, elapsed = _collect_stream_response(
                client=client,
                model_name=model_name,
                contents=str(prompt or ""),
                generation_config=generation_config,
            )
            full_text = _strip_visible_thinking(raw_full_text)
            logger.info(f"generate response: {full_text}", extra={"verbose": True})
            prompt_name = infer_prompt_name("generate")
            log_llm_usage(
                cfg=cfg,
                prompt_name=prompt_name,
                mode="generate_stream",
                provider="gemini",
                model=model_name,
                usage=final_usage,
                seconds=elapsed,
                finish_reason=final_finish_reason,
                max_tokens=config_params.get("max_output_tokens"),
                parsed_ok=True,
                prompt_parts=[
                    {"name": "prompt", "role": "user", "content": str(prompt or "")},
                    *(
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
                ],
                estimated_completion_text=full_text,
                extra={
                    "usage_estimated_when_missing": not bool(final_usage),
                    "estimated_prompt_tokens_when_missing": estimate_text_tokens(str(prompt or "")) if not final_usage else 0,
                },
            )
            if _finish_reason_is_length(final_finish_reason) and json_schema is not None:
                logger.warning(
                    "Gemini generate response hit max_output_tokens with json_schema enabled; automatic continuation is disabled for structured output"
                )

            allow_continuation = json_schema is None
            continuation_round = 0
            while (
                allow_continuation
                and _finish_reason_is_length(final_finish_reason)
                and continuation_round < max_continuation_rounds
            ):
                continuation_round += 1
                logger.warning(
                    "Gemini generate response truncated by max_output_tokens (%s); requesting continuation round %s/%s",
                    config_params.get("max_output_tokens"),
                    continuation_round,
                    max_continuation_rounds,
                )
                continuation_prompt = _build_continuation_prompt(
                    str(prompt or ""),
                    raw_full_text,
                    max_tokens=config_params.get("max_output_tokens"),
                    round_index=continuation_round,
                )
                continuation_text, final_finish_reason, continuation_usage, continuation_elapsed = _collect_stream_response(
                    client=client,
                    model_name=model_name,
                    contents=continuation_prompt,
                    generation_config=generation_config,
                )
                raw_full_text = _append_with_overlap(
                    raw_full_text,
                    continuation_text,
                    max_scan_chars=continuation_overlap_scan_chars,
                )
                full_text = _strip_visible_thinking(raw_full_text)
                log_llm_usage(
                    cfg=cfg,
                    prompt_name=f"{prompt_name}:continuation_{continuation_round}",
                    mode="generate_stream_continuation",
                    provider="gemini",
                    model=model_name,
                    usage=continuation_usage,
                    seconds=continuation_elapsed,
                    finish_reason=final_finish_reason,
                    max_tokens=config_params.get("max_output_tokens"),
                    parsed_ok=True,
                    prompt_parts=[{"name": "prompt", "role": "user", "content": continuation_prompt}],
                    estimated_completion_text=_strip_visible_thinking(continuation_text),
                    extra={
                        "usage_estimated_when_missing": not bool(continuation_usage),
                        "estimated_prompt_tokens_when_missing": estimate_text_tokens(continuation_prompt)
                        if not continuation_usage
                        else 0,
                        "continuation_round": continuation_round,
                    },
                )
            if allow_continuation and _finish_reason_is_length(final_finish_reason):
                logger.warning(
                    "Gemini generate response still truncated after %s continuation round(s); returning partial output",
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
