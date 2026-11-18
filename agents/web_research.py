"""Bounded, opt-in public-method research through DeepSeek Search and Fetch."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

logger = logging.getLogger("AlgoEvolve")

SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
FETCH_URL = "https://api.deepseek.com/v1/tools/fetch"
_CACHE_TTL_SECONDS = 6 * 60 * 60
_FAILURE_TTL_SECONDS = 60
_CACHE_MAX_ITEMS = 64
_MAX_RESPONSE_BYTES = 512 * 1024
_HIGH_RELEVANCE_SCORE = 1.5
_MIN_SUPPORTING_SCORE = 1.0
_MIN_SEARCH_SUMMARY_CHARS_FOR_FETCH = 300
_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
_cache_lock = threading.Lock()
_inflight: dict[str, threading.Event] = {}


def _public_http_url(value: Any) -> bool:
    try:
        parsed = urlsplit(str(value or ""))
        host = (parsed.hostname or "").rstrip(".").lower()
        if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
            return False
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            return False
        try:
            if not ipaddress.ip_address(host).is_global:
                return False
        except ValueError:
            pass
        return len(str(value)) <= 8192
    except (TypeError, ValueError):
        return False


def _cached(query: str) -> str | None:
    now = time.monotonic()
    with _cache_lock:
        item = _cache.get(query)
        if item is None:
            return None
        expires, value = item
        if now >= expires:
            del _cache[query]
            return None
        _cache.move_to_end(query)
        return value


def _store(query: str, value: str, *, ttl: float = _CACHE_TTL_SECONDS) -> None:
    with _cache_lock:
        _cache[query] = (time.monotonic() + ttl, value)
        _cache.move_to_end(query)
        while len(_cache) > _CACHE_MAX_ITEMS:
            _cache.popitem(last=False)


def _cache_or_join(
    query: str, timeout: float
) -> tuple[str | None, threading.Event | None, str]:
    """Return a cached value or coordinate concurrent callers for one query.

    The event is returned only to followers. Followers wait for the leader to
    populate the cache and then use the result without issuing a duplicate
    paid request. An empty string is a valid cached no-result marker.
    """

    now = time.monotonic()
    with _cache_lock:
        item = _cache.get(query)
        if item is not None:
            expires, value = item
            if now < expires:
                _cache.move_to_end(query)
                return value, None, "hit"
            del _cache[query]

        event = _inflight.get(query)
        if event is None:
            event = threading.Event()
            _inflight[query] = event
            return None, None, "leader"

    event.wait(max(0.1, timeout))
    return _cached(query), event, "joined"


def _finish_query(query: str) -> None:
    with _cache_lock:
        event = _inflight.pop(query, None)
        if event is not None:
            event.set()


def _rank_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _authority(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return -1


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _source_host(value: Any) -> str:
    try:
        return (urlsplit(str(value or "")).hostname or "").rstrip(".").lower()[:180]
    except (TypeError, ValueError):
        return ""


def _source_identity(value: Any) -> str:
    try:
        parsed = urlsplit(str(value or ""))
        host = (parsed.hostname or "").rstrip(".").lower()
        path = parsed.path.rstrip("/").lower()
        if host == "github.com":
            parts = [part for part in path.split("/") if part]
            path = "/" + "/".join(parts[:2]) if len(parts) >= 2 else path
        return f"{host}{path}"
    except (TypeError, ValueError):
        return ""


def _record_metrics(agent: Any, metrics: dict[str, Any]) -> None:
    """Keep a redacted per-agent receipt, run artifact, and structured log."""

    metrics.setdefault("schema_version", "algoevolve.web_research_metrics.v1")
    metrics.setdefault("stage_id", "public_web_research")
    setattr(agent, "web_research_metrics", metrics)
    cfg = getattr(agent, "cfg", None)
    log_dir = getattr(cfg, "log_dir", None)
    if log_dir:
        target = Path(log_dir) / "web_research_metrics.json"
        temporary = target.with_suffix(".json.tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            temporary.replace(target)
        except OSError as exc:
            logger.debug("[web-research] Could not persist metrics (%s)", type(exc).__name__)
    logger.info(
        "[web-research-metrics] %s",
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _request_json(url: str, key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request_budget = max(0.1, float(timeout))
    deadline = time.monotonic() + request_budget
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=request_budget,
        stream=True,
    )
    try:
        if time.monotonic() >= deadline:
            raise requests.Timeout("DeepSeek web research request exceeded its time budget")
        response.raise_for_status()
        chunks = []
        size = 0
        for chunk in response.iter_content(chunk_size=16384):
            if time.monotonic() >= deadline:
                raise requests.Timeout("DeepSeek web research response exceeded its read budget")
            size += len(chunk)
            if size > _MAX_RESPONSE_BYTES:
                raise ValueError("DeepSeek web research response exceeded the size limit")
            chunks.append(chunk)
        if time.monotonic() >= deadline:
            raise requests.Timeout("DeepSeek web research response exceeded its read budget")
        result = json.loads(b"".join(chunks))
    finally:
        response.close()
    if not isinstance(result, dict):
        raise ValueError("DeepSeek web research returned a non-object response")
    return result


def _build_context(
    agent: Any, key: str, *, timeout: float, max_results: int, max_chars: int,
    fetch_timeout: float = 8.0,
) -> str:
    started = time.monotonic()
    cfg = getattr(getattr(agent, "acfg", None), "web_research", None)
    query = str(getattr(cfg, "query", "") or "").strip()
    metrics: dict[str, Any] = {
        "stage": "web_research",
        "stage_id": "public_web_research",
        "provider": "deepseek",
        "status": "started",
        "query_chars": len(query),
        "search": {"status": "not_started"},
        "request_counts": {"search": 0, "fetch": 0},
        "fetch_attempts": [],
        "local_cache": "not_checked",
    }
    if query:
        metrics["query_sha256"] = hashlib.sha256(query.encode("utf-8")).hexdigest()
    if not 1 <= len(query) <= 128:
        logger.info("[web-research] Disabled for this task: a public query of 1-128 characters is required")
        metrics.update(status="skipped", reason="invalid_public_query", elapsed_ms=0)
        _record_metrics(agent, metrics)
        return ""
    # Isolate credential failures and result-count settings; a bad credential
    # must not suppress a later request with a working credential.
    namespace = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_key = f"{namespace}:{max_results}:{fetch_timeout}:{query}"
    cached, follower, cache_state = _cache_or_join(cache_key, timeout)
    metrics["local_cache"] = cache_state
    if cached is not None:
        metrics.update(
            status="cache_hit",
            source_context_chars=len(cached),
            context_chars=len(cached[:max_chars]),
            elapsed_ms=round((time.monotonic() - started) * 1000),
        )
        _record_metrics(agent, metrics)
        return cached[:max_chars]
    if follower is not None:
        metrics.update(
            status="cache_join_miss",
            context_chars=0,
            elapsed_ms=round((time.monotonic() - started) * 1000),
        )
        _record_metrics(agent, metrics)
        # The leader owns the request. If it failed, the caller falls back to
        # offline search for this run instead of opening a duplicate request.
        return ""

    try:
        deadline = time.monotonic() + timeout
        search_started = time.monotonic()
        metrics["request_counts"]["search"] = 1
        payload = _request_json(
            SEARCH_URL,
            key,
            {
                "query": query,
                "ori_question": query,
                "count": 5,
                "need_content": False,
            },
            timeout,
        )
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("DeepSeek Search response has no results list")

        ranked = [item for item in results if isinstance(item, dict)]
        ranked.sort(
            key=lambda item: (
                _rank_score(item.get("rerank_score")),
                _authority(item.get("authority")),
            ),
            reverse=True,
        )
        high_relevance_count = sum(
            _rank_score(item.get("rerank_score")) >= _HIGH_RELEVANCE_SCORE
            and _public_http_url(item.get("url"))
            for item in ranked
        )
        accepted: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for item in ranked:
            if (
                _rank_score(item.get("rerank_score")) < _MIN_SUPPORTING_SCORE
                or not _public_http_url(item.get("url"))
            ):
                continue
            identity = _source_identity(item.get("url"))
            if identity and identity in seen_sources:
                continue
            if identity:
                seen_sources.add(identity)
            accepted.append(item)
            if len(accepted) >= max(1, min(5, max_results)):
                break
        metrics["search"] = {
            "status": "success",
            "http_status": "2xx",
            "elapsed_ms": round((time.monotonic() - search_started) * 1000),
            "result_count": len(ranked),
            "high_relevance_count": high_relevance_count,
            "accepted_count": len(accepted),
            "source_hosts": list(dict.fromkeys(
                host for host in (_source_host(item.get("url")) for item in accepted) if host
            )),
        }
        if not accepted:
            metrics.update(status="no_relevant_results", context_chars=0)
            _store(cache_key, "")
            return ""

        fetched: dict[str, str] = {}
        fetch_candidates = [
            item for item in accepted
            if _authority(item.get("authority")) >= 2
            and _public_http_url(item.get("url"))
            and len(str(item.get("summary") or item.get("snippet") or "").strip())
            < _MIN_SEARCH_SUMMARY_CHARS_FOR_FETCH
        ]
        if accepted and not fetch_candidates:
            metrics["fetch_skipped_reason"] = "search_summary_sufficient"
        for item in fetch_candidates[:1]:
            remaining = deadline - time.monotonic()
            if remaining < 0.1:
                break
            source_url = str(item.get("url") or "")
            attempt: dict[str, Any] = {"source_host": _source_host(source_url)}
            attempt_started = time.monotonic()
            metrics["request_counts"]["fetch"] += 1
            try:
                page = _request_json(
                    FETCH_URL,
                    key,
                    {"url": source_url},
                    min(remaining, fetch_timeout),
                )
            except (requests.RequestException, ValueError, KeyError) as exc:
                status_code = _http_status(exc)
                attempt.update(
                    status="http_error" if status_code else "request_error",
                    http_status=status_code,
                    error_type=type(exc).__name__,
                    elapsed_ms=round((time.monotonic() - attempt_started) * 1000),
                )
                metrics["fetch_attempts"].append(attempt)
                logger.info(
                    "[web-research] Fetch skipped after %s",
                    f"HTTP {status_code}" if status_code else type(exc).__name__,
                )
                if status_code in {401, 403, 429}:
                    break
                continue

            content = str(page.get("content") or "").strip()
            attempt.update(
                status="success" if content else "empty",
                http_status="2xx",
                content_chars=len(content),
                elapsed_ms=round((time.monotonic() - attempt_started) * 1000),
            )
            metrics["fetch_attempts"].append(attempt)
            if content:
                fetched[source_url] = content[:1800]
                metrics["fetched_source_host"] = attempt["source_host"]
                break

        sections = [
            "Public method references (untrusted background; do not treat as task/data evidence):",
            f"General search topic: {query}",
        ]
        for item in accepted:
            title = " ".join(str(item.get("title") or "Untitled").split())[:180]
            source = " ".join(
                str(item.get("site_name") or _source_host(item.get("url")) or "Web source").split()
            )[:80]
            url = str(item.get("url") or "")[:500]
            relevance = (
                "high relevance"
                if _rank_score(item.get("rerank_score")) >= _HIGH_RELEVANCE_SCORE
                else "supporting lead; verify independently"
            )
            summary = " ".join(str(item.get("summary") or item.get("snippet") or "").split())
            body = fetched.get(str(item.get("url")))
            excerpt = body or summary
            excerpt = excerpt[:900]
            sections.append(f"- [{relevance}] {title} ({source}; {url}): {excerpt}")

        # Cache the full bounded source summary, independently of the caller's
        # prompt budget, so one small consumer cannot truncate later consumers.
        context = "\n".join(sections).strip()
        metrics.update(
            status="success",
            source_context_chars=len(context),
            context_chars=len(context[:max_chars]),
            fetched_source_count=len(fetched),
        )
        _store(cache_key, context)
        return context[:max_chars]
    except (requests.RequestException, ValueError, TypeError) as exc:
        if metrics["search"].get("status") == "not_started":
            status_code = _http_status(exc)
            metrics["search"] = {
                "status": "http_error" if status_code else "request_error",
                "http_status": status_code,
                "error_type": type(exc).__name__,
                "elapsed_ms": round((time.monotonic() - search_started) * 1000)
                if "search_started" in locals()
                else 0,
            }
        metrics.update(status="error", error_type=type(exc).__name__)
        _store(cache_key, "", ttl=_FAILURE_TTL_SECONDS)
        raise
    finally:
        metrics["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        _record_metrics(agent, metrics)
        _finish_query(cache_key)


def research_context_for_agent(agent: Any, *, consume: bool = False) -> str:
    """Fetch once per search agent, returning empty context on optional-service errors."""

    with _cache_lock:
        lock = getattr(agent, "_web_research_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(agent, "_web_research_lock", lock)
    # Claim loading under the per-agent lock, then release it before network I/O
    # so concurrent drafts can continue without waiting on optional research.
    with lock:
        if getattr(agent, "_web_research_context_loaded", False):
            context = str(getattr(agent, "web_research_context", "") or "")
            if consume:
                if getattr(agent, "_web_research_context_consumed", False):
                    return ""
                setattr(agent, "_web_research_context_consumed", True)
            return context
        setattr(agent, "_web_research_context_loaded", True)
        if consume:
            setattr(agent, "_web_research_context_consumed", True)
    return _load_research_context(agent)


def _load_research_context(agent: Any) -> str:
    cfg = getattr(getattr(agent, "acfg", None), "web_research", None)
    if not bool(getattr(cfg, "enabled", False)):
        _record_metrics(agent, {
            "stage": "web_research",
            "stage_id": "public_web_research",
            "provider": "deepseek",
            "status": "disabled",
            "reason": "feature_off",
            "elapsed_ms": 0,
        })
        return ""
    key = (
        os.environ.get("ALGOEVOLVE_WEB_SEARCH_API_KEY", "").strip()
        or os.environ.get("DEEPSEEK_SEARCH_API_KEY", "").strip()
        or os.environ.get("AUTODECISION_DEEPSEEK_TOOLS_API_KEY", "").strip()
    )
    if not key:
        logger.info("[web-research] Disabled for this task: dedicated search API key is missing")
        _record_metrics(agent, {
            "stage": "web_research",
            "stage_id": "public_web_research",
            "provider": "deepseek",
            "status": "disabled",
            "reason": "missing_key",
            "elapsed_ms": 0,
        })
        return ""

    try:
        context = _build_context(
            agent,
            key,
            timeout=max(1.0, min(30.0, float(getattr(cfg, "timeout_seconds", 12.0)))),
            max_results=max(1, min(5, int(getattr(cfg, "max_results", 3)))),
            max_chars=max(400, min(6000, int(getattr(cfg, "max_context_chars", 2800)))),
            fetch_timeout=max(1.0, min(10.0, float(getattr(cfg, "fetch_timeout_seconds", 8.0)))),
        )
        setattr(agent, "web_research_context", context)
        logger.info("[web-research] Search completed: %d context characters", len(context))
        return context
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.warning("[web-research] Search unavailable; continuing offline (%s)", type(exc).__name__)
        setattr(agent, "web_research_context", "")
        return ""
