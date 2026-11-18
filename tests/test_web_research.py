from __future__ import annotations
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import requests

from agents import web_research


@pytest.fixture(autouse=True)
def clear_research_cache():
    with web_research._cache_lock:
        web_research._cache.clear()
        web_research._inflight.clear()
    yield
    with web_research._cache_lock:
        web_research._cache.clear()
        web_research._inflight.clear()


def make_agent(
    *,
    enabled: bool = True,
    task: str = "Predict the next process outcome",
    query: str = "industrial time series forecasting strong validation baselines",
):
    return SimpleNamespace(
        task_desc=task,
        coldstart_description="",
        data_preview="private local rows must never be sent to public search",
        acfg=SimpleNamespace(
            web_research=SimpleNamespace(
                enabled=enabled,
                query=query,
                timeout_seconds=1.0,
                max_results=3,
                max_context_chars=2800,
            )
        ),
    )


def search_result(*, url: str = "https://example.com/method", summary: str = "A reusable method overview."):
    return {
        "results": [
            {
                "title": "Method overview",
                "url": url,
                "site_name": "Example",
                "summary": summary,
                "rerank_score": 2.0,
                "authority": 3,
            }
        ]
    }


def test_disabled_and_missing_key_never_call_external_service(monkeypatch):
    calls = []
    monkeypatch.delenv("ALGOEVOLVE_WEB_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("AUTODECISION_DEEPSEEK_TOOLS_API_KEY", raising=False)
    monkeypatch.setattr(web_research, "_request_json", lambda *args: calls.append(args))

    assert web_research.research_context_for_agent(make_agent(enabled=False)) == ""
    assert web_research.research_context_for_agent(make_agent()) == ""
    assert calls == []


def test_search_and_fetch_use_explicit_public_query_and_include_fetched_excerpt(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_WEB_SEARCH_API_KEY", "test-only")
    calls = []

    def request(url, key, payload, timeout):
        calls.append((url, key, payload, timeout))
        if url == web_research.SEARCH_URL:
            return search_result()
        return {"content": "Fetched public method details."}

    monkeypatch.setattr(web_research, "_request_json", request)
    private_task = "Predict private plant batch XY-91827 at D:/confidential/input.csv"
    agent = make_agent(task=private_task)
    context = web_research.research_context_for_agent(agent)

    assert len(calls) == 2
    assert calls[0][0] == web_research.SEARCH_URL
    assert calls[0][1] == "test-only"
    assert calls[0][2]["query"] == "industrial time series forecasting strong validation baselines"
    assert calls[0][2]["ori_question"] == calls[0][2]["query"]
    assert calls[0][2]["count"] == 5
    assert private_task not in calls[0][2]["query"]
    assert "private local rows" not in calls[0][2]["query"]
    assert calls[1][0] == web_research.FETCH_URL
    assert calls[1][2] == {"url": "https://example.com/method"}
    assert "Fetched public method details." in context
    assert "untrusted background" in context
    assert agent.web_research_metrics["query_chars"] == len(agent.acfg.web_research.query)
    assert agent.web_research_metrics["search"]["accepted_count"] == 1
    assert agent.web_research_metrics["fetch_attempts"][0]["status"] == "success"
    assert agent.acfg.web_research.query not in str(agent.web_research_metrics)


def test_fetch_failure_keeps_search_fallback_without_trying_another_page(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_WEB_SEARCH_API_KEY", "test-only")
    payload = search_result()
    payload["results"].append({
        "title": "Alternative method",
        "url": "https://backup.example/method",
        "site_name": "Backup",
        "summary": "Alternative search summary.",
        "rerank_score": 1.8,
        "authority": 2,
    })
    calls = []

    def request(url, key, body, timeout):
        calls.append((url, body))
        if url == web_research.SEARCH_URL:
            return payload
        if body["url"] == "https://example.com/method":
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError("not found", response=response)
        return {"content": "Fetched from the alternate public source."}

    monkeypatch.setattr(web_research, "_request_json", request)
    agent = make_agent(query="public method query")
    context = web_research.research_context_for_agent(agent)

    assert [url for url, _ in calls] == [
        web_research.SEARCH_URL,
        web_research.FETCH_URL,
    ]
    assert "Alternative search summary." in context
    assert [attempt["http_status"] for attempt in agent.web_research_metrics["fetch_attempts"]] == [404]
    assert agent.web_research_metrics["fetched_source_count"] == 0


def test_supporting_score_result_is_kept_but_marked_for_verification(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_SEARCH_API_KEY", "test-only")
    payload = {
        "results": [
            {
                "title": "Average precision documentation",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html",
                "site_name": "scikit-learn",
                "summary": "Official API reference for average precision.",
                "rerank_score": 1.02,
                "authority": 3,
            },
            {
                "title": "Weak match",
                "url": "https://example.com/weak-match",
                "summary": "Unrelated.",
                "rerank_score": 0.99,
                "authority": 3,
            },
        ]
    }
    monkeypatch.setattr(web_research, "_request_json", lambda url, *_args: payload)

    agent = make_agent()
    context = web_research.research_context_for_agent(agent)

    assert "Average precision documentation" in context
    assert "supporting lead; verify independently" in context
    assert "Weak match" not in context
    assert agent.web_research_metrics["search"]["high_relevance_count"] == 0
    assert agent.web_research_metrics["search"]["accepted_count"] == 1


def test_search_skips_fetch_when_result_summary_is_sufficient(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_WEB_SEARCH_API_KEY", "test-only")
    calls = []
    summary = "Useful source details. " * 20
    monkeypatch.setattr(
        web_research,
        "_request_json",
        lambda url, *_args: calls.append(url) or search_result(summary=summary),
    )

    agent = make_agent()
    context = web_research.research_context_for_agent(agent)

    assert calls == [web_research.SEARCH_URL]
    assert summary[:100] in context
    assert agent.web_research_metrics["fetch_skipped_reason"] == "search_summary_sufficient"
    assert agent.web_research_metrics["request_counts"] == {"search": 1, "fetch": 0}


def test_duplicate_resource_results_do_not_crowd_out_other_sources(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_WEB_SEARCH_API_KEY", "test-only")
    payload = {
        "results": [
            {
                "title": "Repository README",
                "url": "https://github.com/example/method/blob/main/README.md",
                "summary": "Repository method notes.",
                "rerank_score": 2.2,
                "authority": 3,
            },
            {
                "title": "Repository overview",
                "url": "https://github.com/example/method",
                "summary": "The same repository.",
                "rerank_score": 2.1,
                "authority": 3,
            },
            {
                "title": "Research paper",
                "url": "https://arxiv.org/abs/2601.12345",
                "summary": "A distinct method source.",
                "rerank_score": 1.4,
                "authority": 4,
            },
        ]
    }
    calls = []

    def request(url, _key, _payload, _timeout):
        calls.append(url)
        return payload if url == web_research.SEARCH_URL else {"content": "Fetched repository details."}

    monkeypatch.setattr(web_research, "_request_json", request)
    agent = make_agent()
    agent.acfg.web_research.max_results = 3
    context = web_research.research_context_for_agent(agent)

    assert calls == [web_research.SEARCH_URL, web_research.FETCH_URL]
    assert "Repository README" in context
    assert "Repository overview" not in context
    assert "Research paper" in context
    assert agent.web_research_metrics["search"]["accepted_count"] == 2


@pytest.mark.parametrize("query", ["", " " * 3, "q" * 129])
def test_missing_or_overlong_public_query_never_calls_external_service(monkeypatch, query):
    monkeypatch.setenv("ALGOEVOLVE_WEB_SEARCH_API_KEY", "test-only")
    calls = []
    monkeypatch.setattr(web_research, "_request_json", lambda *args: calls.append(args))

    assert web_research.research_context_for_agent(make_agent(query=query)) == ""
    assert calls == []


def test_fetch_failure_falls_back_to_search_summary(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_SEARCH_API_KEY", "test-only")
    calls = []

    def request(url, key, payload, timeout):
        calls.append(url)
        if url == web_research.SEARCH_URL:
            return search_result(summary="Search summary remains useful.")
        raise requests.Timeout("simulated timeout")

    monkeypatch.setattr(web_research, "_request_json", request)
    context = web_research.research_context_for_agent(make_agent())

    assert calls == [web_research.SEARCH_URL, web_research.FETCH_URL]
    assert "Search summary remains useful." in context


def test_fetch_uses_configured_budget_and_keeps_search_summary(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_WEB_SEARCH_API_KEY", "test-only")
    budgets = []

    def request(url, key, payload, timeout):
        budgets.append((url, timeout))
        if url == web_research.SEARCH_URL:
            return search_result(summary="Useful result when the page is slow.")
        raise requests.Timeout("slow page")

    monkeypatch.setattr(web_research, "_request_json", request)
    agent = make_agent()
    agent.acfg.web_research.timeout_seconds = 12
    context = web_research.research_context_for_agent(agent)

    assert budgets[0] == (web_research.SEARCH_URL, 12.0)
    assert budgets[1] == (web_research.FETCH_URL, 8.0)
    assert "Useful result when the page is slow." in context


def test_autodecision_tools_key_alias_is_supported(monkeypatch):
    monkeypatch.delenv("ALGOEVOLVE_WEB_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("AUTODECISION_DEEPSEEK_TOOLS_API_KEY", "test-alias")
    calls = []
    monkeypatch.setattr(web_research, "_request_json", lambda url, key, *_: calls.append(key) or {"results": []})

    assert web_research.research_context_for_agent(make_agent()) == ""
    assert calls == ["test-alias"]


def test_malformed_search_scores_are_ignored_without_aborting_results(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_SEARCH_API_KEY", "test-only")
    payload = search_result()
    payload["results"].insert(0, {"title": "Malformed", "rerank_score": "not-a-number", "authority": []})
    monkeypatch.setattr(
        web_research,
        "_request_json",
        lambda url, *_args: payload if url == web_research.SEARCH_URL else {"content": "Fetched."},
    )

    context = web_research.research_context_for_agent(make_agent())

    assert "Method overview" in context
    assert "Malformed" not in context


def test_cached_full_context_is_not_truncated_by_first_small_consumer(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_SEARCH_API_KEY", "test-only")
    calls = []
    long_summary = "Useful source detail. " * 100
    monkeypatch.setattr(
        web_research,
        "_request_json",
        lambda url, *_args: calls.append(url) or search_result(summary=long_summary),
    )
    small = make_agent()
    small.acfg.web_research.max_context_chars = 400
    large = make_agent()

    small_context = web_research.research_context_for_agent(small)
    large_context = web_research.research_context_for_agent(large)

    assert len(small_context) == 400
    assert len(large_context) > len(small_context)
    assert calls == [web_research.SEARCH_URL]


def test_empty_search_result_is_cached(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_WEB_SEARCH_API_KEY", "test-only")
    calls = []
    monkeypatch.setattr(web_research, "_request_json", lambda *args: calls.append(args) or {"results": []})

    first = make_agent()
    second = make_agent()
    assert web_research.research_context_for_agent(first) == ""
    assert web_research.research_context_for_agent(second) == ""
    assert len(calls) == 1
    assert first.web_research_metrics["status"] == "no_relevant_results"
    assert second.web_research_metrics["status"] == "cache_hit"
    assert second.web_research_metrics["local_cache"] == "hit"


def test_metrics_are_persisted_without_public_query_or_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_SEARCH_API_KEY", "never-write-this-key")
    monkeypatch.setattr(web_research, "_request_json", lambda *_args: {"results": []})
    agent = make_agent(query="public query that must not be persisted")
    agent.cfg = SimpleNamespace(log_dir=tmp_path)

    web_research.research_context_for_agent(agent)

    path = tmp_path / "web_research_metrics.json"
    receipt = path.read_text(encoding="utf-8")
    payload = json.loads(receipt)
    assert payload["schema_version"] == "algoevolve.web_research_metrics.v1"
    assert payload["stage_id"] == "public_web_research"
    assert payload["status"] == "no_relevant_results"
    assert "public query that must not be persisted" not in receipt
    assert "never-write-this-key" not in receipt


def test_concurrent_agents_share_one_paid_search(monkeypatch):
    monkeypatch.setenv("ALGOEVOLVE_WEB_SEARCH_API_KEY", "test-only")
    calls = []
    calls_lock = threading.Lock()

    def request(url, key, payload, timeout):
        with calls_lock:
            calls.append(url)
        time.sleep(0.08)
        return search_result() if url == web_research.SEARCH_URL else {"content": "Fetched once."}

    monkeypatch.setattr(web_research, "_request_json", request)
    with ThreadPoolExecutor(max_workers=8) as pool:
        contexts = list(pool.map(lambda _: web_research.research_context_for_agent(make_agent()), range(8)))

    assert len(calls) == 2
    assert calls.count(web_research.SEARCH_URL) == 1
    assert calls.count(web_research.FETCH_URL) == 1
    assert all("Method overview" in context for context in contexts)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/private",
        "http://192.168.1.10/private",
        "http://[::1]/private",
        "http://localhost/private",
        "http://service.internal/private",
        "file:///etc/passwd",
        "https://user:pass@example.com/private",
    ],
)
def test_fetch_rejects_non_public_or_unsafe_urls(url):
    assert web_research._public_http_url(url) is False


def test_external_context_is_consumed_only_once():
    agent = make_agent(enabled=False)
    agent._web_research_context_loaded = True
    agent.web_research_context = "Public method references"

    assert web_research.research_context_for_agent(agent, consume=True) == "Public method references"
    assert web_research.research_context_for_agent(agent, consume=True) == ""


def test_concurrent_consume_is_claimed_by_only_one_draft():
    agent = make_agent(enabled=False)
    agent._web_research_context_loaded = True
    agent.web_research_context = "Public method references"

    with ThreadPoolExecutor(max_workers=8) as pool:
        contexts = list(pool.map(lambda _: web_research.research_context_for_agent(agent, consume=True), range(8)))

    assert contexts.count("Public method references") == 1
    assert contexts.count("") == 7


def test_request_deadline_includes_wait_for_response_headers(monkeypatch):
    class Response:
        closed = False

        def raise_for_status(self):
            raise AssertionError("expired requests must not be parsed")

        def close(self):
            self.closed = True

    response = Response()
    clock = iter([10.0, 11.1])
    monkeypatch.setattr(web_research.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(web_research.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(requests.Timeout):
        web_research._request_json(web_research.SEARCH_URL, "test-only", {}, 1.0)

    assert response.closed is True


def test_oversized_response_is_rejected_and_closed(monkeypatch):
    class Response:
        closed = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"x" * (web_research._MAX_RESPONSE_BYTES + 1)

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr(web_research.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(ValueError, match="size limit"):
        web_research._request_json(web_research.SEARCH_URL, "test-only", {}, 1.0)

    assert response.closed is True
