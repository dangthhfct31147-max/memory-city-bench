"""Unit tests for the optional OmniRoute-style gateway reader.

These use httpx.MockTransport so they never require a real localhost server —
safe to run in CI. httpx is an optional extra, so the whole module skips if it
is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

from memcity.readers.reader import (  # noqa: E402
    GATEWAY_NO_CACHE_HEADER,
    OpenAICompatibleReader,
    ResponseCache,
    header_config_hash,
)


def _chat_response(model: str = "qwen3-0.6b", provider: str = "") -> httpx.Response:
    body = {
        "model": model,
        "choices": [
            {"message": {"content": '{"answer": "42", "evidence_ids": ["ep-1"]}'}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    headers = {}
    if provider:
        headers["X-OmniRoute-Provider"] = provider
    return httpx.Response(200, json=body, headers=headers)


def _make_reader(tmp_path: Path, handler, **kwargs) -> OpenAICompatibleReader:
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleReader(
        base_url="http://localhost:20128/v1",
        model="qwen3-0.6b",
        cache_path=tmp_path / "cache.jsonl",
        transport=transport,
        **kwargs,
    )


# ── Auth header ────────────────────────────────────────────────────────────────


def test_authorization_bearer_header_is_sent(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return _chat_response()

    reader = _make_reader(tmp_path, handler, api_key="secret-key-123")
    reader.answer(sample_id="s1", query="q", evidence=[{"id": "ep-1", "text": "x"}])
    assert seen["auth"] == "Bearer secret-key-123"


def test_no_auth_header_when_no_key(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return _chat_response()

    reader = _make_reader(tmp_path, handler)
    reader.answer(sample_id="s1", query="q", evidence=[{"id": "ep-1", "text": "x"}])
    assert seen["auth"] is None


# ── No-cache + extra headers ────────────────────────────────────────────────────


def test_gateway_no_cache_header_sent(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[GATEWAY_NO_CACHE_HEADER] = request.headers.get(GATEWAY_NO_CACHE_HEADER)
        return _chat_response()

    reader = _make_reader(tmp_path, handler, gateway_no_cache=True)
    reader.answer(sample_id="s1", query="q", evidence=[])
    assert seen[GATEWAY_NO_CACHE_HEADER] == "true"


def test_extra_headers_forwarded(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["route"] = request.headers.get("X-OmniRoute-Route")
        return _chat_response()

    reader = _make_reader(tmp_path, handler, extra_headers={"X-OmniRoute-Route": "pinned"})
    reader.answer(sample_id="s1", query="q", evidence=[])
    assert seen["route"] == "pinned"


# ── Answer parsing ───────────────────────────────────────────────────────────────


def test_answer_parses_and_records_effective_model(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(model="qwen3-0.6b", provider="local")

    reader = _make_reader(tmp_path, handler)
    rr = reader.answer(sample_id="s1", query="q", evidence=[{"id": "ep-1", "text": "x"}])
    assert rr.schema_ok is True
    assert rr.answer is not None and rr.answer.answer == "42"
    assert rr.requested_model == "qwen3-0.6b"
    assert rr.effective_model == "qwen3-0.6b"
    assert rr.effective_provider == "local"
    assert rr.fallback_warning is False


def test_fallback_warning_when_model_differs(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        # Gateway served a different model than requested → fallback.
        return _chat_response(model="gpt-4o-mini", provider="openai")

    reader = _make_reader(tmp_path, handler)
    rr = reader.answer(sample_id="s1", query="q", evidence=[])
    assert rr.effective_model == "gpt-4o-mini"
    assert rr.fallback_warning is True


# ── Health check ─────────────────────────────────────────────────────────────────


def test_health_check_reads_models(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [{"id": "qwen3-0.6b"}, {"id": "other"}]})

    reader = _make_reader(tmp_path, handler)
    health = reader.health_check()
    assert health["ok"] is True
    assert health["model_available"] is True
    assert "qwen3-0.6b" in health["models"]


def test_health_check_reports_missing_model(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "some-other-model"}]})

    reader = _make_reader(tmp_path, handler)
    health = reader.health_check()
    assert health["ok"] is True
    assert health["model_available"] is False


def test_health_check_handles_unreachable(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    reader = _make_reader(tmp_path, handler)
    health = reader.health_check()
    assert health["ok"] is False
    assert health["model_available"] is False


# ── Caching ──────────────────────────────────────────────────────────────────────


def test_second_identical_call_hits_cache(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _chat_response()

    reader = _make_reader(tmp_path, handler)
    r1 = reader.answer(sample_id="s1", query="q", evidence=[{"id": "ep-1", "text": "x"}])
    r2 = reader.answer(sample_id="s2", query="q", evidence=[{"id": "ep-1", "text": "x"}])
    assert calls["n"] == 1  # second call served from cache
    assert r1.cache_hit is False
    assert r2.cache_hit is True
    # A cache hit performs no upstream call.
    assert r2.upstream_latency_ms == 0.0
    # First (real) call records upstream latency, not a local lookup.
    assert r1.upstream_latency_ms >= 0.0


def test_temperature_change_busts_cache(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _chat_response()

    cache = tmp_path / "cache.jsonl"
    r1 = OpenAICompatibleReader(
        base_url="http://localhost:20128/v1", model="qwen3-0.6b",
        temperature=0.0, cache_path=cache, transport=httpx.MockTransport(handler),
    )
    r1.answer(sample_id="s1", query="q", evidence=[])
    r2 = OpenAICompatibleReader(
        base_url="http://localhost:20128/v1", model="qwen3-0.6b",
        temperature=0.7, cache_path=cache, transport=httpx.MockTransport(handler),
    )
    r2.answer(sample_id="s1", query="q", evidence=[])
    assert calls["n"] == 2  # different temperature → different key → new request


def test_max_tokens_and_seed_are_in_cache_key(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _chat_response()

    cache = tmp_path / "cache.jsonl"
    base = dict(base_url="http://localhost:20128/v1", model="qwen3-0.6b", cache_path=cache)
    OpenAICompatibleReader(**base, seed=1, max_tokens=64,
                           transport=httpx.MockTransport(handler)).answer("s", "q", [])
    OpenAICompatibleReader(**base, seed=2, max_tokens=64,
                           transport=httpx.MockTransport(handler)).answer("s", "q", [])
    OpenAICompatibleReader(**base, seed=2, max_tokens=128,
                           transport=httpx.MockTransport(handler)).answer("s", "q", [])
    assert calls["n"] == 3  # each distinct (seed, max_tokens) is a separate key


def test_header_config_hash_changes_cache_key(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _chat_response()

    cache = tmp_path / "cache.jsonl"
    OpenAICompatibleReader(
        base_url="http://localhost:20128/v1", model="qwen3-0.6b", cache_path=cache,
        extra_headers={"X-A": "1"}, transport=httpx.MockTransport(handler),
    ).answer("s", "q", [])
    OpenAICompatibleReader(
        base_url="http://localhost:20128/v1", model="qwen3-0.6b", cache_path=cache,
        extra_headers={"X-A": "2"}, transport=httpx.MockTransport(handler),
    ).answer("s", "q", [])
    assert calls["n"] == 2  # different header config → different cache key


# ── Secret hygiene ───────────────────────────────────────────────────────────────


def test_api_key_never_in_cache_file(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response()

    cache_path = tmp_path / "cache.jsonl"
    reader = _make_reader(tmp_path, handler, api_key="TOP-SECRET-KEY")
    reader.answer(sample_id="s1", query="q", evidence=[{"id": "ep-1", "text": "x"}])
    contents = cache_path.read_text(encoding="utf-8")
    assert "TOP-SECRET-KEY" not in contents


def test_api_key_never_in_repr(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response()

    reader = _make_reader(tmp_path, handler, api_key="TOP-SECRET-KEY")
    assert "TOP-SECRET-KEY" not in repr(reader)


def test_api_key_redacted_from_errors(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.HTTPError("auth failed for token TOP-SECRET-KEY")

    reader = _make_reader(tmp_path, handler, api_key="TOP-SECRET-KEY")
    rr = reader.answer(sample_id="s1", query="q", evidence=[])
    assert "TOP-SECRET-KEY" not in rr.error
    assert "***REDACTED***" in rr.error


def test_header_config_hash_excludes_key():
    # Two hashes with the same headers/flags are identical regardless of any key
    # (the key is never an input to the hash).
    h1 = header_config_hash({"X-A": "1"}, auth_enabled=True, gateway_no_cache=True)
    h2 = header_config_hash({"X-A": "1"}, auth_enabled=True, gateway_no_cache=True)
    assert h1 == h2
    # auth_enabled flag flips the hash.
    h3 = header_config_hash({"X-A": "1"}, auth_enabled=False, gateway_no_cache=True)
    assert h1 != h3


def test_cache_key_is_stable_and_field_sensitive():
    base = {
        "base_url": "u", "model": "m", "temperature": 0.0, "seed": 42,
        "max_tokens": 256, "prompt": "p", "header_config_hash": "h",
    }
    k1 = ResponseCache.make_key(base)
    k2 = ResponseCache.make_key(dict(base))
    assert k1 == k2
    k3 = ResponseCache.make_key({**base, "prompt": "different"})
    assert k1 != k3


# ── Cache modes ──────────────────────────────────────────────────────────────────


def test_cache_mode_off_never_reads_or_writes(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _chat_response()

    cache = tmp_path / "cache.jsonl"
    kwargs = dict(base_url="http://localhost:20128/v1", model="qwen3-0.6b", cache_path=cache)
    OpenAICompatibleReader(**kwargs, cache_mode="off",
                           transport=httpx.MockTransport(handler)).answer("s", "q", [])
    r2 = OpenAICompatibleReader(**kwargs, cache_mode="off",
                                transport=httpx.MockTransport(handler)).answer("s", "q", [])
    assert calls["n"] == 2  # never served from disk
    assert r2.cache_hit is False
    assert not cache.exists() or cache.read_text(encoding="utf-8").strip() == ""


def test_cache_mode_read_only_reads_but_does_not_write(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _chat_response()

    cache = tmp_path / "cache.jsonl"
    kwargs = dict(base_url="http://localhost:20128/v1", model="qwen3-0.6b", cache_path=cache)
    # Prime the cache with a read-write reader.
    OpenAICompatibleReader(**kwargs, cache_mode="read-write",
                           transport=httpx.MockTransport(handler)).answer("s", "q", [])
    assert calls["n"] == 1
    # read-only reader serves the primed entry (no new call)...
    r2 = OpenAICompatibleReader(**kwargs, cache_mode="read-only",
                                transport=httpx.MockTransport(handler)).answer("s", "q", [])
    assert r2.cache_hit is True
    assert calls["n"] == 1
    # ...but a fresh prompt is fetched and NOT written back.
    r3reader = OpenAICompatibleReader(**kwargs, cache_mode="read-only",
                                      transport=httpx.MockTransport(handler))
    r3reader.answer("s", "different question", [])
    assert calls["n"] == 2
    # Re-reading that prompt still misses because read-only never wrote it.
    OpenAICompatibleReader(**kwargs, cache_mode="read-only",
                           transport=httpx.MockTransport(handler)).answer("s", "different question", [])
    assert calls["n"] == 3


def test_invalid_cache_mode_raises(tmp_path):
    with pytest.raises(ValueError):
        OpenAICompatibleReader(
            base_url="http://localhost:20128/v1", model="m",
            cache_path=tmp_path / "c.jsonl", cache_mode="bogus",
        )


# ── Provider-locked route ────────────────────────────────────────────────────────


def test_omniroute_provider_changes_chat_url(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return _chat_response()

    reader = _make_reader(tmp_path, handler, omniroute_provider="openai")
    assert reader.chat_completions_url.endswith("/v1/providers/openai/chat/completions")
    # Health check stays on the plain /models endpoint.
    assert reader.models_url.endswith("/v1/models")
    reader.answer(sample_id="s1", query="q", evidence=[])
    assert seen["path"].endswith("/v1/providers/openai/chat/completions")


def test_effective_provider_falls_back_to_pinned_provider(tmp_path):
    # No X-OmniRoute-Provider header, but a provider was pinned → best-effort
    # attribution uses the pinned id.
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response()  # no provider header

    reader = _make_reader(tmp_path, handler, omniroute_provider="anthropic")
    rr = reader.answer(sample_id="s1", query="q", evidence=[])
    assert rr.effective_provider == "anthropic"


def test_provider_mismatch_400_is_surfaced_as_error(tmp_path):
    # A provider-locked route returns 400 when the model is not served by that
    # provider — this must become a clean error, not a silent fallback.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "model not available for provider"})

    reader = _make_reader(tmp_path, handler, omniroute_provider="openai")
    rr = reader.answer(sample_id="s1", query="q", evidence=[])
    assert rr.answer is None
    assert rr.error != ""
    assert rr.cache_hit is False
