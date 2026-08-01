"""Unit tests for Phase 5 KV/prefix-cache metrics.

Tests are fully offline — no real HTTP call is made. We use httpx's mock
transport (MockTransport) to inject synthetic server responses containing
various ``usage`` payloads, covering:

* OpenAI-style prompt_tokens_details.cached_tokens
* llama.cpp-style usage.timings
* backend that reports nothing (all fields should be None)
* prefix_cache_hit derivation (True/False/None)
* prompt_prefix_hash stability across different queries
* compute_e2e_metrics aggregation of cold/warm/decode columns
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("httpx")

from memcity.evaluation.e2e_metrics import compute_e2e_metrics
from memcity.readers.reader import (
    PROMPT_PREFIX_HASH,
    OpenAICompatibleReader,
    _extract_cache_metrics,
    build_prompt,
)

# ── _extract_cache_metrics unit tests ────────────────────────────────────────

def test_openai_style_cached_tokens():
    usage = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 64}}
    m = _extract_cache_metrics(usage)
    assert m["cached_prompt_tokens"] == 64
    assert m["prefix_cache_hit"] is True
    assert m["prompt_eval_ms"] is None


def test_openai_style_zero_cached_tokens_is_false():
    usage = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}}
    m = _extract_cache_metrics(usage)
    assert m["cached_prompt_tokens"] == 0
    assert m["prefix_cache_hit"] is False


def test_llamacpp_timings():
    usage = {
        "timings": {
            "prompt_ms": 120.5,
            "prompt_n_cached": 32,
            "predicted_ms": 450.0,
            "prompt_per_second": 640.0,
            "predicted_per_second": 24.5,
        }
    }
    m = _extract_cache_metrics(usage)
    assert m["prompt_eval_ms"] == pytest.approx(120.5)
    assert m["decode_ms"] == pytest.approx(450.0)
    assert m["cached_prompt_tokens"] == 32
    assert m["prompt_tokens_per_second"] == pytest.approx(640.0)
    assert m["generation_tokens_per_second"] == pytest.approx(24.5)
    assert m["prefix_cache_hit"] is True


def test_no_cache_fields_gives_all_none():
    usage = {"prompt_tokens": 50, "completion_tokens": 20}
    m = _extract_cache_metrics(usage)
    assert m["cached_prompt_tokens"] is None
    assert m["prefix_cache_hit"] is None
    assert m["prompt_eval_ms"] is None
    assert m["decode_ms"] is None


# ── build_prompt prefix hash ─────────────────────────────────────────────────

def test_prompt_prefix_hash_stable_across_queries():
    p1 = build_prompt("Who is Alice?", [{"id": "e1", "text": "Alice likes cats."}], 1024)
    p2 = build_prompt("What does Bob do?", [{"id": "e2", "text": "Bob codes."}], 1024)
    assert PROMPT_PREFIX_HASH != ""
    # Both prompts start with the same system prompt prefix.
    from memcity.readers.reader import _SYSTEM_PROMPT
    assert p1.startswith(_SYSTEM_PROMPT)
    assert p2.startswith(_SYSTEM_PROMPT)
    # The hash at module level matches the actual system prompt.
    import hashlib
    expected = hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest()
    assert PROMPT_PREFIX_HASH == expected


# ── OpenAICompatibleReader end-to-end via mock transport ─────────────────────

def _make_transport(usage: dict):
    """Return an httpx mock transport that injects a synthetic chat completion."""
    import httpx

    body = json.dumps({
        "choices": [{"message": {"content": '{"answer":"Paris","evidence_ids":["e1"],"abstained":false,"confidence":1.0}'}}],
        "model": "test-model",
        "usage": usage,
    }).encode()

    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request):
            return httpx.Response(200, content=body)

    return FakeTransport()


def _reader(usage: dict, tmp_path) -> OpenAICompatibleReader:
    return OpenAICompatibleReader(
        base_url="http://fake",
        model="test-model",
        cache_path=tmp_path / "cache.jsonl",
        cache_mode="off",
        transport=_make_transport(usage),
    )


def test_reader_captures_openai_cached_tokens(tmp_path):
    r = _reader({"prompt_tokens": 80, "prompt_tokens_details": {"cached_tokens": 48}}, tmp_path)
    res = r.answer("s1", "Capital of France?", [{"id": "e1", "text": "France capital is Paris."}])
    assert res.cached_prompt_tokens == 48
    assert res.prefix_cache_hit is True
    assert res.prompt_prefix_hash == PROMPT_PREFIX_HASH


def test_reader_captures_llamacpp_timings(tmp_path):
    usage = {
        "prompt_tokens": 60,
        "timings": {"prompt_ms": 90.0, "predicted_ms": 200.0, "prompt_n_cached": 0},
    }
    r = _reader(usage, tmp_path)
    res = r.answer("s2", "Some question?", [{"id": "e2", "text": "Some answer."}])
    assert res.prompt_eval_ms == pytest.approx(90.0)
    assert res.decode_ms == pytest.approx(200.0)
    assert res.prefix_cache_hit is False


def test_reader_none_when_backend_silent(tmp_path):
    r = _reader({"prompt_tokens": 50, "completion_tokens": 10}, tmp_path)
    res = r.answer("s3", "Anything?", [])
    assert res.cached_prompt_tokens is None
    assert res.prefix_cache_hit is None
    assert res.prompt_eval_ms is None
    # prompt_prefix_hash is always populated
    assert res.prompt_prefix_hash == PROMPT_PREFIX_HASH


# ── compute_e2e_metrics aggregation ─────────────────────────────────────────

def _stub_sample(sample_id: str, answer: str):
    class S:
        def __init__(self):
            self.sample_id = sample_id
            self.answer = answer
            self.evidence_episode_ids = []
            self.category = type("C", (), {"value": "direct_fact"})()
    return S()


def test_e2e_metrics_aggregates_cache_columns():
    results = [
        {
            "sample_id": "q1", "schema_ok": True, "latency_ms": 300.0,
            "prompt_tokens": 50, "completion_tokens": 10,
            "answer": {"answer": "Paris", "evidence_ids": [], "abstained": False, "confidence": 1.0},
            "prompt_eval_ms": 120.0, "decode_ms": 180.0,
            "cached_prompt_tokens": 32, "prefix_cache_hit": True,
        },
        {
            "sample_id": "q2", "schema_ok": True, "latency_ms": 400.0,
            "prompt_tokens": 60, "completion_tokens": 12,
            "answer": {"answer": "Berlin", "evidence_ids": [], "abstained": False, "confidence": 1.0},
            "prompt_eval_ms": 200.0, "decode_ms": 200.0,
            "cached_prompt_tokens": 0, "prefix_cache_hit": False,
        },
    ]
    samples = [_stub_sample("q1", "paris"), _stub_sample("q2", "berlin")]
    m = compute_e2e_metrics(results, samples)

    assert "prompt_eval_ms_mean" in m
    assert m["prompt_eval_ms_mean"] == pytest.approx(160.0)
    assert "decode_ms_mean" in m
    assert m["decode_ms_mean"] == pytest.approx(190.0)
    assert "prefix_cache_hit_rate" in m
    assert m["prefix_cache_hit_rate"] == pytest.approx(0.5)
    assert m["prefix_cache_reported_rate"] == pytest.approx(1.0)
    assert "cached_prompt_tokens_mean" in m
    assert m["cached_prompt_tokens_mean"] == pytest.approx(16.0)


def test_e2e_metrics_omits_cache_columns_when_not_reported():
    results = [
        {
            "sample_id": "q1", "schema_ok": True, "latency_ms": 100.0,
            "prompt_tokens": 30, "completion_tokens": 5,
            "answer": {"answer": "x", "evidence_ids": [], "abstained": False, "confidence": 1.0},
        },
    ]
    samples = [_stub_sample("q1", "x")]
    m = compute_e2e_metrics(results, samples)
    # When backend reports nothing these keys must be absent — not 0.
    assert "prompt_eval_ms_mean" not in m
    assert "decode_ms_mean" not in m
    assert "prefix_cache_hit_rate" not in m
