"""Unit tests for BM25 and Hybrid retrievers."""

from __future__ import annotations

import pytest

from memcity.retrieval.baselines import (
    BM25Retriever,
    HybridRRFRetriever,
    RandomRetriever,
    RecencyRetriever,
)


CORPUS = [
    {
        "id": f"ep_{i:03d}",
        "node_type": "episode",
        "text": text,
        "user_text": text,
        "assistant_text": "",
        "session_id": "s1",
        "timestamp": float(i),
        "source_episode_ids": [f"ep_{i:03d}"],
    }
    for i, text in enumerate([
        "Alice uses Python for data science",
        "Bob works with FastAPI on the backend",
        "Project Alpha status is in planning",
        "The database version is PostgreSQL 14",
        "Alice is the senior engineer",
        "Bob is located in Berlin",
        "Project Alpha deadline is Q3 2025",
        "Database host is db.internal",
    ])
]


class TestBM25:
    @pytest.fixture
    def bm25(self):
        r = BM25Retriever()
        r.build(CORPUS)
        return r

    def test_returns_top_k(self, bm25):
        result = bm25.query("Alice Python", top_k=3)
        assert len(result.items) == 3

    def test_relevant_first(self, bm25):
        result = bm25.query("Alice Python", top_k=5)
        ids = result.ids()
        assert "ep_000" in ids[:3]  # Alice Python is ep_000

    def test_latency_positive(self, bm25):
        result = bm25.query("database version", top_k=5)
        assert result.latency_ms > 0

    def test_episode_ids_populated(self, bm25):
        result = bm25.query("FastAPI", top_k=3)
        assert all(item.source_episode_ids for item in result.items)


class TestRecency:
    def test_returns_most_recent_first(self):
        r = RecencyRetriever()
        r.build(CORPUS)
        result = r.query("anything", top_k=3)
        ts = [item.metadata.get("timestamp", 0) for item in result.items]
        # Top result should be highest timestamp
        scores = [item.score for item in result.items]
        assert scores[0] >= scores[-1]

    def test_returns_top_k(self):
        r = RecencyRetriever()
        r.build(CORPUS)
        result = r.query("x", top_k=3)
        assert len(result.items) == 3


class TestRandom:
    def test_deterministic_with_seed(self):
        r1 = RandomRetriever(seed=42)
        r2 = RandomRetriever(seed=42)
        r1.build(CORPUS)
        r2.build(CORPUS)
        res1 = r1.query("test", top_k=5)
        res2 = r2.query("test", top_k=5)
        assert res1.ids() == res2.ids()


class TestHybridRRF:
    @pytest.fixture
    def hybrid(self):
        r = HybridRRFRetriever(rrf_k=60)
        r.build(CORPUS)
        return r

    def test_returns_results(self, hybrid):
        result = hybrid.query("Alice Python", top_k=5)
        assert len(result.items) > 0

    def test_relevant_in_top_results(self, hybrid):
        result = hybrid.query("Alice Python", top_k=5)
        ids = result.ids()
        assert "ep_000" in ids


class TestRetrieverProtocol:
    """Verify all retrievers implement the required interface."""

    def test_all_have_name(self):
        from memcity.retrieval.baselines import BASELINE_REGISTRY
        for name, cls in BASELINE_REGISTRY.items():
            assert hasattr(cls, "name") or hasattr(cls(), "name")

    def test_all_have_build_query_close(self):
        from memcity.retrieval.baselines import BASELINE_REGISTRY
        for name, cls in BASELINE_REGISTRY.items():
            if name == "oracle":
                continue
            inst = cls()
            assert hasattr(inst, "build")
            assert hasattr(inst, "query")
            assert hasattr(inst, "close")
