"""Unit tests for the cross-encoder reranker wrapper.

The reranker itself needs sentence-transformers (extra ``embeddings``). To keep
the core suite dependency-free we test two layers separately:

* Fallback / plumbing tests use a stub base retriever and a stub cross-encoder,
  so they run everywhere and never touch the optional dependency.
* One test is gated behind ``importorskip('sentence_transformers')`` for the real
  model path.
"""

from __future__ import annotations

from typing import Any

import pytest

from memcity.retrieval.protocol import (
    IndexStats,
    RetrievalResult,
    RetrievalTrace,
    RetrievedItem,
)
from memcity.retrieval.rerank import RerankRetriever


class StubBase:
    """A base retriever whose ranking we control, to verify reranking effects."""

    name = "stub"

    def __init__(self, items: list[RetrievedItem]) -> None:
        self._items = items
        self.built = False
        self.closed = False

    def build(self, corpus: list[dict], config: Any = None) -> IndexStats:
        self.built = True
        return IndexStats(method=self.name, node_count=len(self._items))

    def query(
        self, query: str, top_k: int = 10, trace: bool = False, candidate_k: int = 100
    ) -> RetrievalResult:
        items = self._items[:top_k]
        tr = None
        if trace:
            ids = [it.id for it in self._items[:candidate_k]]
            tr = RetrievalTrace(query=query, union_candidates=ids, pre_rerank=ids)
        return RetrievalResult(query=query, items=list(items), top_k=top_k, trace=tr)

    def close(self) -> None:
        self.closed = True


class StubCrossEncoder:
    """Scores pairs by a lookup on the candidate text (higher = more relevant)."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores = scores_by_text

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._scores.get(text, 0.0) for _q, text in pairs]


def _items(texts: list[str]) -> list[RetrievedItem]:
    return [
        RetrievedItem(id=f"ep_{i}", text=t, score=float(len(texts) - i),
                      source_episode_ids=[f"ep_{i}"])
        for i, t in enumerate(texts)
    ]


def test_fallback_is_passthrough_when_model_missing():
    base = StubBase(_items(["a", "b", "c", "d"]))
    r = RerankRetriever(base, candidate_k=4, rerank_k=4, final_k=3)
    r.build([])
    # Simulate the optional dependency being absent.
    r._available = False
    res = r.query("q", top_k=3, trace=True)
    assert [it.id for it in res.items] == ["ep_0", "ep_1", "ep_2"]
    assert res.trace is not None
    assert res.trace.metadata["reranked"] is False


def test_reranker_promotes_buried_evidence():
    # Evidence text "gold" starts at rank 3 (last) in the base ranking.
    base = StubBase(_items(["noise0", "noise1", "noise2", "gold"]))
    r = RerankRetriever(base, candidate_k=4, rerank_k=4, final_k=2)
    r.build([])
    r._available = True
    r._model = StubCrossEncoder({"gold": 9.0, "noise0": 0.1, "noise1": 0.2, "noise2": 0.3})
    res = r.query("find the gold", top_k=2, trace=True)
    ids = [it.id for it in res.items]
    assert ids[0] == "ep_3"  # gold promoted to the top
    assert res.trace is not None
    assert res.trace.metadata["reranked"] is True
    assert res.trace.pre_rerank[:1] == ["ep_0"]
    assert res.trace.post_rerank[0] == "ep_3"
    assert res.items[0].stage_scores["rerank"] == 9.0


def test_only_rerank_k_scored():
    base = StubBase(_items(["a", "b", "c", "d", "e"]))
    r = RerankRetriever(base, candidate_k=5, rerank_k=2, final_k=5)
    r.build([])
    r._available = True
    # Only the first two candidates are passed to predict.
    seen: list[str] = []

    class SpyEncoder:
        def predict(self, pairs):
            seen.extend(t for _q, t in pairs)
            return [1.0 for _ in pairs]

    r._model = SpyEncoder()
    r.query("q", top_k=5)
    assert seen == ["a", "b"]


def test_close_delegates_to_base():
    base = StubBase(_items(["a"]))
    r = RerankRetriever(base)
    r.close()
    assert base.closed is True


def test_registry_exposes_rerank_methods():
    from memcity.retrieval.registry import get_retriever_factory, list_methods

    assert "rerank_hybrid" in list_methods()
    assert "rerank_memory_city" in list_methods()
    # Factory builds without importing the optional model (probe is lazy).
    factory = get_retriever_factory("rerank_hybrid")
    inst = factory()
    assert inst.name == "rerank_hybrid"


@pytest.mark.slow
def test_real_cross_encoder_smoke():
    pytest.importorskip("sentence_transformers")
    base = StubBase(_items([
        "The capital of France is Paris.",
        "Bananas are yellow fruit.",
        "Python is a programming language.",
    ]))
    r = RerankRetriever(base, candidate_k=3, rerank_k=3, final_k=3)
    r.build([])
    assert r._available is True
    res = r.query("What is the capital of France?", top_k=3)
    assert res.items[0].id == "ep_0"
