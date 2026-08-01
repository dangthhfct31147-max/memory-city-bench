"""Cross-encoder reranker wrapper.

A two-stage retrieve-then-rerank pipeline (Sentence Transformers' recommended
design): a fast bi-encoder / lexical stage collects candidates, then a slower but
more accurate cross-encoder rescoring reorders the top-k. The cross-encoder reads
``(query, candidate_text)`` jointly and yields a sharper relevance score than a
single-vector cosine.

This is a *wrapper* retriever: it delegates ``build``/``close`` to any base
retriever (Hybrid, Memory City, …) and only re-scores the candidates the base
returns. That keeps ``MemoryCityRetriever`` untouched and lets the same reranker
sit on top of every base for ablation (``Hybrid + CrossEncoder`` vs
``Memory City + CrossEncoder``).

The cross-encoder model is an OPTIONAL dependency (extra ``embeddings``): if
``sentence_transformers`` is missing the wrapper degrades to a no-op that returns
the base ranking unchanged, recording that fact in ``IndexStats.metadata`` and the
result trace. It is never used to compute retrieval metrics directly — it only
reorders candidates the deterministic base already surfaced.
"""

from __future__ import annotations

import time
from typing import Any

from memcity.retrieval.protocol import (
    IndexStats,
    RetrievalResult,
    RetrievalTrace,
    RetrievedItem,
    Retriever,
)

# Default CPU-friendly MiniLM cross-encoder. Small enough for an 8 GB laptop and a
# reasonable baseline for the two-stage design.
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankRetriever:
    """Wrap a base retriever and rerank its candidates with a cross-encoder.

    Pipeline::

        base.query(candidate_k) → top rerank_k → cross-encoder → final_k

    Args:
        base: any object implementing the ``Retriever`` protocol.
        candidate_k: depth requested from the base retriever's candidate pool.
        rerank_k: how many top candidates to actually score with the cross-encoder
            (the expensive stage — kept small so it never runs on the whole corpus).
        final_k: cap on the returned list when the caller does not request a
            smaller ``top_k``.
        model_name: cross-encoder model id (optional dependency).
        device: torch device for the cross-encoder (``cpu`` by default).
        name: retriever name recorded in artifacts.
    """

    def __init__(
        self,
        base: Retriever,
        *,
        candidate_k: int = 50,
        rerank_k: int = 30,
        final_k: int = 10,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        name: str | None = None,
    ) -> None:
        self._base = base
        self._candidate_k = candidate_k
        self._rerank_k = rerank_k
        self._final_k = final_k
        self._model_name = model_name
        self._device = device
        self.name = name or f"rerank_{getattr(base, 'name', 'base')}"

        self._model: Any = None
        self._available = False

    def _get_model(self) -> Any:
        """Lazy-load the cross-encoder; returns None when the extra is missing."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def build(self, corpus: list[dict], config: Any = None) -> IndexStats:
        stats = self._base.build(corpus, config)
        # Probe availability once at build time so the fallback is recorded in
        # artifacts rather than discovered silently per query.
        try:
            self._get_model()
            self._available = True
        except ImportError:
            self._available = False
        meta = dict(stats.metadata)
        meta.update(
            {
                "reranker": self._model_name,
                "reranker_available": self._available,
                "reranker_candidate_k": self._candidate_k,
                "reranker_rerank_k": self._rerank_k,
                "reranker_final_k": self._final_k,
            }
        )
        return IndexStats(
            method=self.name,
            node_count=stats.node_count,
            edge_count=stats.edge_count,
            community_count=stats.community_count,
            embedding_count=stats.embedding_count,
            index_size_bytes=stats.index_size_bytes,
            build_wall_time_s=stats.build_wall_time_s,
            build_cpu_time_s=stats.build_cpu_time_s,
            metadata=meta,
        )

    def query(
        self,
        query: str,
        top_k: int = 10,
        trace: bool = False,
        candidate_k: int = 100,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        # Pull a deep candidate pool from the base. Depth is the max of the
        # caller's candidate_k and our own so downstream diagnostics stay fair.
        base_depth = max(self._candidate_k, candidate_k, self._rerank_k, top_k)
        base_res = self._base.query(
            query, top_k=base_depth, trace=trace, candidate_k=base_depth
        )

        final_k = min(self._final_k, top_k) if top_k else self._final_k
        pre_rerank_ids = [it.id for it in base_res.items]

        # Fallback: no cross-encoder installed → return base ranking unchanged.
        if not self._available:
            items = base_res.items[:final_k]
            return self._finalize(
                query, items, base_res, pre_rerank_ids, final_k, t0, trace,
                reranked=False,
            )

        # Only the top rerank_k candidates are scored (the expensive stage).
        head = base_res.items[: self._rerank_k]
        tail = base_res.items[self._rerank_k :]

        model = self._get_model()
        pairs = [(query, it.text) for it in head]
        scores = model.predict(pairs) if pairs else []

        rescored: list[RetrievedItem] = []
        for it, ce_score in zip(head, scores, strict=False):
            stage = dict(it.stage_scores)
            stage["rerank"] = float(ce_score)
            rescored.append(
                it.model_copy(update={"score": float(ce_score), "stage_scores": stage})
            )
        rescored.sort(key=lambda x: x.score, reverse=True)

        # Reranked head first, then any untouched tail (keeps a stable pool for
        # deeper top_k requests without letting the tail outrank scored items).
        items = (rescored + tail)[:final_k]
        return self._finalize(
            query, items, base_res, pre_rerank_ids, final_k, t0, trace, reranked=True,
        )

    def _finalize(
        self,
        query: str,
        items: list[RetrievedItem],
        base_res: RetrievalResult,
        pre_rerank_ids: list[str],
        final_k: int,
        t0: float,
        trace: bool,
        *,
        reranked: bool,
    ) -> RetrievalResult:
        latency_ms = (time.perf_counter() - t0) * 1000
        retrieval_trace: RetrievalTrace | None = None
        if trace:
            final_ids = [it.id for it in items]
            base_trace = base_res.trace
            if base_trace is not None:
                retrieval_trace = base_trace.model_copy(
                    update={
                        "pre_rerank": pre_rerank_ids,
                        "post_rerank": final_ids,
                        "final_ranking": final_ids,
                    }
                )
                retrieval_trace.metadata = {
                    **base_trace.metadata,
                    "reranked": reranked,
                    "reranker_available": self._available,
                }
            else:
                retrieval_trace = RetrievalTrace(
                    query=query,
                    pre_rerank=pre_rerank_ids,
                    post_rerank=final_ids,
                    final_ranking=final_ids,
                    metadata={"reranked": reranked, "reranker_available": self._available},
                )
        return RetrievalResult(
            query=query, items=items, top_k=final_k,
            latency_ms=latency_ms, trace=retrieval_trace,
        )

    def close(self) -> None:
        self._base.close()
