"""Baseline retrievers: random, recency, BM25, vector (numpy cosine), hybrid RRF, oracle."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from memcity.evaluation.metrics import compute_rrf
from memcity.retrieval.protocol import IndexStats, RetrievalResult, RetrievalTrace, RetrievedItem
from memcity.utils.helpers import tokenize

# ── Corpus item helpers ──────────────────────────────────────────────────────

def _item_text(item: dict) -> str:
    parts = []
    for key in ("text", "user_text", "assistant_text"):
        v = item.get(key, "")
        if v:
            parts.append(v)
    return " ".join(parts)


def _index_text(item: dict) -> str:
    """Text fed to the index. Prefers the contextual ``indexed_text`` (Phase 4)
    when present, else falls back to the raw episode text. Retrieved evidence
    always uses ``_item_text`` (raw), so a context prefix never leaks into
    results or citations."""
    return item.get("indexed_text") or _item_text(item)


def _item_ts(item: dict) -> float:
    return float(item.get("timestamp", 0.0))


def _ep_ids(item: dict) -> list[str]:
    ep = item.get("source_episode_ids") or []
    if not ep and item.get("node_type", "episode") == "episode":
        ep = [item["id"]]
    return ep


def _make_retrieved(item: dict, score: float, stage_scores: dict | None = None) -> RetrievedItem:
    return RetrievedItem(
        id=item["id"],
        text=_item_text(item),
        score=score,
        source_episode_ids=_ep_ids(item),
        node_type=item.get("node_type", "episode"),
        stage_scores=stage_scores or {},
    )


# ── Random Retriever ─────────────────────────────────────────────────────────

class RandomRetriever:
    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._corpus: list[dict] = []

    def build(self, corpus: list[dict], config: Any = None) -> IndexStats:
        t0 = time.perf_counter()
        self._corpus = list(corpus)
        return IndexStats(
            method=self.name,
            node_count=len(self._corpus),
            build_wall_time_s=time.perf_counter() - t0,
        )

    def query(
        self, query: str, top_k: int = 10, trace: bool = False, candidate_k: int = 100
    ) -> RetrievalResult:
        import random
        rng = random.Random(self._seed ^ hash(query))
        t0 = time.perf_counter()
        selected = rng.sample(self._corpus, min(top_k, len(self._corpus)))
        items = [_make_retrieved(it, 0.0) for it in selected]
        return RetrievalResult(
            query=query, items=items, top_k=top_k,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def close(self) -> None:
        pass


# ── Recency Retriever ─────────────────────────────────────────────────────────

class RecencyRetriever:
    name = "recency"

    def __init__(self) -> None:
        self._corpus: list[dict] = []

    def build(self, corpus: list[dict], config: Any = None) -> IndexStats:
        t0 = time.perf_counter()
        self._corpus = sorted(corpus, key=_item_ts, reverse=True)
        return IndexStats(
            method=self.name,
            node_count=len(self._corpus),
            build_wall_time_s=time.perf_counter() - t0,
        )

    def query(
        self, query: str, top_k: int = 10, trace: bool = False, candidate_k: int = 100
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        items = [_make_retrieved(it, float(len(self._corpus) - i))
                 for i, it in enumerate(self._corpus[:top_k])]
        retrieval_trace: RetrievalTrace | None = None
        if trace:
            ranked = [it["id"] for it in self._corpus[:candidate_k]]
            retrieval_trace = RetrievalTrace(
                query=query,
                union_candidates=ranked,
                pre_rerank=ranked,
                post_rerank=[it.id for it in items],
            )
        return RetrievalResult(
            query=query, items=items, top_k=top_k,
            latency_ms=(time.perf_counter() - t0) * 1000, trace=retrieval_trace,
        )

    def close(self) -> None:
        pass


# ── BM25 Retriever ────────────────────────────────────────────────────────────

class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._corpus: list[dict] = []
        self._bm25: Any = None

    def build(self, corpus: list[dict], config: Any = None) -> IndexStats:
        from rank_bm25 import BM25Okapi
        t0 = time.perf_counter()
        self._corpus = list(corpus)
        tokenized = [tokenize(_index_text(it)) for it in self._corpus]
        self._bm25 = BM25Okapi(tokenized, k1=self._k1, b=self._b)
        return IndexStats(
            method=self.name,
            node_count=len(self._corpus),
            build_wall_time_s=time.perf_counter() - t0,
        )

    def query(
        self, query: str, top_k: int = 10, trace: bool = False, candidate_k: int = 100
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Retrieve candidate_k for trace; final result is top_k
        cand_size = max(top_k, candidate_k)
        cand_idx = np.argsort(scores)[::-1][:cand_size]
        final_idx = cand_idx[:top_k]
        items = [_make_retrieved(self._corpus[i], float(scores[i])) for i in final_idx]
        retrieval_trace: RetrievalTrace | None = None
        if trace:
            cand_ids = [self._corpus[i]["id"] for i in cand_idx]
            final_ids = [it.id for it in items]
            retrieval_trace = RetrievalTrace(
                query=query,
                bm25_candidates=cand_ids,
                candidate_pool=cand_ids,
                union_candidates=cand_ids,
                pre_rerank=cand_ids,
                post_rerank=final_ids,
                final_ranking=final_ids,
                candidate_k=candidate_k,
            )
        return RetrievalResult(
            query=query, items=items, top_k=top_k,
            latency_ms=(time.perf_counter() - t0) * 1000, trace=retrieval_trace,
        )

    def close(self) -> None:
        pass


# ── Vector Retriever (numpy cosine) ──────────────────────────────────────────

class VectorRetriever:
    name = "vector"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._corpus: list[dict] = []
        self._matrix: np.ndarray | None = None
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False,
                            normalize_embeddings=True)
        return vecs.astype(np.float32)

    def build(self, corpus: list[dict], config: Any = None) -> IndexStats:
        t0 = time.perf_counter()
        self._corpus = list(corpus)
        texts = [_index_text(it) for it in self._corpus]
        self._matrix = self._embed(texts)
        return IndexStats(
            method=self.name,
            node_count=len(self._corpus),
            embedding_count=len(self._corpus),
            build_wall_time_s=time.perf_counter() - t0,
        )

    def query(
        self, query: str, top_k: int = 10, trace: bool = False, candidate_k: int = 100
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        q_vec = self._embed([query])[0]
        sims = (self._matrix @ q_vec).tolist()
        cand_size = max(top_k, candidate_k)
        cand_idx = np.argsort(sims)[::-1][:cand_size]
        final_idx = cand_idx[:top_k]
        items = [_make_retrieved(self._corpus[i], float(sims[i])) for i in final_idx]
        retrieval_trace: RetrievalTrace | None = None
        if trace:
            cand_ids = [self._corpus[i]["id"] for i in cand_idx]
            final_ids = [it.id for it in items]
            retrieval_trace = RetrievalTrace(
                query=query,
                vector_candidates=cand_ids,
                candidate_pool=cand_ids,
                union_candidates=cand_ids,
                pre_rerank=cand_ids,
                post_rerank=final_ids,
                final_ranking=final_ids,
                candidate_k=candidate_k,
            )
        return RetrievalResult(
            query=query, items=items, top_k=top_k,
            latency_ms=(time.perf_counter() - t0) * 1000, trace=retrieval_trace,
        )

    def close(self) -> None:
        pass


# ── Hybrid RRF Retriever ──────────────────────────────────────────────────────

class HybridRRFRetriever:
    name = "hybrid_rrf"

    def __init__(self, rrf_k: int = 60, **kwargs) -> None:
        self._k = rrf_k
        self._bm25 = BM25Retriever()
        self._vector: VectorRetriever | None = None
        self._has_vector = False

    def build(self, corpus: list[dict], config: Any = None) -> IndexStats:
        t0 = time.perf_counter()
        bm25_stats = self._bm25.build(corpus, config)
        try:
            self._vector = VectorRetriever()
            self._vector.build(corpus, config)
            self._has_vector = True
        except ImportError:
            self._has_vector = False
        return IndexStats(
            method=self.name,
            node_count=bm25_stats.node_count,
            embedding_count=bm25_stats.node_count if self._has_vector else 0,
            build_wall_time_s=time.perf_counter() - t0,
        )

    def query(
        self, query: str, top_k: int = 10, trace: bool = False, candidate_k: int = 100
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        # Use candidate_k for depth on each sub-ranker so candidate pool is measured fairly.
        cand_size = max(top_k, candidate_k)

        bm25_res = self._bm25.query(query, cand_size)
        bm25_ids = [it.id for it in bm25_res.items]

        if self._has_vector and self._vector is not None:
            vec_res = self._vector.query(query, cand_size)
            vec_ids = [it.id for it in vec_res.items]
            fused = compute_rrf([bm25_ids, vec_ids], k=self._k)
            id_to_item: dict[str, dict] = {}
            for res in (bm25_res, vec_res):
                for it in res.items:
                    id_to_item.setdefault(it.id, {
                        "id": it.id, "text": it.text,
                        "source_episode_ids": it.source_episode_ids,
                        "node_type": it.node_type,
                    })
            fused_all = [
                RetrievedItem(
                    id=doc_id,
                    text=id_to_item[doc_id].get("text", ""),
                    score=score,
                    source_episode_ids=id_to_item[doc_id].get("source_episode_ids", []),
                    node_type=id_to_item[doc_id].get("node_type", "episode"),
                    stage_scores={"rrf": score},
                )
                for doc_id, score in fused
                if doc_id in id_to_item
            ]
            items = fused_all[:top_k]
        else:
            fused_all = bm25_res.items
            items = fused_all[:top_k]

        retrieval_trace: RetrievalTrace | None = None
        if trace:
            pool = list({
                **{it.id: it for it in bm25_res.items},
                **({it.id: it for it in (vec_res.items if self._has_vector else [])})
            })
            union_ids = [it.id for it in fused_all]
            final_ids = [it.id for it in items]
            retrieval_trace = RetrievalTrace(
                query=query,
                bm25_candidates=bm25_ids,
                vector_candidates=[it.id for it in (vec_res.items if self._has_vector else [])],
                candidate_pool=pool,
                union_candidates=union_ids,
                post_fusion_ranking=union_ids,
                pre_rerank=union_ids,
                post_rerank=final_ids,
                final_ranking=final_ids,
                candidate_k=candidate_k,
            )
        return RetrievalResult(
            query=query, items=items, top_k=top_k,
            latency_ms=(time.perf_counter() - t0) * 1000, trace=retrieval_trace,
        )

    def close(self) -> None:
        self._bm25.close()
        if self._vector:
            self._vector.close()


# ── Oracle Retriever (uses ground-truth labels) ───────────────────────────────

class OracleRetriever:
    """Returns ground-truth evidence with perfect rank. Measures reader ceiling."""
    name = "oracle"

    def __init__(self) -> None:
        self._id_to_item: dict[str, dict] = {}

    def build(self, corpus: list[dict], config: Any = None) -> IndexStats:
        t0 = time.perf_counter()
        self._id_to_item = {it["id"]: it for it in corpus}
        return IndexStats(
            method=self.name,
            node_count=len(corpus),
            build_wall_time_s=time.perf_counter() - t0,
        )

    def query(
        self,
        query: str,
        top_k: int = 10,
        trace: bool = False,
        candidate_k: int = 100,
        evidence_ids: list[str] | None = None,
    ) -> RetrievalResult:
        """Oracle requires evidence_ids to be passed from sample metadata."""
        t0 = time.perf_counter()
        items: list[RetrievedItem] = []
        for eid in (evidence_ids or []):
            it = self._id_to_item.get(eid)
            if it:
                items.append(_make_retrieved(it, 1.0))
        if len(items) < top_k:
            # Fill remaining with highest-timestamp items as context
            extra = [it for eid, it in self._id_to_item.items()
                     if eid not in (evidence_ids or [])]
            extra.sort(key=_item_ts, reverse=True)
            for it in extra[: top_k - len(items)]:
                items.append(_make_retrieved(it, 0.1))
        return RetrievalResult(
            query=query, items=items[:top_k], top_k=top_k,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def close(self) -> None:
        pass


BASELINE_REGISTRY: dict[str, type] = {
    "random": RandomRetriever,
    "recency": RecencyRetriever,
    "bm25": BM25Retriever,
    "vector": VectorRetriever,
    "hybrid_rrf": HybridRRFRetriever,
    "oracle": OracleRetriever,
}
