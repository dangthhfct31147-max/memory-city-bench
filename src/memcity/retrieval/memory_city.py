"""Memory City full retriever — integrates all components."""

from __future__ import annotations

import time
from typing import Any

import networkx as nx

from memcity.coordinator.heuristic import Coordinator, Route
from memcity.evaluation.metrics import compute_rrf
from memcity.graph.builder import MemoryCityGraphBuilder
from memcity.memory.schema import NodeType
from memcity.memory.store import Store
from memcity.retrieval.baselines import BM25Retriever, VectorRetriever, _item_text, _make_retrieved
from memcity.retrieval.protocol import IndexStats, RetrievalResult, RetrievalTrace, RetrievedItem
from memcity.utils.helpers import tokenize


class MemoryCityRetriever:
    """Full Memory City retrieval pipeline.

    Combines lexical, vector, graph expansion, temporal filtering,
    community routing, and provenance-aware reranking.
    All decisions are logged in the trace.
    """

    name = "memory_city_full"

    def __init__(
        self,
        rrf_k: int = 60,
        graph_hop_limit: int = 2,
        semantic_threshold: float = 0.75,
        num_topic_hubs: int = 8,
        temporal_decay_days: float = 365.0,
        contradiction_penalty: float = 0.2,
        provenance_boost: float = 0.1,
        coordinator_enabled: bool = True,
    ) -> None:
        self._rrf_k = rrf_k
        self._hop_limit = graph_hop_limit
        self._sem_thresh = semantic_threshold
        self._n_hubs = num_topic_hubs
        self._decay_days = temporal_decay_days
        self._contra_penalty = contradiction_penalty
        self._prov_boost = provenance_boost
        self._coord_enabled = coordinator_enabled

        self._store: Store | None = None
        self._graph: nx.DiGraph | None = None
        self._bm25: BM25Retriever | None = None
        self._vector: VectorRetriever | None = None
        self._corpus: list[dict] = []
        self._coordinator = Coordinator()
        self._has_vector = False

    def build(self, corpus: list[dict], config: Any = None) -> IndexStats:
        t0 = time.perf_counter()
        self._corpus = list(corpus)
        self._store = Store()  # in-memory store for graph data

        # Build BM25 index
        self._bm25 = BM25Retriever()
        self._bm25.build(corpus, config)

        # Build vector index (optional)
        try:
            self._vector = VectorRetriever()
            self._vector.build(corpus, config)
            self._has_vector = True
        except ImportError:
            self._has_vector = False

        # Build Memory City graph
        builder = MemoryCityGraphBuilder(
            store=self._store,
            semantic_threshold=self._sem_thresh,
            num_topic_hubs=self._n_hubs,
        )
        self._graph = builder.build(corpus)

        return IndexStats(
            method=self.name,
            node_count=self._graph.number_of_nodes(),
            edge_count=self._graph.number_of_edges(),
            community_count=sum(
                1 for _, d in self._graph.nodes(data=True)
                if d.get("node_type") == NodeType.COMMUNITY.value
            ),
            embedding_count=sum(
                1 for _, d in self._graph.nodes(data=True)
                if d.get("node_type") == NodeType.EPISODE.value
            ) if self._has_vector else 0,
            build_wall_time_s=time.perf_counter() - t0,
        )

    def query(self, query: str, top_k: int = 10, trace: bool = False) -> RetrievalResult:
        t0 = time.perf_counter()
        stage_latency: dict[str, float] = {}

        # ── Coordinator routing ───────────────────────────────────────────────
        if self._coord_enabled:
            t1 = time.perf_counter()
            decision = self._coordinator.route(query)
            stage_latency["coordinator"] = (time.perf_counter() - t1) * 1000
        else:
            from memcity.coordinator.heuristic import CoordinatorDecision
            decision = CoordinatorDecision(
                routes=[Route.HYBRID],
                features={},
                weights={"lexical": 0.5, "vector": 0.5},
            )

        candidates = top_k * 4

        # ── BM25 retrieval ────────────────────────────────────────────────────
        t1 = time.perf_counter()
        bm25_res = self._bm25.query(query, candidates) if self._bm25 else None
        stage_latency["bm25"] = (time.perf_counter() - t1) * 1000
        bm25_ids = [it.id for it in bm25_res.items] if bm25_res else []

        # ── Vector retrieval ──────────────────────────────────────────────────
        vec_ids: list[str] = []
        if self._has_vector and self._vector:
            t1 = time.perf_counter()
            vec_res = self._vector.query(query, candidates)
            stage_latency["vector"] = (time.perf_counter() - t1) * 1000
            vec_ids = [it.id for it in vec_res.items]

        # ── Community / README routing ────────────────────────────────────────
        community_ids: list[str] = []
        if Route.COMMUNITY_GLOBAL in decision.routes and self._graph:
            t1 = time.perf_counter()
            community_ids = self._community_search(query, candidates)
            stage_latency["community"] = (time.perf_counter() - t1) * 1000

        # ── Fuse rankings ─────────────────────────────────────────────────────
        t1 = time.perf_counter()
        ranking_lists = [l for l in [bm25_ids, vec_ids, community_ids] if l]
        if ranking_lists:
            fused = compute_rrf(ranking_lists, k=self._rrf_k)
            fused_ids = [doc_id for doc_id, _ in fused]
        else:
            fused_ids = bm25_ids or vec_ids
        stage_latency["fusion"] = (time.perf_counter() - t1) * 1000

        # ── Graph expansion ───────────────────────────────────────────────────
        if Route.LOCAL_GRAPH in decision.routes and self._graph:
            t1 = time.perf_counter()
            expanded = self._graph_expand(fused_ids[:top_k], max_extra=top_k)
            # Add expanded to end if not already present
            seen = set(fused_ids)
            for eid in expanded:
                if eid not in seen:
                    fused_ids.append(eid)
                    seen.add(eid)
            stage_latency["graph_expand"] = (time.perf_counter() - t1) * 1000

        # ── Temporal filtering ────────────────────────────────────────────────
        if Route.TEMPORAL in decision.routes:
            t1 = time.perf_counter()
            fused_ids = self._temporal_filter(fused_ids, query)
            stage_latency["temporal"] = (time.perf_counter() - t1) * 1000

        # ── Provenance reranking ──────────────────────────────────────────────
        t1 = time.perf_counter()
        fused_ids = self._provenance_rerank(fused_ids)
        stage_latency["provenance_rerank"] = (time.perf_counter() - t1) * 1000

        # ── Assemble results ──────────────────────────────────────────────────
        id_to_corpus: dict[str, dict] = {it["id"]: it for it in self._corpus}
        items: list[RetrievedItem] = []
        score_map = {doc_id: score for doc_id, score in (
            compute_rrf(ranking_lists, k=self._rrf_k) if ranking_lists else []
        )}
        for doc_id in fused_ids[:top_k]:
            it = id_to_corpus.get(doc_id)
            if it:
                score = score_map.get(doc_id, 0.0)
                items.append(_make_retrieved(it, score))

        total_latency = (time.perf_counter() - t0) * 1000

        retrieval_trace: RetrievalTrace | None = None
        if trace:
            retrieval_trace = RetrievalTrace(
                query=query,
                routes=[r.value for r in decision.routes],
                features=decision.features,
                weights=decision.weights,
                candidate_count_before=len(fused_ids),
                candidate_count_after=len(items),
                stage_latency_ms=stage_latency,
                total_latency_ms=total_latency,
            )

        return RetrievalResult(
            query=query, items=items, top_k=top_k,
            latency_ms=total_latency,
            trace=retrieval_trace,
        )

    def _community_search(self, query: str, top_k: int) -> list[str]:
        """Find episode IDs via community README matching."""
        if not self._graph:
            return []
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, str]] = []
        for nid, data in self._graph.nodes(data=True):
            if data.get("node_type") != NodeType.COMMUNITY.value:
                continue
            readme = data.get("metadata", {}).get("readme", {})
            terms = set(readme.get("top_terms", []) + readme.get("top_entities", []))
            overlap = len(query_tokens & terms) / (len(terms) + 1)
            if overlap > 0:
                member_eps = readme.get("source_episode_ids", [])
                for ep_id in member_eps[:top_k]:
                    scored.append((overlap, ep_id))
        scored.sort(reverse=True)
        seen: set[str] = set()
        result: list[str] = []
        for _, ep_id in scored:
            if ep_id not in seen:
                seen.add(ep_id)
                result.append(ep_id)
        return result[:top_k]

    def _graph_expand(self, seed_ids: list[str], max_extra: int) -> list[str]:
        """BFS expand from seed episode nodes up to hop_limit."""
        if not self._graph:
            return []
        visited: set[str] = set(seed_ids)
        frontier: list[str] = list(seed_ids)
        result: list[str] = []

        for _ in range(self._hop_limit):
            next_frontier: list[str] = []
            for nid in frontier:
                if not self._graph.has_node(nid):
                    continue
                for nbr in list(self._graph.successors(nid)) + list(self._graph.predecessors(nid)):
                    if nbr not in visited:
                        visited.add(nbr)
                        ntype = self._graph.nodes[nbr].get("node_type", "")
                        if ntype == NodeType.EPISODE.value:
                            result.append(nbr)
                        next_frontier.append(nbr)
                        if len(result) >= max_extra:
                            return result
            frontier = next_frontier
        return result[:max_extra]

    def _temporal_filter(self, ids: list[str], query: str) -> list[str]:
        """Sort episodes with temporal keywords: prefer newer unless 'originally/before' query."""
        if not self._graph:
            return ids
        q_lower = query.lower()
        prefer_old = any(kw in q_lower for kw in ("originally", "before", "used to", "was", "old"))

        id_to_ts: dict[str, float] = {}
        for nid in ids:
            if self._graph.has_node(nid):
                id_to_ts[nid] = self._graph.nodes[nid].get("timestamp", 0.0)
            else:
                id_to_ts[nid] = 0.0

        return sorted(ids, key=lambda nid: id_to_ts.get(nid, 0.0), reverse=not prefer_old)

    def _provenance_rerank(self, ids: list[str]) -> list[str]:
        """Boost items that have richer provenance (more source_episode_ids)."""
        if not self._graph:
            return ids
        scored: list[tuple[float, str]] = []
        for rank, nid in enumerate(ids):
            base_score = 1.0 / (1 + rank)
            if self._graph.has_node(nid):
                ep_count = len(self._graph.nodes[nid].get("source_episode_ids", []))
                boost = ep_count * self._prov_boost
            else:
                boost = 0.0
            scored.append((base_score + boost, nid))
        scored.sort(reverse=True)
        return [nid for _, nid in scored]

    def close(self) -> None:
        if self._store:
            self._store.close()
        if self._bm25:
            self._bm25.close()
        if self._vector:
            self._vector.close()
