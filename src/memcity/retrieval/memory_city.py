"""Memory City full retriever — integrates all components.

Design constraints (from the diagnostic review):

- Scores are additive. Temporal and provenance signals *adjust* a candidate's
  score; they never hard-reorder the whole list by timestamp. A strong lexical
  or vector hit can only be displaced by an equally strong combined signal.
- Graph expansion only *supplements* the candidate pool. Expanded neighbours are
  added with a discounted score and can never evict a strong BM25/vector hit.
- When the coordinator is not confident, the retriever falls back to plain
  Hybrid RRF (BM25 + vector) and skips graph/temporal/community/provenance.
- Only raw episodes reach the final result. README / topic-hub / community nodes
  are used purely to *route* toward raw episodes.
- Every stage is snapshotted in the trace so diagnostics can attribute where
  evidence enters or is dropped.
"""

from __future__ import annotations

import time
from typing import Any

import networkx as nx

from memcity.coordinator.heuristic import Coordinator, CoordinatorDecision, Route
from memcity.graph.builder import MemoryCityGraphBuilder
from memcity.memory.schema import NodeType
from memcity.memory.store import Store
from memcity.retrieval.baselines import BM25Retriever, VectorRetriever, _make_retrieved
from memcity.retrieval.protocol import IndexStats, RetrievalResult, RetrievalTrace, RetrievedItem
from memcity.utils.helpers import tokenize

# Score below which a coordinator decision is treated as unreliable and the
# retriever falls back to plain Hybrid RRF.
_CONFIDENCE_FALLBACK = 0.45
# Discount applied to graph-expanded candidates so they supplement but never
# outrank a genuine lexical/vector hit.
_GRAPH_SUPPLEMENT_SCALE = 0.25
# Weight of the additive temporal nudge relative to the fused retrieval score.
_TEMPORAL_NUDGE = 0.15


class MemoryCityRetriever:
    """Full Memory City retrieval pipeline with per-component ablation flags.

    The ``enable_*`` flags let the benchmark run true ablations against the same
    index: turning a component off removes only that component's contribution,
    leaving everything else identical.
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
        *,
        enable_vector: bool = True,
        enable_graph_expansion: bool = True,
        enable_temporal: bool = True,
        enable_community: bool = True,
        enable_provenance: bool = True,
        enable_be: bool = True,
        name: str | None = None,
    ) -> None:
        self._rrf_k = rrf_k
        self._hop_limit = graph_hop_limit
        self._sem_thresh = semantic_threshold
        self._n_hubs = num_topic_hubs
        self._decay_days = temporal_decay_days
        self._contra_penalty = contradiction_penalty
        self._prov_boost = provenance_boost
        self._coord_enabled = coordinator_enabled

        self._enable_vector = enable_vector
        self._enable_graph_expansion = enable_graph_expansion
        self._enable_temporal = enable_temporal
        self._enable_community = enable_community
        self._enable_provenance = enable_provenance
        self._enable_be = enable_be
        if name:
            self.name = name

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
        self._store = Store()

        self._bm25 = BM25Retriever()
        self._bm25.build(corpus, config)

        self._has_vector = False
        if self._enable_vector:
            try:
                self._vector = VectorRetriever()
                self._vector.build(corpus, config)
                self._has_vector = True
            except ImportError:
                self._has_vector = False

        builder = MemoryCityGraphBuilder(
            store=self._store,
            semantic_threshold=self._sem_thresh,
            num_topic_hubs=self._n_hubs,
            enable_be=self._enable_be,
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

    def query(
        self,
        query: str,
        top_k: int = 10,
        trace: bool = False,
        candidate_k: int = 100,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        stage_latency: dict[str, float] = {}
        # Use caller-supplied candidate_k so every retriever is evaluated at the
        # same depth (review issue #1: candidate recall fairness).
        candidates = max(top_k, candidate_k)

        # ── Coordinator routing ───────────────────────────────────────────────
        if self._coord_enabled:
            t1 = time.perf_counter()
            decision = self._coordinator.route(query)
            stage_latency["coordinator"] = (time.perf_counter() - t1) * 1000
        else:
            decision = CoordinatorDecision(
                routes=[Route.HYBRID], features={}, weights={"lexical": 0.5, "vector": 0.5},
                confidence=1.0,
            )

        low_confidence = decision.confidence < _CONFIDENCE_FALLBACK

        # _route_active: a component should fire when its enable flag is on AND
        # either (a) the coordinator is off (pure ablation, fire unconditionally)
        # or (b) the coordinator is on, confidence is adequate, and the correct
        # route was selected.  This decouples ablation flags from coordinator
        # routing so each variant tests exactly what its name says.
        def _route_active(flag: bool, route: Route) -> bool:
            if not flag:
                return False
            if not self._coord_enabled:
                return True
            return not low_confidence and route in decision.routes

        # ── Component retrieval ───────────────────────────────────────────────
        t1 = time.perf_counter()
        bm25_res = self._bm25.query(query, candidates) if self._bm25 else None
        stage_latency["bm25"] = (time.perf_counter() - t1) * 1000
        bm25_ids = [it.id for it in bm25_res.items] if bm25_res else []

        vec_ids: list[str] = []
        if self._has_vector and self._vector:
            t1 = time.perf_counter()
            vec_res = self._vector.query(query, candidates)
            stage_latency["vector"] = (time.perf_counter() - t1) * 1000
            vec_ids = [it.id for it in vec_res.items]

        community_ids: list[str] = []
        if _route_active(self._enable_community, Route.COMMUNITY_GLOBAL) and self._graph:
            t1 = time.perf_counter()
            community_ids = self._community_search(query, candidates)
            stage_latency["community"] = (time.perf_counter() - t1) * 1000

        # ── Weighted fusion of lexical + vector (+ community routing) ──────────
        weights = decision.weights if not low_confidence else {"lexical": 0.5, "vector": 0.5}
        w_lex = weights.get("lexical", 0.5)
        w_vec = weights.get("vector", 0.5)
        w_com = weights.get("community", 0.5) if community_ids else 0.0
        t1 = time.perf_counter()
        base_scores = self._weighted_rrf(
            [(bm25_ids, w_lex), (vec_ids, w_vec), (community_ids, w_com)]
        )
        stage_latency["fusion"] = (time.perf_counter() - t1) * 1000

        # Snapshot after fusion, before any graph stage (pre_graph = post_fusion).
        post_fusion_ranking = self._rank_episode_ids(base_scores)
        pre_graph = post_fusion_ranking

        scores = dict(base_scores)

        # ── Graph expansion (supplement only) ─────────────────────────────────
        graph_ids: list[str] = []
        if _route_active(self._enable_graph_expansion, Route.LOCAL_GRAPH) and self._graph:
            t1 = time.perf_counter()
            seeds = [d for d, _ in sorted(base_scores.items(), key=lambda x: x[1], reverse=True)][:top_k]
            graph_ids = self._graph_expand(seeds, max_extra=candidates)
            min_base = min(base_scores.values(), default=0.0)
            for rank, eid in enumerate(graph_ids):
                if eid in scores:
                    continue
                # Supplement below the weakest genuine hit; never evicts one.
                scores[eid] = (min_base * _GRAPH_SUPPLEMENT_SCALE) / (1 + rank)
            stage_latency["graph_expand"] = (time.perf_counter() - t1) * 1000

        # Snapshot after graph expansion (post_graph = pre_temporal).
        post_graph = self._rank_episode_ids(scores)
        pre_temporal = post_graph

        # ── Temporal nudge (additive, never a hard re-sort) ───────────────────
        if _route_active(self._enable_temporal, Route.TEMPORAL):
            t1 = time.perf_counter()
            scores = self._temporal_nudge(scores, query)
            stage_latency["temporal"] = (time.perf_counter() - t1) * 1000
        post_temporal = self._rank_episode_ids(scores)

        # ── Provenance nudge (additive, connectivity-weighted) ────────────────
        # Gated only by enable flag and confidence (no dedicated route).
        pre_provenance = post_temporal
        if self._enable_provenance and not low_confidence:
            t1 = time.perf_counter()
            scores = self._provenance_nudge(scores)
            stage_latency["provenance_rerank"] = (time.perf_counter() - t1) * 1000
        post_provenance = self._rank_episode_ids(scores)

        # ── Assemble raw-episode results, deduped by source episode ───────────
        id_to_corpus: dict[str, dict] = {it["id"]: it for it in self._corpus}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        items: list[RetrievedItem] = []
        seen_sources: set[str] = set()
        for doc_id, score in ranked:
            corpus_item = id_to_corpus.get(doc_id)
            if corpus_item is None:
                continue  # README/hub/community nodes never become final evidence
            src = corpus_item.get("source_episode_ids") or [doc_id]
            canonical = src[0]
            if canonical in seen_sources:
                continue
            seen_sources.add(canonical)
            items.append(_make_retrieved(corpus_item, score))
            if len(items) >= top_k:
                break

        total_latency = (time.perf_counter() - t0) * 1000

        retrieval_trace: RetrievalTrace | None = None
        if trace:
            # Ranked union of all raw-episode candidates (candidate_ranked_recall).
            union = self._rank_episode_ids(scores)
            # Unordered pool for candidate_pool_recall.
            pool_set = set(bm25_ids) | set(vec_ids) | set(graph_ids) | set(community_ids)
            pool_list = list(pool_set)
            final_ids = [it.id for it in items]
            retrieval_trace = RetrievalTrace(
                query=query,
                routes=[r.value for r in decision.routes],
                features=decision.features,
                weights=weights,
                candidate_count_before=len(scores),
                candidate_count_after=len(items),
                stage_latency_ms=stage_latency,
                total_latency_ms=total_latency,
                bm25_candidates=bm25_ids,
                vector_candidates=vec_ids,
                graph_candidates=graph_ids,
                community_candidates=community_ids,
                candidate_pool=pool_list,
                union_candidates=union,
                post_fusion_ranking=post_fusion_ranking,
                pre_graph=pre_graph,
                post_graph=post_graph,
                post_temporal=post_temporal,
                post_provenance=post_provenance,
                pre_rerank=post_fusion_ranking,
                post_rerank=final_ids,
                final_ranking=final_ids,
                candidate_k=candidate_k,
                metadata={
                    "confidence": decision.confidence,
                    "low_confidence_fallback": low_confidence,
                    "pre_temporal": pre_temporal,
                    "post_temporal": post_temporal,
                    "pre_provenance": pre_provenance,
                    "post_provenance": post_provenance,
                },
            )

        return RetrievalResult(
            query=query, items=items, top_k=top_k,
            latency_ms=total_latency, trace=retrieval_trace,
        )

    def _weighted_rrf(self, lists: list[tuple[list[str], float]]) -> dict[str, float]:
        """Weighted Reciprocal Rank Fusion.

        Each ranked list contributes ``weight / (rrf_k + rank)`` per document.
        This is where coordinator weights actually influence the score, rather
        than being logged and ignored.
        """
        scores: dict[str, float] = {}
        for ranked, weight in lists:
            if not ranked or weight <= 0:
                continue
            for rank, doc_id in enumerate(ranked, 1):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (self._rrf_k + rank)
        return scores

    def _rank_episode_ids(self, scores: dict[str, float]) -> list[str]:
        """Return corpus (raw-episode) ids ranked by score, dropping graph-only nodes."""
        corpus_ids = {it["id"] for it in self._corpus}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked if doc_id in corpus_ids]

    def _community_search(self, query: str, top_k: int) -> list[str]:
        """Route to raw episode ids via community README term overlap."""
        if not self._graph:
            return []
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, str]] = []
        for _, data in self._graph.nodes(data=True):
            if data.get("node_type") != NodeType.COMMUNITY.value:
                continue
            readme = data.get("metadata", {}).get("readme", {})
            terms = set(readme.get("top_terms", []) + readme.get("top_entities", []))
            overlap = len(query_tokens & terms) / (len(terms) + 1)
            if overlap > 0:
                for ep_id in readme.get("source_episode_ids", [])[:top_k]:
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
        """BFS from seed episode nodes up to hop_limit, returning episode ids only."""
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
                neighbours = list(self._graph.successors(nid)) + list(
                    self._graph.predecessors(nid)
                )
                for nbr in neighbours:
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    if self._graph.nodes[nbr].get("node_type") == NodeType.EPISODE.value:
                        result.append(nbr)
                    next_frontier.append(nbr)
                    if len(result) >= max_extra:
                        return result
            frontier = next_frontier
        return result[:max_extra]

    def _temporal_nudge(self, scores: dict[str, float], query: str) -> dict[str, float]:
        """Add a small recency (or antiquity) bonus without reordering by time.

        The bonus is bounded by ``_TEMPORAL_NUDGE`` times the current max score, so
        it can break ties and gently reorder near-equal candidates but cannot
        overturn a decisively stronger lexical/vector hit.
        """
        if not self._graph or not scores:
            return scores
        q_lower = query.lower()
        prefer_old = any(
            kw in q_lower for kw in ("originally", "before", "used to", "first", "old")
        )
        timestamps: dict[str, float] = {}
        for nid in scores:
            if self._graph.has_node(nid):
                timestamps[nid] = self._graph.nodes[nid].get("timestamp", 0.0)
        if not timestamps:
            return scores
        lo, hi = min(timestamps.values()), max(timestamps.values())
        span = (hi - lo) or 1.0
        max_score = max(scores.values()) or 1.0
        nudged = dict(scores)
        for nid, ts in timestamps.items():
            frac = (ts - lo) / span  # 0 oldest … 1 newest
            if prefer_old:
                frac = 1.0 - frac
            nudged[nid] += _TEMPORAL_NUDGE * max_score * frac
        return nudged

    def _provenance_nudge(self, scores: dict[str, float]) -> dict[str, float]:
        """Boost episodes supported by multiple independent graph paths.

        The nudge is proportional to the number of distinct in-neighbour episode
        nodes in the graph — i.e. how many other episodes corroborate this one
        through entity/semantic/community edges. This is a better proxy for
        provenance strength than the raw source_episode_ids count (which is always
        1 for a plain raw episode).

        The nudge is bounded to ``_TEMPORAL_NUDGE * max_score`` so it can break
        ties and reorder near-equal candidates without overturning a decisive
        lexical/vector hit (same additive contract as the temporal nudge).
        """
        if not self._graph or not scores:
            return scores
        max_score = max(scores.values()) or 1.0
        nudged = dict(scores)
        for nid in scores:
            if not self._graph.has_node(nid):
                continue
            # Count distinct episode predecessors (independent corroborating paths)
            episode_predecessors = sum(
                1 for pred in self._graph.predecessors(nid)
                if self._graph.nodes[pred].get("node_type") == "episode"
            )
            if episode_predecessors > 0:
                # Bounded boost: at most _TEMPORAL_NUDGE * max_score regardless of path count
                boost = self._prov_boost * min(episode_predecessors, 5) * max_score
                nudged[nid] += boost
        return nudged

    def close(self) -> None:
        if self._store:
            self._store.close()
        if self._bm25:
            self._bm25.close()
        if self._vector:
            self._vector.close()
