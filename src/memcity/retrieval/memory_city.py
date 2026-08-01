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

import math
import time
from typing import Any

import networkx as nx

from memcity.coordinator.heuristic import (
    Coordinator,
    CoordinatorDecision,
    Route,
    TemporalConstraint,
    parse_temporal_constraint,
)
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

# Per-edge-type base weights for weighted / PPR graph propagation. These answer
# "which path is most relevant to the current query", replacing the flat BFS that
# treats every neighbour the same.
#
#   MENTIONS               — episode ↔ entity; the strongest "same subject" signal.
#   NEXT / BEGINS / ENDS_WITH — procedural / temporal adjacency.
#   SEMANTICALLY_RELATED   — carries a cosine weight already; multiplied in.
#   SAME_SESSION           — weak co-occurrence.
#   BELONGS_TO_TOPIC       — soft topical grouping.
#   BELONGS_TO_COMMUNITY   — routing only; excluded from propagation (weight 0).
_DEFAULT_EDGE_TYPE_WEIGHTS: dict[str, float] = {
    "MENTIONS": 1.0,
    "NEXT": 0.6,
    "BEGINS": 0.5,
    "ENDS_WITH": 0.5,
    "SEMANTICALLY_RELATED": 1.0,
    "SAME_SESSION": 0.2,
    "BELONGS_TO_TOPIC": 0.3,
    "BELONGS_TO_COMMUNITY": 0.0,
}
# Route-conditioned multipliers: a temporal/procedural query boosts sequential
# edges; a local-graph query boosts entity mentions.
_ROUTE_EDGE_BOOST: dict[Route, dict[str, float]] = {
    Route.TEMPORAL: {"NEXT": 1.5, "BEGINS": 1.5, "ENDS_WITH": 1.5},
    Route.LOCAL_GRAPH: {"MENTIONS": 1.3, "SEMANTICALLY_RELATED": 1.2},
}
# PageRank damping; networkx default, restated for reproducibility.
_PPR_ALPHA = 0.85


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
        graph_mode: str = "bfs",
        degree_penalty: bool = False,
        max_graph_contribution_per_episode: float = 1.0,
        edge_type_weights: dict[str, float] | None = None,
        enable_vector: bool = True,
        enable_graph_expansion: bool = True,
        enable_temporal: bool = True,
        enable_temporal_kg: bool = False,
        enable_community: bool = True,
        enable_provenance: bool = True,
        enable_be: bool = True,
        enable_hierarchical: bool = False,
        summary_max_levels: int = 1,
        summary_max_sentences: int = 3,
        summary_branching_factor: int = 5,
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

        # Graph propagation controls (Phase 2). ``graph_mode`` selects how the
        # graph stage turns seeds into supplementary episodes:
        #   "bfs"      — legacy undirected BFS (flat, ignores edge weight);
        #   "weighted" — edge-type + confidence weighted spread with hop decay;
        #   "ppr"      — Personalized PageRank seeded on the fused top hits.
        self._graph_mode = graph_mode
        self._degree_penalty = degree_penalty
        self._max_graph_contribution = max_graph_contribution_per_episode
        self._edge_type_weights = dict(_DEFAULT_EDGE_TYPE_WEIGHTS)
        if edge_type_weights:
            self._edge_type_weights.update(edge_type_weights)

        self._enable_vector = enable_vector
        self._enable_graph_expansion = enable_graph_expansion
        self._enable_temporal = enable_temporal
        # Phase 3: bi-temporal fact layer. When on, the builder emits FactNodes
        # with valid_from/valid_to and SUPERSEDES/CONTRADICTS edges, and temporal
        # queries are resolved against event time rather than a recency nudge.
        self._enable_temporal_kg = enable_temporal_kg
        self._enable_community = enable_community
        self._enable_provenance = enable_provenance
        self._enable_be = enable_be
        # Phase 6: hierarchical summary tier. When on, the builder emits an
        # extractive SUMMARY tree over the episodes and global/overview queries are
        # routed through it (summary → children → raw episodes) before fusion.
        self._enable_hierarchical = enable_hierarchical
        self._summary_max_levels = summary_max_levels
        self._summary_max_sentences = summary_max_sentences
        self._summary_branching = summary_branching_factor
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
            enable_facts=self._enable_temporal_kg,
            enable_summaries=self._enable_hierarchical,
            summary_max_levels=self._summary_max_levels,
            summary_max_sentences=self._summary_max_sentences,
            summary_branching_factor=self._summary_branching,
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

        # ── Hierarchical tree routing (Phase 6) ────────────────────────────────
        # A global/overview query descends the summary tree to the raw episodes
        # beneath the best-matching summaries. Same route as community (global),
        # but a distinct pool so the ablation delta isolates the tree's effect.
        tree_ids: list[str] = []
        if _route_active(self._enable_hierarchical, Route.COMMUNITY_GLOBAL) and self._graph:
            t1 = time.perf_counter()
            tree_ids = self._tree_search(query, candidates)
            stage_latency["tree"] = (time.perf_counter() - t1) * 1000

        # ── Weighted fusion of lexical + vector (+ community routing) ──────────
        weights = decision.weights if not low_confidence else {"lexical": 0.5, "vector": 0.5}
        w_lex = weights.get("lexical", 0.5)
        w_vec = weights.get("vector", 0.5)
        w_com = weights.get("community", 0.5) if community_ids else 0.0
        # Tree routing reuses the community (global) weight: both answer overview
        # queries and neither should overturn a decisive lexical/vector hit.
        w_tree = weights.get("community", 0.5) if tree_ids else 0.0
        t1 = time.perf_counter()
        base_scores = self._weighted_rrf(
            [(bm25_ids, w_lex), (vec_ids, w_vec), (community_ids, w_com),
             (tree_ids, w_tree)]
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
            ranked_seeds = [
                d for d, _ in sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            seeds = ranked_seeds[:top_k]
            # Dispatch on graph_mode. BFS returns a flat neighbour ordering; the
            # weighted / PPR modes return (episode_id, propagation_score) pairs so
            # the supplement respects "which path fits the query".
            if self._graph_mode == "bfs":
                graph_ids = self._graph_expand(seeds, max_extra=candidates)
                min_base = min(base_scores.values(), default=0.0)
                for rank, eid in enumerate(graph_ids):
                    if eid in scores:
                        continue
                    # Supplement below the weakest genuine hit; never evicts one.
                    scores[eid] = (min_base * _GRAPH_SUPPLEMENT_SCALE) / (1 + rank)
            else:
                graph_scored = self._graph_propagate(
                    base_scores, decision.routes, max_extra=candidates,
                )
                graph_ids = [eid for eid, _ in graph_scored]
                min_base = min(base_scores.values(), default=0.0)
                max_prop = max((s for _, s in graph_scored), default=0.0) or 1.0
                for eid, prop in graph_scored:
                    if eid in scores:
                        continue
                    # Normalise propagation mass to [0, 1], cap per-episode, then
                    # scale below the weakest genuine hit (same supplement contract
                    # as BFS: a graph path never evicts a real lexical/vector hit).
                    norm = min(prop / max_prop, self._max_graph_contribution)
                    scores[eid] = min_base * _GRAPH_SUPPLEMENT_SCALE * norm
            stage_latency["graph_expand"] = (time.perf_counter() - t1) * 1000

        # Snapshot after graph expansion (post_graph = pre_temporal).
        post_graph = self._rank_episode_ids(scores)
        pre_temporal = post_graph

        # ── Temporal stage ────────────────────────────────────────────────────
        # Two modes: the legacy additive recency nudge, or (Phase 3) a bi-temporal
        # fact resolution that answers the query against event time. The KG mode
        # takes precedence when enabled and the query carries a temporal cue.
        constraint = parse_temporal_constraint(query)
        if self._enable_temporal_kg and constraint.is_active and self._graph:
            t1 = time.perf_counter()
            scores = self._temporal_kg_resolve(scores, query, constraint)
            stage_latency["temporal_kg"] = (time.perf_counter() - t1) * 1000
        elif _route_active(self._enable_temporal, Route.TEMPORAL):
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
            pool_set = (
                set(bm25_ids) | set(vec_ids) | set(graph_ids)
                | set(community_ids) | set(tree_ids)
            )
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
                tree_candidates=tree_ids,
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

    def _tree_search(self, query: str, top_k: int) -> list[str]:
        """Route a global query down the summary tree to raw episodes.

        RAPTOR-style traversal: score every SUMMARY node by query-term overlap,
        pick the best-matching summaries, then surface the raw episodes beneath
        them (their flattened ``source_episode_ids``). Higher-level summaries with
        broader coverage are preferred as entry points; ties fall back to raw
        overlap. Only raw episodes are returned — the summary text itself never
        becomes final evidence.
        """
        if not self._graph:
            return []
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        scored: list[tuple[float, int, str]] = []
        for nid, data in self._graph.nodes(data=True):
            if data.get("node_type") != NodeType.SUMMARY.value:
                continue
            summary_tokens = set(tokenize(data.get("text", "")))
            if not summary_tokens:
                continue
            overlap = len(query_tokens & summary_tokens) / (len(summary_tokens) + 1)
            if overlap <= 0:
                continue
            level = int(data.get("metadata", {}).get("level", 1))
            scored.append((overlap, level, nid))
        # Prefer stronger overlap, then higher (broader) summaries as entry points.
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        seen: set[str] = set()
        result: list[str] = []
        for _overlap, _level, nid in scored:
            for ep_id in self._graph.nodes[nid].get("source_episode_ids", []):
                if ep_id not in seen:
                    seen.add(ep_id)
                    result.append(ep_id)
                    if len(result) >= top_k:
                        return result
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

    def _edge_weight(self, data: dict, routes: list[Route]) -> float:
        """Effective propagation weight for one edge under the current routes.

        base = edge-type weight × edge confidence (``Edge.weight``), then boosted
        by any route-conditioned multiplier (temporal query lifts NEXT/BEGINS,
        local-graph query lifts MENTIONS). Community edges resolve to 0 and are
        dropped from propagation (they only route, never carry relevance).
        """
        etype = data.get("edge_type", "")
        base = self._edge_type_weights.get(etype, 0.0)
        if base <= 0.0:
            return 0.0
        confidence = float(data.get("weight", 1.0)) or 1.0
        boost = 1.0
        for route in routes:
            mult = _ROUTE_EDGE_BOOST.get(route, {}).get(etype)
            if mult:
                boost = max(boost, mult)
        return base * confidence * boost

    def _propagation_subgraph(self, routes: list[Route]) -> nx.DiGraph:
        """Build a weighted, symmetric propagation graph honouring edge types.

        Relevance flows both ways along an edge (an entity mention connects its
        episodes in either direction), so we add both orientations with the same
        effective weight. Zero-weight (community/routing) edges are omitted.
        """
        assert self._graph is not None
        sub = nx.DiGraph()
        sub.add_nodes_from(self._graph.nodes())
        for src, dst, data in self._graph.edges(data=True):
            w = self._edge_weight(data, routes)
            if w <= 0.0:
                continue
            for a, b in ((src, dst), (dst, src)):
                if sub.has_edge(a, b):
                    sub[a][b]["weight"] += w
                else:
                    sub.add_edge(a, b, weight=w)
        return sub

    def _graph_propagate(
        self,
        base_scores: dict[str, float],
        routes: list[Route],
        max_extra: int,
    ) -> list[tuple[str, float]]:
        """Weighted / Personalized-PageRank propagation from the fused seeds.

        Returns ``(episode_id, score)`` for episodes NOT already in ``base_scores``,
        ranked by propagated relevance. ``graph_mode`` picks the algorithm:

          * ``"weighted"`` — one-/multi-hop spread: each seed pushes its fused
            score along outgoing edges with ``edge_weight`` and geometric hop
            decay; accumulates on neighbours.
          * ``"ppr"`` — Personalized PageRank with the personalization vector set
            to the seed fused scores (HippoRAG-style relevance propagation).

        A degree penalty (optional) divides a node's mass by ``log1p(degree)`` so a
        super-hub with hundreds of neighbours cannot hoard score from many paths.
        """
        if not self._graph or not base_scores:
            return []
        corpus_ids = {it["id"] for it in self._corpus}
        seeds = {d: s for d, s in base_scores.items() if self._graph.has_node(d)}
        if not seeds:
            return []
        sub = self._propagation_subgraph(routes)

        propagated: dict[str, float] = {}
        if self._graph_mode == "ppr":
            total = sum(seeds.values()) or 1.0
            personalization = {n: 0.0 for n in sub.nodes()}
            for d, s in seeds.items():
                personalization[d] = s / total
            try:
                propagated = nx.pagerank(
                    sub, alpha=_PPR_ALPHA, personalization=personalization,
                    weight="weight",
                )
            except (nx.PowerIterationFailedConvergence, ZeroDivisionError):
                propagated = dict(personalization)
        else:  # "weighted"
            frontier = dict(seeds)
            decay = 1.0
            for _ in range(self._hop_limit):
                decay *= 0.5
                next_frontier: dict[str, float] = {}
                for nid, mass in frontier.items():
                    if not sub.has_node(nid) or mass <= 0.0:
                        continue
                    out = list(sub.successors(nid))
                    w_sum = sum(sub[nid][nbr]["weight"] for nbr in out) or 1.0
                    for nbr in out:
                        share = mass * (sub[nid][nbr]["weight"] / w_sum) * decay
                        propagated[nbr] = propagated.get(nbr, 0.0) + share
                        next_frontier[nbr] = next_frontier.get(nbr, 0.0) + share
                frontier = next_frontier

        if self._degree_penalty:
            for nid in list(propagated):
                deg = self._graph.degree(nid) if self._graph.has_node(nid) else 0
                propagated[nid] /= math.log1p(deg) or 1.0

        # Keep only raw episodes that are genuinely new (not already scored seeds).
        out = [
            (nid, score)
            for nid, score in propagated.items()
            if nid in corpus_ids and nid not in base_scores and score > 0.0
        ]
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:max_extra]

    def _temporal_kg_resolve(
        self,
        scores: dict[str, float],
        query: str,
        constraint: TemporalConstraint,
    ) -> dict[str, float]:
        """Resolve a temporal query against fact nodes' event-time validity.

        For each fact matching the query subject we decide whether it satisfies
        the constraint (``current`` → ``valid_to is None``; ``original`` → the
        earliest fact; ``before`` → a superseded fact; ``after``/default →
        later facts) and boost or penalise the *source episode* accordingly. The
        boost is additive and bounded (same contract as the recency nudge): it
        reorders near-equal candidates and surfaces the correct-era evidence
        without letting a fact match overturn a decisive lexical hit.
        """
        if not self._graph or not scores:
            return scores
        max_score = max(scores.values()) or 1.0
        q_tokens = set(tokenize(query))
        nudged = dict(scores)

        # Group fact nodes by subject so "current vs original" is decided per key.
        facts_by_subject: dict[str, list[dict]] = {}
        for _, data in self._graph.nodes(data=True):
            if data.get("node_type") != NodeType.FACT.value:
                continue
            meta = data.get("metadata", {})
            subject = str(meta.get("subject", ""))
            # Only consider facts whose subject overlaps the query tokens, so an
            # unrelated fact timeline never perturbs the ranking.
            subj_tokens = set(tokenize(subject))
            if not subj_tokens or not (subj_tokens & q_tokens):
                continue
            facts_by_subject.setdefault(subject, []).append(data)

        for _subject, facts in facts_by_subject.items():
            facts.sort(key=lambda d: d.get("valid_from") or d.get("timestamp", 0.0))
            for idx, data in enumerate(facts):
                meta = data.get("metadata", {})
                is_current = data.get("valid_to") is None
                is_original = idx == 0
                if constraint.kind == "current":
                    satisfies = is_current
                elif constraint.kind in ("original", "before"):
                    satisfies = not is_current  # a superseded (older) fact
                    if constraint.kind == "original":
                        satisfies = is_original
                elif constraint.kind == "after":
                    satisfies = not is_original
                else:
                    satisfies = True
                sign = 1.0 if satisfies else -1.0
                for eid in data.get("source_episode_ids", []):
                    if eid in nudged:
                        nudged[eid] += sign * _TEMPORAL_NUDGE * max_score
        return nudged

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
