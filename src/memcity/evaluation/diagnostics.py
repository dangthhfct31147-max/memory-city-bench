"""Retrieval diagnostics computed from the extended RetrievalTrace.

These metrics do not change the definition of Recall@k. They explain *why* a
method wins or loses relative to a strong lexical baseline by measuring where
evidence enters and where it is dropped along the pipeline.

Candidate recall metrics:
- candidate_pool_recall@k: is the evidence anywhere in the unordered candidate
  union (BM25 ∪ vector ∪ graph ∪ community), capped at k? This answers
  "did the pipeline ever see the evidence?"
- candidate_ranked_recall@k: is the evidence in the ranked fusion of all
  candidates, at depth k? (= old ``candidate_recall@k``; renamed for clarity).

Stage-attributed loss metrics (review issue #2/#3 fix):
- fusion_induced_loss: evidence that BM25/vector had in the candidate window but
  that fell out after fusion (weighted RRF).
- graph_induced_loss: evidence present post-fusion but absent post-graph.
  Correctly uses pre_graph → post_graph snapshots, not lexvec → final.
- temporal_induced_loss: evidence lost between post-graph and post-temporal.
- provenance_induced_loss: evidence lost between post-temporal and post-provenance.
- evidence_lost_by_reranking: catch-all: post-fusion → final (all stages combined).

graph_only_gain: evidence that graph candidates brought in and that survived
to the final top-k.
"""

from __future__ import annotations

from memcity.retrieval.protocol import RetrievalResult

_ROUTING_NODE_TYPES = {"readme", "community", "topic_hub", "begin", "end", "entity"}


def _recall(ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return sum(1 for r in ids[:k] if r in relevant) / len(relevant)


def _pool_recall(pool: list[str], relevant: set[str], k: int) -> float:
    """Recall for an unordered pool: any k elements that cover relevant."""
    if not relevant:
        return 1.0
    return min(1.0, sum(1 for r in pool if r in relevant) / len(relevant))


def _hit(ids: list[str], relevant: set[str], k: int) -> bool:
    return any(r in relevant for r in ids[:k])


def _stage_loss(
    before: list[str],
    after: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Evidence present in ``before[:k]`` but absent from ``after[:k]``."""
    if not relevant:
        return 0.0
    lost = set(before[:k]) & relevant - set(after[:k])
    return len(lost) / len(relevant)


def compute_diagnostics(
    result: RetrievalResult,
    relevant: set[str],
    scope_episode_ids: set[str],
    top_ks: tuple[int, ...],
    candidate_ks: tuple[int, ...],
) -> dict[str, float]:
    """Compute pipeline diagnostics for one query from its trace."""
    trace = result.trace
    out: dict[str, float] = {}

    final_ids = result.episode_ids()
    bm25 = trace.bm25_candidates if trace else []
    vector = trace.vector_candidates if trace else []
    graph = trace.graph_candidates if trace else []
    pool = trace.candidate_pool if trace else []
    # Ranked union after fusion (legacy field ``union_candidates``).
    union = trace.union_candidates if trace else final_ids

    # Stage snapshots (populated by MemoryCityRetriever; empty for baselines).
    post_fusion = trace.post_fusion_ranking if trace else union
    pre_graph_snap = trace.pre_graph if trace else post_fusion
    post_graph_snap = trace.post_graph if trace else post_fusion
    post_temporal_snap = trace.post_temporal if trace else post_graph_snap
    post_provenance_snap = trace.post_provenance if trace else post_temporal_snap

    max_final = max(top_ks)

    # ── Candidate recall at multiple depths ──────────────────────────────────
    for k in candidate_ks:
        # Ranked recall: evidence in the fused ranking up to depth k.
        out[f"candidate_recall@{k}"] = _recall(union, relevant, k)
        # Pool recall: evidence anywhere in the unordered candidate union.
        out[f"candidate_pool_recall@{k}"] = _pool_recall(pool or union, relevant, k)

    for k in top_ks:
        out[f"final_recall@{k}"] = _recall(final_ids, relevant, k)

    if relevant:
        post_window = set(final_ids[:max_final])

        # Catch-all: any evidence present post-fusion but gone from final top-k.
        out["evidence_lost_by_reranking"] = _stage_loss(
            post_fusion, list(post_window), relevant, max_final
        )

        # Per-stage attribution using correct snapshot pairs.
        out["fusion_induced_loss"] = _stage_loss(
            list(set(bm25[:max_final]) | set(vector[:max_final])),
            post_fusion,
            relevant, max_final,
        )
        out["graph_induced_loss"] = _stage_loss(
            pre_graph_snap, post_graph_snap, relevant, max_final
        )
        out["temporal_induced_loss"] = _stage_loss(
            post_graph_snap, post_temporal_snap, relevant, max_final
        )
        out["provenance_induced_loss"] = _stage_loss(
            post_temporal_snap, post_provenance_snap, relevant, max_final
        )

        # Legacy name kept for backward compat with existing reports.
        out["evidence_lost_by_temporal"] = out["temporal_induced_loss"]
        out["evidence_lost_by_provenance"] = out["provenance_induced_loss"]

        # graph_only_gain: evidence in final top-k that lexical+vector never had.
        lexvec = set(bm25) | set(vector)
        graph_set = set(graph)
        gained = {
            r for r in relevant
            if r in post_window and r not in lexvec and r in graph_set
        }
        out["graph_only_gain"] = len(gained) / len(relevant)
    else:
        # No ground-truth evidence for this query (abstention / category-5).
        # Zero out all attribution metrics so they do not inflate aggregates.
        for key in (
            "evidence_lost_by_reranking",
            "fusion_induced_loss",
            "graph_induced_loss",
            "temporal_induced_loss",
            "provenance_induced_loss",
            "evidence_lost_by_temporal",
            "evidence_lost_by_provenance",
            "graph_only_gain",
        ):
            out[key] = 0.0

    # Composition of the returned list.
    items = result.items[:max_final]
    n_items = len(items) or 1
    raw = sum(1 for it in items if it.node_type == "episode")
    routing = sum(1 for it in items if it.node_type in _ROUTING_NODE_TYPES)
    out["raw_episode_ratio"] = raw / n_items
    out["readme_ratio"] = routing / n_items

    # Duplicate source episodes among returned items.
    seen: set[str] = set()
    dup = 0
    total_slots = 0
    for it in items:
        for ep in (it.source_episode_ids or [it.id]):
            total_slots += 1
            if ep in seen:
                dup += 1
            else:
                seen.add(ep)
    out["duplicate_source_rate"] = dup / total_slots if total_slots else 0.0

    return out
