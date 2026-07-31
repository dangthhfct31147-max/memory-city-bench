"""Retrieval diagnostics computed from the extended RetrievalTrace.

These metrics do not change the definition of Recall@k. They explain *why* a
method wins or loses relative to a strong lexical baseline by measuring where
evidence enters and where it is dropped along the pipeline:

- candidate_recall@k: is the evidence anywhere in the fused candidate union?
- final_recall@k: is the evidence in the returned top-k?
- evidence_lost_by_reranking: evidence present pre-rerank but gone from top-k
  post-rerank (i.e. reranking pushed a correct item out).
- evidence_lost_by_temporal / _by_provenance: stage-attributed variants.
- duplicate_source_rate: fraction of returned slots wasted on duplicate source
  episodes (dedup should drive this to 0).
- raw_episode_ratio: fraction of returned items that are raw episodes (as
  opposed to README/hub/community nodes, which must not be final evidence).
- readme_ratio: fraction of returned items that are README/community/hub nodes.
- graph_only_gain: evidence found *only* because graph candidates added it.
- graph_induced_loss: evidence that BM25/vector had in-window but that dropped
  out of the final top-k after graph/rerank stages ran.
"""

from __future__ import annotations

from memcity.retrieval.protocol import RetrievalResult

_ROUTING_NODE_TYPES = {"readme", "community", "topic_hub", "begin", "end", "entity"}


def _recall(ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return sum(1 for r in ids[:k] if r in relevant) / len(relevant)


def _hit(ids: list[str], relevant: set[str], k: int) -> bool:
    return any(r in relevant for r in ids[:k])


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
    union = trace.union_candidates if trace else final_ids
    pre = trace.pre_rerank if trace else final_ids
    post = trace.post_rerank if trace else final_ids

    for k in candidate_ks:
        out[f"candidate_recall@{k}"] = _recall(union, relevant, k)
    for k in top_ks:
        out[f"final_recall@{k}"] = _recall(final_ids, relevant, k)

    max_final = max(top_ks)

    # Evidence present before rerank (within a comparable window) but not in the
    # final returned top-k → attributed to the reranking stage.
    if relevant:
        pre_window = set(pre[:max_final])
        post_window = set(final_ids[:max_final])
        lost = (pre_window & relevant) - post_window
        out["evidence_lost_by_reranking"] = len(lost) / len(relevant)

        # Stage attribution via trace snapshots (if present in metadata).
        meta = trace.metadata if trace else {}
        pre_temporal = set((meta.get("pre_temporal") or [])[:max_final])
        post_temporal = set((meta.get("post_temporal") or [])[:max_final])
        if pre_temporal or post_temporal:
            t_lost = (pre_temporal & relevant) - post_temporal
            out["evidence_lost_by_temporal"] = len(t_lost) / len(relevant)
        else:
            out["evidence_lost_by_temporal"] = 0.0

        pre_prov = set((meta.get("pre_provenance") or [])[:max_final])
        post_prov = set((meta.get("post_provenance") or [])[:max_final])
        if pre_prov or post_prov:
            p_lost = (pre_prov & relevant) - post_prov
            out["evidence_lost_by_provenance"] = len(p_lost) / len(relevant)
        else:
            out["evidence_lost_by_provenance"] = 0.0

        # graph_only_gain: evidence in final top-k that lexical+vector never had.
        lexvec = set(bm25) | set(vector)
        graph_set = set(graph)
        gained = {
            r for r in relevant
            if r in post_window and r not in lexvec and r in graph_set
        }
        out["graph_only_gain"] = len(gained) / len(relevant)

        # graph_induced_loss: evidence lexical/vector had in-window that fell out
        # of the final top-k (graph/rerank stages displaced it).
        lexvec_window = set(bm25[:max_final]) | set(vector[:max_final])
        displaced = (lexvec_window & relevant) - post_window
        out["graph_induced_loss"] = len(displaced) / len(relevant)
    else:
        out["evidence_lost_by_reranking"] = 0.0
        out["evidence_lost_by_temporal"] = 0.0
        out["evidence_lost_by_provenance"] = 0.0
        out["graph_only_gain"] = 0.0
        out["graph_induced_loss"] = 0.0

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
