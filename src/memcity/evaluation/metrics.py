"""Retrieval quality metrics: Recall@k, Precision@k, MRR, nDCG, Evidence metrics."""

from __future__ import annotations

import math
from typing import Sequence


def _at_k(retrieved: list[str], relevant: set[str], k: int) -> tuple[list[str], set[str]]:
    return retrieved[:k], relevant


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def any_hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1 if at least one relevant item is in top-k, 0 otherwise."""
    if not relevant:
        return 1.0
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def all_evidence_recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of required evidence found in top-k (0.0–1.0)."""
    if not relevant:
        return 1.0
    found = sum(1 for r in retrieved[:k] if r in relevant)
    return found / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary relevance nDCG@k."""
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(retrieved[:k])
        if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mean_rank_first_relevant(retrieved: list[str], relevant: set[str]) -> float | None:
    """Rank of first relevant item (1-indexed), or None if not found."""
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return float(i)
    return None


def evidence_f1(retrieved: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def rrf_score(ranks: list[int], k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a document given its ranks in multiple lists."""
    return sum(1.0 / (k + rank) for rank in ranks)


def compute_rrf(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: Each element is a ranked list of document IDs.
        k: RRF constant.

    Returns:
        Sorted list of (doc_id, score) tuples (descending).
    """
    scores: dict[str, float] = {}
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def compute_all_metrics(
    retrieved: list[str],
    relevant: set[str],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute a full suite of retrieval metrics."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        out[f"any_hit@{k}"] = any_hit_at_k(retrieved, relevant, k)
        out[f"all_evidence_recall@{k}"] = all_evidence_recall_at_k(retrieved, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
        out[f"evidence_f1@{k}"] = evidence_f1(retrieved, relevant, k)
    out["mrr"] = mrr(retrieved, relevant)
    rank = mean_rank_first_relevant(retrieved, relevant)
    # Keep the legacy definition but store the sentinel as the top-k boundary
    # (k+1) instead of +inf so JSON stays finite and averages stay meaningful.
    boundary = max(ks) + 1 if ks else 1
    out["mean_rank_first_relevant"] = rank if rank is not None else float(boundary)
    # Diagnostics that do not redefine any existing metric:
    #  - mean_rank_on_hits: rank averaged over queries that actually hit
    #    (omitted when this query misses, so aggregation averages hits only).
    #  - miss_rate: 1 when a query with real evidence returns none of it.
    if relevant:
        if rank is not None:
            out["mean_rank_on_hits"] = rank
        out["miss_rate"] = 0.0 if rank is not None else 1.0
    return out


def aggregate_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Average numeric metrics across queries.

    Skips non-numeric and non-finite values so a single missed query cannot
    inject Infinity/NaN into the reported means or the serialized JSON.
    """
    if not per_query:
        return {}
    all_keys = {k for q in per_query for k in q}
    out: dict[str, float] = {}
    for k in all_keys:
        vals = [
            q[k]
            for q in per_query
            if k in q and isinstance(q[k], (int, float)) and math.isfinite(q[k])
        ]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap confidence interval for the mean."""
    import random
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    boot_means = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int((1 - ci) / 2 * n_resamples)]
    hi = boot_means[int((1 + ci) / 2 * n_resamples)]
    return lo, hi
