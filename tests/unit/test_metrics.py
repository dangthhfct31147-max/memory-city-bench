"""Unit tests for retrieval metrics."""

from __future__ import annotations

import math

import pytest

from memcity.evaluation.metrics import (
    aggregate_metrics,
    all_evidence_recall_at_k,
    any_hit_at_k,
    bootstrap_ci,
    compute_all_metrics,
    compute_rrf,
    evidence_f1,
    mean_rank_first_relevant,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 5) == 1.0

    def test_zero(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, 5) == 0.0

    def test_partial(self):
        assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 3) == pytest.approx(2 / 3)

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set(), 5) == 1.0

    def test_k_truncates(self):
        # "b" is at position 3, beyond k=2
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == pytest.approx(0.5)


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_zero(self):
        assert precision_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_partial(self):
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(2 / 3)

    def test_k_zero(self):
        assert precision_at_k([], {"a"}, 0) == 0.0


class TestAnyHitAtK:
    def test_hit(self):
        assert any_hit_at_k(["x", "a"], {"a"}, 2) == 1.0

    def test_miss(self):
        assert any_hit_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_beyond_k(self):
        assert any_hit_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_empty_relevant(self):
        assert any_hit_at_k(["x"], set(), 5) == 1.0


class TestAllEvidenceRecallAtK:
    def test_all_found(self):
        assert all_evidence_recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0

    def test_partial(self):
        assert all_evidence_recall_at_k(["a"], {"a", "b"}, 5) == pytest.approx(0.5)

    def test_none_found(self):
        assert all_evidence_recall_at_k(["x"], {"a", "b"}, 5) == 0.0


class TestMRR:
    def test_first(self):
        assert mrr(["a", "b"], {"a"}) == 1.0

    def test_second(self):
        assert mrr(["x", "a"], {"a"}) == pytest.approx(0.5)

    def test_not_found(self):
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_empty(self):
        assert mrr([], {"a"}) == 0.0


class TestNDCGAtK:
    def test_perfect(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_reversed(self):
        perfect = ndcg_at_k(["a", "b"], {"a", "b"}, 2)
        suboptimal = ndcg_at_k(["x", "a"], {"a", "b"}, 2)
        assert perfect >= suboptimal

    def test_zero(self):
        assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["x"], set(), 3) == 0.0


class TestMeanRank:
    def test_first(self):
        assert mean_rank_first_relevant(["a", "b"], {"a"}) == 1.0

    def test_not_found(self):
        assert mean_rank_first_relevant(["x"], {"a"}) is None


class TestRRF:
    def test_agreement_boosts(self):
        fused = compute_rrf([["a", "b", "c"], ["a", "c", "b"]])
        ids = [doc_id for doc_id, _ in fused]
        assert ids[0] == "a"  # agreed upon by both lists at rank 1

    def test_all_unique(self):
        fused = compute_rrf([["a"], ["b"]])
        assert len(fused) == 2

    def test_empty(self):
        assert compute_rrf([]) == []


class TestComputeAllMetrics:
    def test_keys(self):
        m = compute_all_metrics(["a", "b"], {"a"}, ks=(1, 5))
        assert "recall@1" in m
        assert "recall@5" in m
        assert "mrr" in m
        assert "ndcg@5" in m

    def test_values_in_range(self):
        m = compute_all_metrics(["a", "x", "b", "y"], {"a", "b"}, ks=(1, 3, 5))
        for v in m.values():
            assert 0.0 <= v <= 1.0 or v == float("inf")


class TestAggregateMetrics:
    def test_mean(self):
        queries = [{"recall@1": 1.0}, {"recall@1": 0.0}]
        agg = aggregate_metrics(queries)
        assert agg["recall@1"] == pytest.approx(0.5)

    def test_empty(self):
        assert aggregate_metrics([]) == {}


class TestBootstrapCI:
    def test_returns_tuple(self):
        lo, hi = bootstrap_ci([0.5, 0.6, 0.7], n_resamples=100, seed=0)
        assert lo <= hi

    def test_empty(self):
        lo, hi = bootstrap_ci([], n_resamples=100, seed=0)
        assert lo == 0.0 and hi == 0.0
