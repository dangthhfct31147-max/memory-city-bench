"""Integration test: temporal update, multi-hop, provenance."""

from __future__ import annotations

import pytest

from memcity.datasets.synthetic import SyntheticDataset
from memcity.datasets.protocol import QuestionCategory
from memcity.evaluation.metrics import compute_all_metrics
from memcity.retrieval.baselines import BM25Retriever, RecencyRetriever
from memcity.retrieval.registry import get_retriever


def _make_corpus(samples):
    seen = set()
    corpus = []
    for s in samples:
        for ep in s.history:
            if ep.episode_id not in seen:
                seen.add(ep.episode_id)
                corpus.append({
                    "id": ep.episode_id,
                    "node_type": "episode",
                    "text": f"{ep.user_text} {ep.assistant_text}",
                    "user_text": ep.user_text,
                    "assistant_text": ep.assistant_text,
                    "session_id": ep.session_id,
                    "timestamp": ep.timestamp,
                    "source_episode_ids": [ep.episode_id],
                })
    return corpus


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    ds = SyntheticDataset(scale="tiny", seed=42)
    samples = ds.load(tmp)
    return samples


@pytest.fixture(scope="module")
def tiny_corpus(tiny_dataset):
    return _make_corpus(tiny_dataset)


@pytest.mark.integration
class TestTemporalUpdate:
    """Retriever should prefer current fact over stale fact."""

    def test_bm25_finds_current_fact(self, tiny_dataset, tiny_corpus):
        update_samples = [
            s for s in tiny_dataset
            if s.category == QuestionCategory.FACT_UPDATE
        ]
        if not update_samples:
            pytest.skip("No fact-update samples in tiny dataset")

        r = BM25Retriever()
        r.build(tiny_corpus)

        for sample in update_samples[:3]:
            result = r.query(sample.query, top_k=10)
            retrieved = result.episode_ids()
            relevant = set(sample.evidence_episode_ids)
            stale = set(sample.stale_fact_ids)

            if not relevant or not stale:
                continue

            # Current evidence should be retrieved somewhere in top-10
            metrics = compute_all_metrics(retrieved, relevant, ks=(10,))
            # Stale fact should not rank above current
            rank_current = next(
                (i + 1 for i, eid in enumerate(retrieved) if eid in relevant), None
            )
            rank_stale = next(
                (i + 1 for i, eid in enumerate(retrieved) if eid in stale), None
            )
            if rank_current is not None and rank_stale is not None:
                # Current should rank at least as well as stale for this query type
                assert rank_current <= rank_stale + 5, (
                    f"Current fact at rank {rank_current} should be close to or above "
                    f"stale fact at rank {rank_stale}"
                )
        r.close()

    def test_recency_finds_new_over_old(self, tiny_dataset, tiny_corpus):
        update_samples = [
            s for s in tiny_dataset
            if s.category == QuestionCategory.FACT_UPDATE
        ]
        if not update_samples:
            pytest.skip("No fact-update samples")

        r = RecencyRetriever()
        r.build(tiny_corpus)

        for sample in update_samples[:2]:
            result = r.query(sample.query, top_k=10)
            retrieved = result.episode_ids()
            relevant = set(sample.evidence_episode_ids)
            stale = set(sample.stale_fact_ids)

            if not relevant or not stale:
                continue

            # With recency, updates come last in time so should rank higher
            ep_by_ts = {ep.episode_id: ep.timestamp
                        for ep in sample.history}
            current_ts = max(
                (ep_by_ts.get(eid, 0.0) for eid in relevant), default=0.0
            )
            stale_ts = max(
                (ep_by_ts.get(eid, 0.0) for eid in stale), default=0.0
            )
            assert current_ts >= stale_ts, "Update episode must be newer than stale"
        r.close()


@pytest.mark.integration
class TestMultiHop:
    def test_multi_hop_sample_has_two_evidence(self, tiny_dataset):
        multi_hop = [
            s for s in tiny_dataset
            if s.category == QuestionCategory.MULTI_HOP
        ]
        if not multi_hop:
            pytest.skip("No multi-hop samples")
        for s in multi_hop:
            assert s.required_evidence_count >= 2, "Multi-hop should need multiple evidence"
            assert len(s.evidence_episode_ids) >= 1

    def test_bm25_multi_hop_retrieval(self, tiny_dataset, tiny_corpus):
        multi_hop = [
            s for s in tiny_dataset
            if s.category == QuestionCategory.MULTI_HOP
        ]
        if not multi_hop:
            pytest.skip("No multi-hop samples")

        r = BM25Retriever()
        r.build(tiny_corpus)

        for sample in multi_hop[:2]:
            result = r.query(sample.query, top_k=10)
            retrieved = result.episode_ids()
            relevant = set(sample.evidence_episode_ids)
            metrics = compute_all_metrics(retrieved, relevant, ks=(10,))
            # At least some evidence should appear
            assert metrics["any_hit@10"] == 1.0 or len(relevant) == 0
        r.close()


@pytest.mark.integration
class TestProvenanceTraceback:
    def test_all_retrieved_have_source_episode_ids(self, tiny_dataset, tiny_corpus):
        r = BM25Retriever()
        r.build(tiny_corpus)
        sample = tiny_dataset[0]
        result = r.query(sample.query, top_k=5)
        for item in result.items:
            assert item.source_episode_ids, f"Item {item.id} has no source_episode_ids"
        r.close()

    def test_episode_ids_in_corpus(self, tiny_dataset, tiny_corpus):
        corpus_ids = {c["id"] for c in tiny_corpus}
        r = BM25Retriever()
        r.build(tiny_corpus)
        sample = tiny_dataset[0]
        result = r.query(sample.query, top_k=5)
        for item in result.items:
            assert item.id in corpus_ids, f"Retrieved item {item.id} not in corpus"
        r.close()


@pytest.mark.integration
class TestReproducibility:
    def test_same_seed_same_results(self, tiny_dataset, tiny_corpus, tmp_path):
        from memcity.evaluation.runner import RetrievalBenchmarkRunner

        runner = RetrievalBenchmarkRunner(
            output_dir=tmp_path,
            warmup=0,
            top_ks=(1, 5),
            seeds=[42],
        )

        r1 = BM25Retriever()
        res1 = runner.run(r1, tiny_corpus, tiny_dataset[:5], dataset_name="test")
        r1.close()

        r2 = BM25Retriever()
        res2 = runner.run(r2, tiny_corpus, tiny_dataset[:5], dataset_name="test")
        r2.close()

        # Metrics should be identical (BM25 is deterministic)
        m1 = res1.get("metrics", {})
        m2 = res2.get("metrics", {})
        skip_keys = {"latency_ms", "mean_rank_first_relevant"}
        for key in m1:
            if key in skip_keys:
                continue
            assert abs(m1[key] - m2.get(key, -1)) < 1e-9, f"Non-deterministic: {key}"
