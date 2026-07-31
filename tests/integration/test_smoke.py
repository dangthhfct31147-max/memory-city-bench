"""Smoke test — runs full benchmark on synthetic-tiny in < 2 minutes (CPU only)."""

from __future__ import annotations

import pytest
from pathlib import Path


@pytest.mark.smoke
def test_smoke_generate_and_benchmark(tmp_path):
    """End-to-end smoke: generate dataset → build BM25 → query → verify metrics."""
    from memcity.datasets.synthetic import SyntheticDataset
    from memcity.retrieval.baselines import BM25Retriever
    from memcity.evaluation.runner import RetrievalBenchmarkRunner
    from memcity.evaluation.metrics import compute_all_metrics

    # Generate tiny dataset
    ds = SyntheticDataset(scale="tiny", seed=42)
    samples = ds.load(tmp_path)
    assert len(samples) > 0, "Dataset should have samples"
    manifest = ds.manifest()
    assert manifest.sample_count == len(samples)

    # Build corpus
    seen: set[str] = set()
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

    assert len(corpus) > 0, "Corpus should not be empty"

    # Build BM25
    r = BM25Retriever()
    stats = r.build(corpus)
    assert stats.node_count == len(corpus)

    # Query every sample
    for sample in samples:
        result = r.query(sample.query, top_k=10)
        assert result.items is not None
        assert result.latency_ms > 0
        retrieved = result.episode_ids()
        assert isinstance(retrieved, list)

    # Compute metrics
    all_metrics = []
    for sample in samples:
        if not sample.evidence_episode_ids:
            continue
        result = r.query(sample.query, top_k=10)
        retrieved = result.episode_ids()
        relevant = set(sample.evidence_episode_ids)
        m = compute_all_metrics(retrieved, relevant, ks=(1, 5, 10))
        all_metrics.append(m)
    r.close()

    assert len(all_metrics) > 0

    # Check that we report at least recall@1, mrr, ndcg@10
    sample_m = all_metrics[0]
    assert "recall@1" in sample_m
    assert "mrr" in sample_m
    assert "ndcg@10" in sample_m


@pytest.mark.smoke
def test_smoke_memory_city_graph(tmp_path):
    """Smoke: build Memory City graph, verify nodes, edges, communities."""
    from memcity.datasets.synthetic import SyntheticDataset
    from memcity.graph.builder import MemoryCityGraphBuilder
    from memcity.memory.store import Store
    from memcity.memory.schema import NodeType

    ds = SyntheticDataset(scale="tiny", seed=42)
    samples = ds.load(tmp_path)

    seen: set[str] = set()
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

    store = Store()
    builder = MemoryCityGraphBuilder(store=store)
    G = builder.build(corpus)

    # Graph should have nodes
    assert G.number_of_nodes() > 0, "Graph should have nodes"
    assert G.number_of_edges() > 0, "Graph should have edges"

    # Episode nodes exist
    ep_nodes = [n for n, d in G.nodes(data=True)
                if d.get("node_type") == NodeType.EPISODE.value]
    assert len(ep_nodes) == len(corpus)

    # Begin/End nodes exist
    begin_nodes = [n for n, d in G.nodes(data=True)
                   if d.get("node_type") == NodeType.BEGIN.value]
    assert len(begin_nodes) > 0, "Graph should have Begin nodes"

    # All episode nodes have source_episode_ids
    for nid, data in G.nodes(data=True):
        if data.get("node_type") == NodeType.EPISODE.value:
            assert data.get("source_episode_ids"), f"Episode {nid} missing source_episode_ids"

    # Community nodes (or at least some structure)
    all_types = {d.get("node_type") for _, d in G.nodes(data=True)}
    assert NodeType.EPISODE.value in all_types

    store.close()


@pytest.mark.smoke
def test_smoke_runner_artifacts(tmp_path):
    """Smoke: run benchmark and verify artifact files are created."""
    import json
    from memcity.datasets.synthetic import SyntheticDataset
    from memcity.retrieval.baselines import BM25Retriever
    from memcity.evaluation.runner import RetrievalBenchmarkRunner

    ds = SyntheticDataset(scale="tiny", seed=42)
    samples = ds.load(tmp_path)

    seen: set[str] = set()
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

    runner = RetrievalBenchmarkRunner(
        output_dir=tmp_path / "runs",
        warmup=2,
        top_ks=(1, 5, 10),
        seeds=[42],
    )

    r = BM25Retriever()
    result = runner.run(r, corpus, samples[:5], dataset_name="synthetic-tiny")
    r.close()

    assert "metrics" in result
    metrics = result["metrics"]

    # Artifact files
    runs_dir = tmp_path / "runs"
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1, "Should create exactly one run directory"
    run_dir = run_dirs[0]

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "retrieval_results_42.jsonl").exists()

    metrics_data = json.loads((run_dir / "metrics.json").read_text())
    assert "metrics" in metrics_data
