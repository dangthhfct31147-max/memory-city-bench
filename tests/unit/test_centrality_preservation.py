"""Regression test: strong evidence must not be lost to low graph centrality.

A raw episode that is a decisive lexical match for the query must survive to the
final top-k even when it is an isolated, low-centrality node in the Memory City
graph. Graph expansion, temporal, and provenance stages may only *supplement*
scores — they must never evict a strong lexical/vector hit.
"""

from __future__ import annotations

from memcity.retrieval.memory_city import MemoryCityRetriever


def _corpus() -> list[dict]:
    """Build a corpus where most episodes cluster on a shared topic, plus one
    isolated episode holding the unique answer to a rare-term query.

    The clustered episodes share entities/terms so the graph connects them
    densely (high centrality). The evidence episode uses a rare token that
    appears nowhere else, so it is lexically decisive but graph-peripheral.
    """
    clustered = [
        "Alice and Bob discussed the platform roadmap and the platform team",
        "The platform team reviewed the platform roadmap with Alice",
        "Bob and Alice planned the platform migration for the platform team",
        "Alice updated the platform roadmap and told the platform team",
        "The platform team and Alice closed the platform roadmap review",
    ]
    corpus = [
        {
            "id": f"ep_{i:03d}",
            "node_type": "episode",
            "text": text,
            "user_text": text,
            "assistant_text": "",
            "session_id": "s_cluster",
            "timestamp": float(i),
            "source_episode_ids": [f"ep_{i:03d}"],
        }
        for i, text in enumerate(clustered)
    ]
    # Isolated evidence episode: unique rare token, unrelated to the cluster.
    corpus.append(
        {
            "id": "ep_evidence",
            "node_type": "episode",
            "text": "The zorblaxian artifact was catalogued in vault 7",
            "user_text": "The zorblaxian artifact was catalogued in vault 7",
            "assistant_text": "",
            "session_id": "s_isolated",
            "timestamp": 999.0,
            "source_episode_ids": ["ep_evidence"],
        }
    )
    return corpus


def test_isolated_evidence_survives_to_top_k():
    """The decisive lexical hit is returned first despite low centrality."""
    retriever = MemoryCityRetriever(enable_vector=False)
    retriever.build(_corpus())
    result = retriever.query("Where was the zorblaxian artifact catalogued?", top_k=5)
    ids = result.episode_ids()
    assert "ep_evidence" in ids, "isolated evidence episode dropped from top-k"
    assert ids[0] == "ep_evidence", "strong lexical hit was displaced from rank 1"
    retriever.close()


def test_graph_expansion_actually_runs_when_routed():
    """The graph route must genuinely fire — the old test never exercised it.

    The reviewer's point (issue #10): the previous query used a bare rare token
    with no proper noun, so the coordinator's confidence stayed below the fallback
    threshold and Memory City silently fell back to Hybrid RRF. Naming a proper
    noun ("Alice") that the coordinator recognises routes to LOCAL_GRAPH and makes
    graph expansion actually run. This test asserts the *mechanism* fired; it does
    not assert ranking, because adding the entity term deliberately shifts the
    lexical signal toward the connected cluster.
    """
    retriever = MemoryCityRetriever(enable_vector=False, enable_graph_expansion=True)
    retriever.build(_corpus())
    result = retriever.query(
        "What did Alice say about the platform roadmap?", top_k=3, trace=True
    )
    assert result.trace is not None
    assert "LOCAL_GRAPH" in result.trace.routes
    assert "graph_expand" in result.trace.stage_latency_ms
    # Graph expansion should have surfaced at least one supplemental candidate.
    assert result.trace.graph_candidates
    retriever.close()


def test_graph_expansion_does_not_evict_strong_hit():
    """With graph expansion on, the strong hit still ranks first.

    Graph expansion adds neighbours of the dense cluster, but those are
    supplements: they must not push the decisive isolated hit out of top-k.

    The coordinator is disabled so graph expansion fires unconditionally (pure
    ablation) without the query needing an entity term that would itself distort
    the lexical ranking.
    """
    retriever = MemoryCityRetriever(
        enable_vector=False,
        enable_graph_expansion=True,
        coordinator_enabled=False,
    )
    retriever.build(_corpus())
    result = retriever.query("zorblaxian artifact vault", top_k=3, trace=True)
    ids = result.episode_ids()
    assert "ep_evidence" in ids
    assert ids[0] == "ep_evidence"
    # Graph expansion must have actually run (coordinator off → unconditional).
    assert result.trace is not None
    assert "graph_expand" in result.trace.stage_latency_ms
    retriever.close()


def test_provenance_does_not_reorder_single_source_episodes():
    """All episodes have one source id, so provenance must not change ranking.

    This is the specific bug from the review: the old provenance rerank added a
    uniform boost to every single-source episode, which was a no-op that still
    ran a full re-sort. The rare-term hit must stay first.
    """
    retriever = MemoryCityRetriever(enable_vector=False)
    retriever.build(_corpus())
    result = retriever.query("zorblaxian artifact", top_k=5)
    ids = result.episode_ids()
    assert ids[0] == "ep_evidence"
    retriever.close()
