"""Unit tests for Phase 2 weighted / PPR graph propagation.

These build a tiny multi-hop corpus where the correct evidence sits two hops from
the seed via a MENTIONS path but is lexically invisible to the query. BFS treats
every neighbour equally; weighted/PPR should rank the entity-linked episode above
a same-session but off-topic neighbour.

The graph builder pulls in sentence-transformers for semantic edges, but the
MENTIONS / NEXT / SAME_SESSION edges we rely on here are deterministic and do not
need the extra, so these tests run without embeddings.
"""

from __future__ import annotations

from memcity.retrieval.memory_city import MemoryCityRetriever

# Two sessions. Session A anchors the query ("Alice" + "Postgres"); the answer
# ("Alice chose PostgreSQL 16") lives in session B, linked back only through the
# shared entity "Alice" (MENTIONS) — a 2-hop path. A same-session distractor in A
# is adjacent to the seed but off-topic.
CORPUS = [
    {
        "id": "a1", "node_type": "episode",
        "user_text": "Alice asked about the database migration",
        "assistant_text": "We discussed options",
        "session_id": "sA", "turn_index": 0, "timestamp": 1.0,
        "source_episode_ids": ["a1"],
    },
    {
        "id": "a2", "node_type": "episode",
        "user_text": "The weather today is sunny and warm",
        "assistant_text": "Indeed a nice day",
        "session_id": "sA", "turn_index": 1, "timestamp": 2.0,
        "source_episode_ids": ["a2"],
    },
    {
        "id": "b1", "node_type": "episode",
        "user_text": "Alice confirmed the final choice",
        "assistant_text": "Alice chose PostgreSQL 16 for the project",
        "session_id": "sB", "turn_index": 0, "timestamp": 3.0,
        "source_episode_ids": ["b1"],
    },
]


def _build(**kwargs) -> MemoryCityRetriever:
    r = MemoryCityRetriever(
        coordinator_enabled=False,
        enable_graph_expansion=True,
        enable_temporal=False,
        enable_community=False,
        enable_provenance=False,
        enable_be=False,
        enable_vector=False,  # keep deterministic; BM25 only for seeds
        **kwargs,
    )
    r.build(CORPUS)
    return r


def test_ppr_promotes_entity_linked_episode_over_distractor():
    r = _build(graph_mode="ppr", degree_penalty=True)
    res = r.query("Alice database migration", top_k=3, trace=True)
    ids = res.ids()
    assert "b1" in ids  # entity-linked answer reached via MENTIONS
    # b1 (entity path) should outrank a2 (same-session but off-topic distractor).
    assert ids.index("b1") < ids.index("a2")
    r.close()


def test_weighted_mode_runs_and_supplements():
    r = _build(graph_mode="weighted")
    res = r.query("Alice database migration", top_k=3, trace=True)
    assert "b1" in res.ids()
    r.close()


def test_bfs_mode_still_works():
    r = _build(graph_mode="bfs")
    res = r.query("Alice database migration", top_k=3)
    assert len(res.items) > 0
    r.close()


def test_graph_supplement_never_evicts_strong_hit():
    # The strongest lexical hit for this query is a1; graph supplements must not
    # push it out of the top result.
    r = _build(graph_mode="ppr", degree_penalty=True)
    res = r.query("Alice database migration", top_k=3)
    assert res.items[0].id == "a1"
    r.close()


def test_community_edges_excluded_from_propagation():
    r = _build(graph_mode="ppr")
    # BELONGS_TO_COMMUNITY resolves to weight 0 in the edge-type table.
    assert r._edge_type_weights["BELONGS_TO_COMMUNITY"] == 0.0
    r.close()


def test_registry_exposes_ppr_variants():
    from memcity.retrieval.registry import get_retriever_factory, list_methods

    for name in ("hybrid_ppr", "hybrid_graph_weighted", "hybrid_graph_degree"):
        assert name in list_methods()
    inst = get_retriever_factory("hybrid_ppr")()
    assert inst._graph_mode == "ppr"
    assert inst._degree_penalty is True
