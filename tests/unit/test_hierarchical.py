"""Unit tests for the Phase 6 hierarchical summary tier (RAPTOR-style).

The summary tree is deterministic and extractive (TF-IDF), so these tests run
without the embeddings extra. They verify:

* the builder emits SUMMARY nodes with SUMMARIZES / PARENT_SUMMARY edges;
* summaries route a global query to the raw episodes beneath them;
* only raw episodes ever surface as final evidence (summaries never do);
* the registry exposes the ``hybrid_tree`` ablation variant.
"""

from __future__ import annotations

from memcity.graph.builder import MemoryCityGraphBuilder
from memcity.memory.schema import EdgeType, NodeType
from memcity.memory.store import Store
from memcity.retrieval.memory_city import MemoryCityRetriever

# Three sessions, each a distinct topic. The query is a broad/overview one whose
# terms appear across a session's turns but in no single decisive lexical turn.
CORPUS = [
    {
        "id": "a1", "node_type": "episode",
        "user_text": "We planned the garden layout for spring vegetables",
        "assistant_text": "Tomatoes and peppers go in the raised beds",
        "session_id": "sA", "turn_index": 0, "timestamp": 1.0,
        "source_episode_ids": ["a1"],
    },
    {
        "id": "a2", "node_type": "episode",
        "user_text": "The garden needs compost before planting season",
        "assistant_text": "We ordered organic compost for the beds",
        "session_id": "sA", "turn_index": 1, "timestamp": 2.0,
        "source_episode_ids": ["a2"],
    },
    {
        "id": "b1", "node_type": "episode",
        "user_text": "The quarterly budget review covered marketing spend",
        "assistant_text": "Marketing was over budget by ten percent",
        "session_id": "sB", "turn_index": 0, "timestamp": 3.0,
        "source_episode_ids": ["b1"],
    },
    {
        "id": "b2", "node_type": "episode",
        "user_text": "Finance flagged the travel budget overrun",
        "assistant_text": "Travel costs will be capped next quarter",
        "session_id": "sB", "turn_index": 1, "timestamp": 4.0,
        "source_episode_ids": ["b2"],
    },
]


def _build_retriever(**kwargs) -> MemoryCityRetriever:
    r = MemoryCityRetriever(
        coordinator_enabled=False,
        enable_graph_expansion=False,
        enable_temporal=False,
        enable_community=False,
        enable_provenance=False,
        enable_be=False,
        enable_vector=False,  # deterministic: BM25 seeds only
        enable_hierarchical=True,
        **kwargs,
    )
    r.build(CORPUS)
    return r


def test_builder_emits_summary_nodes_and_edges():
    store = Store()
    builder = MemoryCityGraphBuilder(store=store, enable_summaries=True)
    g = builder.build(CORPUS)

    summaries = [
        nid for nid, d in g.nodes(data=True)
        if d.get("node_type") == NodeType.SUMMARY.value
    ]
    # One session-level summary per session.
    assert len(summaries) == 2

    for sid in summaries:
        data = g.nodes[sid]
        assert data["metadata"]["level"] == 1
        assert data["text"]  # extractive digest is non-empty
        # source episodes flatten to the session's turns
        assert set(data["source_episode_ids"]) <= {"a1", "a2", "b1", "b2"}

    # Both edge directions exist between a summary and its children.
    etypes = {d.get("edge_type") for _, _, d in g.edges(data=True)}
    assert EdgeType.SUMMARIZES.value in etypes
    assert EdgeType.PARENT_SUMMARY.value in etypes
    store.close()


def test_multi_level_tree_builds_root():
    store = Store()
    builder = MemoryCityGraphBuilder(
        store=store, enable_summaries=True,
        summary_max_levels=2, summary_branching_factor=2,
    )
    g = builder.build(CORPUS)
    levels = {
        d.get("metadata", {}).get("level")
        for _, d in g.nodes(data=True)
        if d.get("node_type") == NodeType.SUMMARY.value
    }
    # Two session summaries (level 1) grouped under one level-2 summary.
    assert levels == {1, 2}
    store.close()


def test_tree_search_routes_global_query_to_raw_episodes():
    r = _build_retriever()
    # A garden-overview query should surface session A's raw episodes via its
    # summary, even though no single turn lexically dominates.
    res = r.query("overview of the garden planting plans", top_k=4, trace=True)
    ids = res.ids()
    assert "a1" in ids or "a2" in ids
    assert res.trace is not None
    # The tree pool carried raw episodes (not summary node ids).
    assert all(not tid.startswith("summary_") for tid in res.trace.tree_candidates)
    r.close()


def test_summaries_never_become_final_evidence():
    r = _build_retriever()
    res = r.query("overview of the garden and budget topics", top_k=10)
    for item in res.items:
        assert item.node_type == "episode"
        assert not item.id.startswith("summary_")
    r.close()


def test_tree_disabled_yields_no_summary_nodes():
    store = Store()
    builder = MemoryCityGraphBuilder(store=store, enable_summaries=False)
    g = builder.build(CORPUS)
    summaries = [
        nid for nid, d in g.nodes(data=True)
        if d.get("node_type") == NodeType.SUMMARY.value
    ]
    assert summaries == []
    store.close()


def test_registry_exposes_hybrid_tree():
    from memcity.retrieval.registry import (
        get_retriever_factory,
        independent_ablation_methods,
        list_methods,
    )

    assert "hybrid_tree" in list_methods()
    assert "hybrid_tree" in independent_ablation_methods()
    inst = get_retriever_factory("hybrid_tree")()
    assert inst._enable_hierarchical is True
