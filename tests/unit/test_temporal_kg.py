"""Unit tests for Phase 3 bi-temporal fact layer.

A three-episode timeline states a fact, then updates it twice ("finance" → "AI"
→ "AI and fintech"). We verify:

* the builder emits FactNodes with valid_from/valid_to and closes the older
  fact's validity window at the newer fact's start;
* SUPERSEDES / CONTRADICTS edges are added new→old on a genuine value change;
* the coordinator parses temporal cues into an event-time constraint;
* "currently" surfaces the current-era episode over the stale one, and
  "originally"/"before" surfaces the original.

Fact extraction is deterministic (regex), so these run without any extra deps.
"""

from __future__ import annotations

from memcity.coordinator.heuristic import parse_temporal_constraint
from memcity.graph.builder import MemoryCityGraphBuilder
from memcity.memory.schema import EdgeType, NodeType
from memcity.memory.store import Store
from memcity.retrieval.memory_city import MemoryCityRetriever

# Alice's focus evolves over time. Each update states "Alice's focus is now X".
CORPUS = [
    {
        "id": "ep_jan", "node_type": "episode",
        "user_text": "What is Alice focus?",
        "assistant_text": "Alice focus is finance.",
        "session_id": "s0", "turn_index": 0, "timestamp": 100.0,
        "source_episode_ids": ["ep_jan"],
    },
    {
        "id": "ep_apr", "node_type": "episode",
        "user_text": "Update on Alice.",
        "assistant_text": "Alice focus is now AI.",
        "session_id": "s1", "turn_index": 0, "timestamp": 400.0,
        "source_episode_ids": ["ep_apr"],
    },
    {
        "id": "ep_jul", "node_type": "episode",
        "user_text": "Latest on Alice.",
        "assistant_text": "Alice focus is now AI and fintech.",
        "session_id": "s2", "turn_index": 0, "timestamp": 700.0,
        "source_episode_ids": ["ep_jul"],
    },
]


def _fact_nodes(graph):
    return [
        (nid, d) for nid, d in graph.nodes(data=True)
        if d.get("node_type") == NodeType.FACT.value
    ]


def test_parse_temporal_constraint():
    assert parse_temporal_constraint("What does Alice currently focus on?").kind == "current"
    assert parse_temporal_constraint("What was Alice focus before?").kind == "before"
    assert parse_temporal_constraint("What did Alice focus on originally?").kind == "original"
    assert parse_temporal_constraint("What is Alice focus?").kind == "none"


def test_builder_emits_facts_with_validity_windows():
    store = Store()
    builder = MemoryCityGraphBuilder(store=store, enable_be=False, enable_facts=True)
    graph = builder.build(CORPUS)
    facts = _fact_nodes(graph)
    # Three facts for the single subject timeline.
    focus_facts = [d for _, d in facts if d["metadata"]["subject"].startswith("alice")]
    assert len(focus_facts) == 3
    focus_facts.sort(key=lambda d: d["valid_from"])
    # Oldest fact's window closes at the next fact's start; newest stays open.
    assert focus_facts[0]["valid_to"] == focus_facts[1]["valid_from"]
    assert focus_facts[-1]["valid_to"] is None
    assert focus_facts[-1]["metadata"]["current_state"] is True
    store.close()


def test_supersedes_and_contradicts_edges():
    store = Store()
    builder = MemoryCityGraphBuilder(store=store, enable_be=False, enable_facts=True)
    graph = builder.build(CORPUS)
    edge_types = {d.get("edge_type") for _, _, d in graph.edges(data=True)}
    assert EdgeType.SUPERSEDES.value in edge_types
    # Value genuinely changes (finance → AI → AI and fintech), so CONTRADICTS too.
    assert EdgeType.CONTRADICTS.value in edge_types
    store.close()


def _build_retriever(**kwargs) -> MemoryCityRetriever:
    r = MemoryCityRetriever(
        coordinator_enabled=False,
        enable_graph_expansion=False,
        enable_temporal=True,
        enable_community=False,
        enable_provenance=False,
        enable_be=False,
        enable_vector=False,
        **kwargs,
    )
    r.build(CORPUS)
    return r


def test_current_query_prefers_current_era_episode():
    r = _build_retriever(enable_temporal_kg=True)
    res = r.query("What does Alice focus on currently?", top_k=3)
    ids = res.ids()
    # The current-state episode (ep_jul) should rank above the stale ep_jan.
    assert ids.index("ep_jul") < ids.index("ep_jan")
    r.close()


def test_original_query_prefers_original_episode():
    r = _build_retriever(enable_temporal_kg=True)
    res = r.query("What did Alice focus on originally?", top_k=3)
    ids = res.ids()
    assert ids.index("ep_jan") < ids.index("ep_jul")
    r.close()


def test_registry_exposes_temporal_kg():
    from memcity.retrieval.registry import get_retriever_factory, list_methods

    assert "hybrid_temporal_kg" in list_methods()
    inst = get_retriever_factory("hybrid_temporal_kg")()
    assert inst._enable_temporal_kg is True
