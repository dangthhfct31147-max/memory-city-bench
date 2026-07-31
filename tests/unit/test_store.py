"""Unit tests for SQLite store."""

from __future__ import annotations

import time

import pytest

from memcity.memory.store import Store


@pytest.fixture
def store():
    s = Store()  # in-memory
    yield s
    s.close()


class TestEpisodes:
    def test_upsert_and_get(self, store):
        store.upsert_episode("ep1", "s1", 0, "hello", "world", time.time())
        ep = store.get_episode("ep1")
        assert ep is not None
        assert ep["user_text"] == "hello"

    def test_all_episodes(self, store):
        store.upsert_episode("ep1", "s1", 0, "a", "b", 1.0)
        store.upsert_episode("ep2", "s1", 1, "c", "d", 2.0)
        eps = store.all_episodes()
        assert len(eps) == 2
        assert eps[0]["id"] == "ep1"

    def test_fts_search(self, store):
        store.upsert_episode("ep1", "s1", 0, "Alice uses Python", "Correct", 1.0)
        store.upsert_episode("ep2", "s1", 1, "Bob uses FastAPI", "Right", 2.0)
        results = store.search_episodes_fts("Python", limit=5)
        assert any(r["id"] == "ep1" for r in results)

    def test_upsert_idempotent(self, store):
        store.upsert_episode("ep1", "s1", 0, "hello", "world", 1.0)
        store.upsert_episode("ep1", "s1", 0, "hello updated", "world", 1.0)
        ep = store.get_episode("ep1")
        assert ep["user_text"] == "hello updated"


class TestNodes:
    def test_upsert_and_get(self, store):
        node = {
            "id": "n1",
            "node_type": "episode",
            "text": "test node",
            "session_id": "s1",
            "timestamp": 1.0,
            "source_episode_ids": ["ep1"],
        }
        store.upsert_node(node)
        n = store.get_node("n1")
        assert n is not None
        assert n["text"] == "test node"
        assert "ep1" in n["source_episode_ids"]

    def test_filter_by_type(self, store):
        store.upsert_node({"id": "n1", "node_type": "episode", "text": "e", "timestamp": 1.0})
        store.upsert_node({"id": "n2", "node_type": "entity", "text": "ent", "timestamp": 1.0})
        episodes = store.all_nodes("episode")
        assert len(episodes) == 1
        assert episodes[0]["id"] == "n1"


class TestEdges:
    def test_upsert_and_get(self, store):
        edge = {
            "id": "edge1",
            "src_id": "n1",
            "dst_id": "n2",
            "edge_type": "NEXT",
            "timestamp": 1.0,
            "source_episode_ids": ["ep1"],
        }
        store.upsert_edge(edge)
        edges = store.get_edges(node_id="n1")
        assert len(edges) == 1
        assert edges[0]["edge_type"] == "NEXT"

    def test_filter_by_type(self, store):
        store.upsert_edge({"id": "e1", "src_id": "n1", "dst_id": "n2",
                           "edge_type": "NEXT", "timestamp": 1.0})
        store.upsert_edge({"id": "e2", "src_id": "n1", "dst_id": "n3",
                           "edge_type": "MENTIONS", "timestamp": 1.0})
        next_edges = store.get_edges(node_id="n1", edge_type="NEXT")
        assert len(next_edges) == 1


class TestEmbeddings:
    def test_store_and_load(self, store):
        vec = [0.1, 0.2, 0.3]
        store.store_embedding("n1", vec, "test-model")
        loaded = store.load_embedding("n1")
        assert loaded is not None
        assert len(loaded) == 3
        assert abs(loaded[0] - 0.1) < 1e-5

    def test_load_all(self, store):
        store.store_embedding("n1", [1.0, 0.0], "m")
        store.store_embedding("n2", [0.0, 1.0], "m")
        all_embs = store.load_all_embeddings()
        assert "n1" in all_embs and "n2" in all_embs


class TestRuns:
    def test_create_and_finish(self, store):
        store.create_run("run1", "synthetic", "bm25", {"k": 1}, {}, time.time())
        run = store.get_run("run1")
        assert run is not None
        assert run["status"] == "running"
        store.finish_run("run1", {"recall@1": 0.8})
        run = store.get_run("run1")
        assert run["status"] == "done"
