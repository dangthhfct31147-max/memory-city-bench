"""Memory City graph builder — deterministic, no LLM required."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

from memcity.memory.schema import (
    CreationMethod,
    EdgeType,
    NodeType,
)
from memcity.memory.store import Store
from memcity.utils.helpers import make_id, stable_hash

# Simple regex-based entity detection
_ENTITY_RE = re.compile(
    r"\b(?:Alice|Bob|Charlie|Project\s+\w+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)


def extract_entities(text: str) -> list[str]:
    return list({m.group() for m in _ENTITY_RE.finditer(text)})


# Deterministic subject–predicate–object extraction (Phase 3). Matches the
# statement forms the synthetic generator and most conversational updates use:
#   "<subject> is now <object>"   → an update (predicate normalised to "is")
#   "<subject> is <object>"       → a plain assertion
#   "<subject> changed to <object>" / "switched to" / "moved to" / "prefers"
# The subject is captured up to the predicate; the object up to sentence end.
# This is intentionally conservative: a miss just means no fact node, never a
# wrong one, and retrieval degrades to the recency nudge.
_FACT_RE = re.compile(
    r"(?P<subject>[A-Za-z][\w '/-]*?)\s+"
    r"(?P<pred>is now|is|are now|are|changed to|switched to|moved to|"
    r"now prefers|prefers|now uses|uses)\s+"
    r"(?P<object>[^.!?\n]+)",
    re.I,
)
# Predicates that assert a (possibly updated) current state; normalised so
# "is"/"is now"/"changed to" all collapse to one predicate key per subject.
_STATE_PREDICATE = "state"


class MemoryCityGraphBuilder:
    """Build a Memory City graph from episode turns.

    The graph is deterministic and does not require an LLM.
    Optionally uses sentence-transformers for semantic edges.
    """

    def __init__(
        self,
        store: Store,
        semantic_threshold: float = 0.75,
        max_edges_per_node: int = 10,
        num_topic_hubs: int = 8,
        community_algorithm: str = "greedy",
        embedding_model: str | None = None,
        enable_be: bool = True,
        enable_facts: bool = False,
        enable_summaries: bool = False,
        summary_max_levels: int = 1,
        summary_max_sentences: int = 3,
        summary_branching_factor: int = 5,
    ) -> None:
        self._store = store
        self._sem_thresh = semantic_threshold
        self._max_edges = max_edges_per_node
        self._n_hubs = num_topic_hubs
        self._community_alg = community_algorithm
        self._emb_model_name = embedding_model
        self._emb_model: Any = None
        self._enable_be = enable_be
        self._enable_facts = enable_facts
        self._enable_summaries = enable_summaries
        self._summary_max_levels = max(1, summary_max_levels)
        self._summary_max_sentences = max(1, summary_max_sentences)
        self._summary_branching = max(2, summary_branching_factor)
        self.G = nx.DiGraph()

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, episodes: list[dict]) -> nx.DiGraph:
        """Build complete Memory City graph from raw episode dicts."""
        self.G = nx.DiGraph()
        self._build_episode_nodes(episodes)
        if self._enable_be:
            self._build_be_nodes(episodes)
        self._build_entity_nodes(episodes)
        if self._enable_facts:
            self._build_fact_nodes(episodes)
        self._build_semantic_edges()
        self._build_topic_hubs()
        self._build_communities()
        if self._enable_summaries:
            self._build_summary_tree(episodes)
        return self.G

    # ── Episode nodes and temporal edges ─────────────────────────────────────

    def _build_episode_nodes(self, episodes: list[dict]) -> None:
        by_session: dict[str, list[dict]] = defaultdict(list)

        for ep in episodes:
            ep_id = ep["id"]
            raw_text = f"{ep.get('user_text', '')} {ep.get('assistant_text', '')}".strip()
            node_data = {
                "id": ep_id,
                "node_type": NodeType.EPISODE.value,
                # Honour the Phase 4 contextual prefix for indexing/semantic edges;
                # falls back to raw when contextual indexing is off.
                "text": ep.get("indexed_text") or raw_text,
                "session_id": ep.get("session_id", ""),
                "timestamp": ep.get("timestamp", 0.0),
                "source_episode_ids": [ep_id],
                "creation_method": CreationMethod.DETERMINISTIC.value,
                "content_hash": stable_hash(ep.get("user_text", "") + ep.get("assistant_text", "")),
                "metadata": ep.get("metadata", {}),
            }
            self.G.add_node(ep_id, **node_data)
            self._store.upsert_node(node_data)
            by_session[ep.get("session_id", "")].append(ep)

        # NEXT edges (temporal within session)
        for session_id, sess_eps in by_session.items():
            sess_eps.sort(key=lambda e: (e.get("turn_index", 0), e.get("timestamp", 0.0)))
            for i in range(len(sess_eps) - 1):
                self._add_edge(
                    sess_eps[i]["id"], sess_eps[i + 1]["id"],
                    EdgeType.NEXT, session_id,
                    [sess_eps[i]["id"], sess_eps[i + 1]["id"]],
                )

        # SAME_SESSION edges (all pairs in same session — star from first turn)
        for session_id, sess_eps in by_session.items():
            if len(sess_eps) < 2:
                continue
            root = sess_eps[0]["id"]
            for ep in sess_eps[1:]:
                self._add_edge(
                    root, ep["id"], EdgeType.SAME_SESSION, session_id,
                    [root, ep["id"]],
                )

    # ── Begin/End nodes ───────────────────────────────────────────────────────

    def _build_be_nodes(self, episodes: list[dict]) -> None:
        by_session: dict[str, list[dict]] = defaultdict(list)
        for ep in episodes:
            by_session[ep.get("session_id", "")].append(ep)

        for session_id, sess_eps in by_session.items():
            sess_eps.sort(key=lambda e: (e.get("turn_index", 0), e.get("timestamp", 0.0)))
            first, last = sess_eps[0], sess_eps[-1]

            b_id = make_id("begin", f"{session_id}_begin")
            e_id = make_id("end", f"{session_id}_end")

            b_text = (first.get("user_text") or first.get("text") or "")[:256]
            # assistant_text may be empty for LoCoMo (all text is in user_text/text).
            # Fall back to the full turn text so End nodes are never blank.
            e_text = (last.get("assistant_text") or last.get("text") or last.get("user_text") or "")[:256]

            for node_id, text, ntype in ((b_id, b_text, NodeType.BEGIN), (e_id, e_text, NodeType.END)):
                node_data = {
                    "id": node_id,
                    "node_type": ntype.value,
                    "text": text,
                    "session_id": session_id,
                    "timestamp": first["timestamp"] if ntype == NodeType.BEGIN else last["timestamp"],
                    "source_episode_ids": [ep["id"] for ep in sess_eps],
                    "creation_method": CreationMethod.DETERMINISTIC.value,
                    "content_hash": stable_hash(text),
                }
                self.G.add_node(node_id, **node_data)
                self._store.upsert_node(node_data)

            self._add_edge(b_id, first["id"], EdgeType.BEGINS, session_id, [first["id"]])
            self._add_edge(last["id"], e_id, EdgeType.ENDS_WITH, session_id, [last["id"]])

    # ── Entity nodes and MENTIONS edges ──────────────────────────────────────

    def _build_entity_nodes(self, episodes: list[dict]) -> None:
        entity_to_eps: dict[str, list[str]] = defaultdict(list)

        for ep in episodes:
            text = f"{ep.get('user_text', '')} {ep.get('assistant_text', '')}"
            for entity in extract_entities(text):
                entity_to_eps[entity].append(ep["id"])

        for entity, ep_ids in entity_to_eps.items():
            ent_id = make_id("entity", entity)
            node_data = {
                "id": ent_id,
                "node_type": NodeType.ENTITY.value,
                "text": entity,
                "source_episode_ids": list(set(ep_ids)),
                "creation_method": CreationMethod.HEURISTIC.value,
                "content_hash": stable_hash(entity),
                "timestamp": 0.0,
                "metadata": {"label": entity},
            }
            self.G.add_node(ent_id, **node_data)
            self._store.upsert_node(node_data)
            for ep_id in set(ep_ids):
                if self.G.has_node(ep_id):
                    self._add_edge(ep_id, ent_id, EdgeType.MENTIONS, "", [ep_id])

    # ── Fact nodes and bi-temporal SUPERSEDES / CONTRADICTS edges ─────────────

    def _extract_facts(self, episodes: list[dict]) -> list[dict]:
        """Deterministic (subject, object, episode, time) extraction — no LLM.

        Returns one record per matched statement, ordered by event time. The
        predicate is normalised to a single "state" key per subject so that a
        later "X is now Y" supersedes an earlier "X is Z".
        """
        facts: list[dict] = []
        for ep in episodes:
            text = f"{ep.get('user_text', '')} {ep.get('assistant_text', '')}".strip()
            ts = float(ep.get("timestamp", 0.0))
            observed = ep.get("observed_at")
            observed = float(observed) if observed is not None else ts
            for m in _FACT_RE.finditer(text):
                subject = m.group("subject").strip().lower()
                obj = m.group("object").strip().rstrip(".").lower()
                # Skip degenerate captures (question stems, empty objects).
                if not subject or not obj or subject in {"what", "who", "it", "that"}:
                    continue
                facts.append(
                    {
                        "subject": subject,
                        "predicate": _STATE_PREDICATE,
                        "object": obj,
                        "episode_id": ep["id"],
                        "timestamp": ts,
                        "observed_at": observed,
                    }
                )
        # Stable order: by subject then event time, so supersession is well defined.
        facts.sort(key=lambda f: (f["subject"], f["timestamp"]))
        return facts

    def _build_fact_nodes(self, episodes: list[dict]) -> None:
        """Create FactNodes with bi-temporal validity and supersession edges.

        For each (subject, predicate) the facts form a timeline: the earlier
        fact's ``valid_to`` is closed at the next fact's ``valid_from`` and a
        ``SUPERSEDES`` edge is added new→old. When the object actually differs a
        ``CONTRADICTS`` edge is added old→new (a genuine value change, not a
        restated fact); the opposite direction keeps it from colliding with the
        SUPERSEDES edge on the DiGraph. The most recent fact per key keeps
        ``valid_to = None`` — the current state.
        """
        facts = self._extract_facts(episodes)
        by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for f in facts:
            by_key[(f["subject"], f["predicate"])].append(f)

        for (subject, predicate), timeline in by_key.items():
            fact_node_ids: list[str] = []
            for i, f in enumerate(timeline):
                is_last = i == len(timeline) - 1
                valid_from = f["timestamp"]
                valid_to = None if is_last else timeline[i + 1]["timestamp"]
                fact_id = make_id("fact", f"{subject}|{predicate}|{i}|{f['episode_id']}")
                node_data = {
                    "id": fact_id,
                    "node_type": NodeType.FACT.value,
                    "text": f"{subject} {predicate} {f['object']}",
                    "source_episode_ids": [f["episode_id"]],
                    "creation_method": CreationMethod.DETERMINISTIC.value,
                    "content_hash": stable_hash(f"{subject}{predicate}{f['object']}"),
                    "timestamp": f["timestamp"],
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "observed_at": f["observed_at"],
                    "ingested_at": f["timestamp"],
                    "metadata": {
                        "subject": subject,
                        "predicate": predicate,
                        "object": f["object"],
                        "current_state": is_last,
                    },
                }
                self.G.add_node(fact_id, **node_data)
                self._store.upsert_node(node_data)
                # Link the fact to the episode that asserted it (both directions of
                # provenance are useful: fact→episode routes to raw evidence).
                if self.G.has_node(f["episode_id"]):
                    self._add_edge(
                        f["episode_id"], fact_id, EdgeType.SUPPORTS, "",
                        [f["episode_id"]],
                    )
                fact_node_ids.append(fact_id)

                # Supersession / contradiction against the immediately prior fact.
                if i > 0:
                    prev = timeline[i - 1]
                    prev_id = fact_node_ids[i - 1]
                    self._add_edge(
                        fact_id, prev_id, EdgeType.SUPERSEDES, "",
                        [f["episode_id"], prev["episode_id"]],
                    )
                    if prev["object"] != f["object"]:
                        # CONTRADICTS is symmetric; emit it old→new so it does not
                        # collide with the SUPERSEDES edge (new→old) on this
                        # DiGraph, where one edge per (src, dst) is kept.
                        self._add_edge(
                            prev_id, fact_id, EdgeType.CONTRADICTS, "",
                            [f["episode_id"], prev["episode_id"]],
                        )

    # ── Semantic edges ────────────────────────────────────────────────────────

    def _build_semantic_edges(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return

        ep_nodes = [
            (nid, data)
            for nid, data in self.G.nodes(data=True)
            if data.get("node_type") == NodeType.EPISODE.value
        ]
        if len(ep_nodes) < 2:
            return

        ids = [nid for nid, _ in ep_nodes]
        texts = [data.get("text", "") for _, data in ep_nodes]
        model_name = self._emb_model_name or "sentence-transformers/all-MiniLM-L6-v2"
        if self._emb_model is None:
            self._emb_model = SentenceTransformer(model_name, device="cpu")

        vecs = self._emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                       show_progress_bar=False).astype(np.float32)

        # Store embeddings
        for node_id, vec in zip(ids, vecs, strict=False):
            self._store.store_embedding(node_id, vec.tolist(), model_name)

        # Build semantic edges above threshold
        sim_matrix = vecs @ vecs.T
        n = len(ids)
        for i in range(n):
            # Sort by similarity (exclude self)
            sims = sim_matrix[i].copy()
            sims[i] = -1.0
            top_j = np.argsort(sims)[::-1][: self._max_edges]
            for j in top_j:
                if sims[j] >= self._sem_thresh:
                    edge_id = make_id("sem", f"{ids[i]}{ids[j]}")
                    self._add_edge(
                        ids[i], ids[j], EdgeType.SEMANTICALLY_RELATED, "",
                        [ids[i], ids[j]], weight=float(sims[j]), edge_id=edge_id,
                    )

    # ── Topic hubs (k-means on episode embeddings) ────────────────────────────

    def _build_topic_hubs(self) -> None:
        emb_dict = self._store.load_all_embeddings()
        if not emb_dict:
            return

        ep_ids = [
            nid for nid in emb_dict
            if self.G.has_node(nid)
            and self.G.nodes[nid].get("node_type") == NodeType.EPISODE.value
        ]
        if len(ep_ids) < self._n_hubs:
            return

        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        vecs = np.array([emb_dict[nid] for nid in ep_ids], dtype=np.float32)
        n_clusters = min(self._n_hubs, len(ep_ids))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(vecs)

        # TF-IDF for top terms per cluster
        texts_by_cluster: dict[int, list[str]] = defaultdict(list)
        for ep_id, label in zip(ep_ids, labels, strict=False):
            texts_by_cluster[label].append(self.G.nodes[ep_id].get("text", ""))

        tfidf = TfidfVectorizer(max_features=50, stop_words="english")
        for cluster_id in range(n_clusters):
            docs = texts_by_cluster[cluster_id]
            if not docs:
                continue
            hub_id = make_id("hub", f"cluster_{cluster_id}")
            try:
                tfidf.fit(docs)
                vocab = tfidf.get_feature_names_out().tolist()[:10]
            except Exception:
                vocab = []

            node_data = {
                "id": hub_id,
                "node_type": NodeType.TOPIC_HUB.value,
                "text": f"Topic Hub {cluster_id}: {', '.join(vocab[:5])}",
                "source_episode_ids": [
                    ep_ids[i] for i, lbl in enumerate(labels) if lbl == cluster_id
                ],
                "creation_method": CreationMethod.HEURISTIC.value,
                "content_hash": stable_hash(f"hub_{cluster_id}"),
                "timestamp": 0.0,
                "metadata": {"cluster_id": cluster_id, "top_terms": vocab},
            }
            self.G.add_node(hub_id, **node_data)
            self._store.upsert_node(node_data)

            for ep_id, label in zip(ep_ids, labels, strict=False):
                if label == cluster_id:
                    self._add_edge(ep_id, hub_id, EdgeType.BELONGS_TO_TOPIC, "", [ep_id])

    # ── Community detection and README ────────────────────────────────────────

    def _build_communities(self) -> None:
        undirected = self.G.to_undirected()
        # Only consider episode nodes for community detection
        ep_nodes_set = {
            n for n, d in self.G.nodes(data=True)
            if d.get("node_type") == NodeType.EPISODE.value
        }
        subgraph = undirected.subgraph(ep_nodes_set)

        try:
            if self._community_alg == "louvain":
                from community import best_partition  # python-louvain
                partition = best_partition(subgraph)
                communities_map: dict[int, list[str]] = defaultdict(list)
                for node, comm_id in partition.items():
                    communities_map[comm_id].append(node)
                communities = list(communities_map.values())
            else:
                from networkx.algorithms.community import greedy_modularity_communities
                result = greedy_modularity_communities(subgraph)
                communities = [list(c) for c in result]
        except Exception:
            # Fallback: each connected component is a community
            communities = [list(c) for c in nx.connected_components(subgraph)]

        for comm_idx, member_ids in enumerate(communities):
            comm_id = make_id("community", f"comm_{comm_idx}")
            readme = self._build_readme(member_ids, comm_idx)
            node_data = {
                "id": comm_id,
                "node_type": NodeType.COMMUNITY.value,
                "text": readme["topic"],
                "source_episode_ids": readme["source_episode_ids"],
                "creation_method": CreationMethod.DETERMINISTIC.value,
                "content_hash": stable_hash(f"community_{comm_idx}"),
                "timestamp": 0.0,
                "metadata": {
                    "readme": readme,
                    "member_ids": member_ids,
                },
            }
            self.G.add_node(comm_id, **node_data)
            self._store.upsert_node(node_data)

            for ep_id in member_ids:
                if self.G.has_node(ep_id):
                    self._add_edge(ep_id, comm_id, EdgeType.BELONGS_TO_COMMUNITY, "", [ep_id])

    def _build_readme(self, member_ids: list[str], comm_idx: int) -> dict:
        """Extractive community README — no LLM."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [
            self.G.nodes[nid].get("text", "")
            for nid in member_ids
            if self.G.has_node(nid)
        ]
        entities: list[str] = []
        for nid in member_ids:
            if self.G.has_node(nid):
                entities.extend(extract_entities(self.G.nodes[nid].get("text", "")))

        top_entities = list({e for e in entities})[:10]

        top_terms: list[str] = []
        try:
            tfidf = TfidfVectorizer(max_features=30, stop_words="english")
            tfidf.fit(docs or [""])
            top_terms = tfidf.get_feature_names_out().tolist()[:20]
        except Exception:
            pass

        ts_vals = [
            self.G.nodes[nid].get("timestamp", 0.0)
            for nid in member_ids
            if self.G.has_node(nid)
        ]
        time_range = (min(ts_vals), max(ts_vals)) if ts_vals else None

        return {
            "topic": f"Community {comm_idx}: {', '.join(top_terms[:3]) or 'general'}",
            "top_terms": top_terms,
            "top_entities": top_entities,
            "representative_episodes": member_ids[:3],
            "time_range": time_range,
            "node_count": len(member_ids),
            "edge_count": self.G.subgraph(member_ids).number_of_edges(),
            "source_episode_ids": member_ids,
            "creation_method": CreationMethod.DETERMINISTIC.value,
        }

    # ── Hierarchical summary tree (Phase 6, RAPTOR-style, extractive) ─────────

    def _build_summary_tree(self, episodes: list[dict]) -> None:
        """Condense episodes into a navigable extractive summary tree.

        Level 1 groups episodes by session and builds one SUMMARY node per session
        from the top TF-IDF sentences of its turns. Higher levels group the
        previous level's summaries into blocks of ``branching_factor`` and
        summarise those, up to ``max_levels`` (or until a single root remains).

        No LLM is used: summaries are purely extractive, so they are deterministic
        and cheap. Summaries only route — the retriever surfaces the raw episodes
        beneath a matched summary, never the summary text itself.
        """
        by_session: dict[str, list[dict]] = defaultdict(list)
        for ep in episodes:
            by_session[ep.get("session_id", "")].append(ep)

        # ── Level 1: one summary per session ──────────────────────────────────
        level_nodes: list[str] = []
        for session_id, sess_eps in by_session.items():
            sess_eps.sort(key=lambda e: (e.get("turn_index", 0), e.get("timestamp", 0.0)))
            child_ids = [ep["id"] for ep in sess_eps]
            texts = [
                f"{ep.get('user_text', '')} {ep.get('assistant_text', '')}".strip()
                for ep in sess_eps
            ]
            summary_text = self._extractive_summary(texts)
            ts_vals = [float(ep.get("timestamp", 0.0)) for ep in sess_eps]
            sum_id = self._add_summary_node(
                key=f"L1_{session_id}",
                level=1,
                text=summary_text,
                child_ids=child_ids,
                source_episode_ids=child_ids,
                session_id=session_id,
                timestamp=min(ts_vals) if ts_vals else 0.0,
            )
            level_nodes.append(sum_id)

        # ── Higher levels: summarise blocks of the previous level ─────────────
        level = 1
        while level < self._summary_max_levels and len(level_nodes) > 1:
            level += 1
            next_level: list[str] = []
            for block_idx in range(0, len(level_nodes), self._summary_branching):
                block = level_nodes[block_idx : block_idx + self._summary_branching]
                if not block:
                    continue
                child_texts = [self.G.nodes[c].get("text", "") for c in block]
                summary_text = self._extractive_summary(child_texts)
                # Flatten leaf episodes reachable beneath this block.
                leaves: list[str] = []
                for c in block:
                    leaves.extend(self.G.nodes[c].get("source_episode_ids", []))
                sum_id = self._add_summary_node(
                    key=f"L{level}_{block_idx}",
                    level=level,
                    text=summary_text,
                    child_ids=block,
                    source_episode_ids=list(dict.fromkeys(leaves)),
                    session_id="",
                    timestamp=0.0,
                )
                next_level.append(sum_id)
            level_nodes = next_level

    def _add_summary_node(
        self,
        key: str,
        level: int,
        text: str,
        child_ids: list[str],
        source_episode_ids: list[str],
        session_id: str,
        timestamp: float,
    ) -> str:
        """Create one SUMMARY node and wire it to its children (both directions)."""
        sum_id = make_id("summary", key)
        node_data = {
            "id": sum_id,
            "node_type": NodeType.SUMMARY.value,
            "text": text,
            "session_id": session_id,
            "timestamp": timestamp,
            "source_episode_ids": source_episode_ids,
            "creation_method": CreationMethod.DETERMINISTIC.value,
            "content_hash": stable_hash(text),
            "metadata": {"level": level, "child_ids": child_ids},
        }
        self.G.add_node(sum_id, **node_data)
        self._store.upsert_node(node_data)
        for child in child_ids:
            if not self.G.has_node(child):
                continue
            # summary → child (drill-down) and child → summary (roll-up).
            self._add_edge(sum_id, child, EdgeType.SUMMARIZES, session_id, [child])
            self._add_edge(child, sum_id, EdgeType.PARENT_SUMMARY, session_id, [child])
        return sum_id

    def _extractive_summary(self, texts: list[str]) -> str:
        """Pick the most representative sentences from ``texts`` via TF-IDF.

        Each sentence is scored by the sum of its TF-IDF term weights; the top
        ``summary_max_sentences`` are returned in their original order so the
        digest reads coherently. Falls back to the leading sentences when TF-IDF
        cannot be fit (e.g. a single short document).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        joined = " ".join(t for t in texts if t).strip()
        if not joined:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if s.strip()]
        if len(sentences) <= self._summary_max_sentences:
            return " ".join(sentences)
        try:
            tfidf = TfidfVectorizer(stop_words="english")
            matrix = tfidf.fit_transform(sentences)
            sent_scores = matrix.sum(axis=1)
            scores = [float(sent_scores[i, 0]) for i in range(len(sentences))]
        except ValueError:
            return " ".join(sentences[: self._summary_max_sentences])
        top_idx = sorted(
            range(len(sentences)), key=lambda i: scores[i], reverse=True
        )[: self._summary_max_sentences]
        top_idx.sort()
        return " ".join(sentences[i] for i in top_idx)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_edge(
        self,
        src: str,
        dst: str,
        etype: EdgeType,
        session_id: str,
        ep_ids: list[str],
        weight: float = 1.0,
        edge_id: str | None = None,
    ) -> None:
        eid = edge_id or make_id("edge", f"{src}{etype.value}{dst}")
        edge_data = {
            "id": eid,
            "src_id": src,
            "dst_id": dst,
            "edge_type": etype.value,
            "weight": weight,
            "session_id": session_id,
            "timestamp": time.time(),
            "source_episode_ids": ep_ids,
            "creation_method": CreationMethod.DETERMINISTIC.value,
            "content_hash": stable_hash(f"{src}{etype.value}{dst}"),
        }
        self.G.add_edge(src, dst, **edge_data)
        self._store.upsert_edge(edge_data)
