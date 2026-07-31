"""SQLite-backed storage for episodes, nodes, edges, and run registry."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL DEFAULT '',
    turn_index  INTEGER NOT NULL DEFAULT 0,
    user_text   TEXT NOT NULL DEFAULT '',
    assistant_text TEXT NOT NULL DEFAULT '',
    timestamp   REAL NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    id UNINDEXED,
    text_content,
    tokenize='porter ascii'
);

CREATE TABLE IF NOT EXISTS nodes (
    id           TEXT PRIMARY KEY,
    node_type    TEXT NOT NULL,
    text         TEXT NOT NULL DEFAULT '',
    session_id   TEXT NOT NULL DEFAULT '',
    timestamp    REAL NOT NULL,
    confidence   REAL NOT NULL DEFAULT 1.0,
    version      INTEGER NOT NULL DEFAULT 1,
    valid_from   REAL,
    valid_to     REAL,
    content_hash TEXT NOT NULL DEFAULT '',
    creation_method TEXT NOT NULL DEFAULT 'deterministic',
    metadata     TEXT NOT NULL DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    id UNINDEXED,
    text,
    tokenize='porter ascii'
);

CREATE TABLE IF NOT EXISTS node_episodes (
    node_id    TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    PRIMARY KEY (node_id, episode_id)
);

CREATE TABLE IF NOT EXISTS edges (
    id          TEXT PRIMARY KEY,
    src_id      TEXT NOT NULL,
    dst_id      TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    session_id  TEXT NOT NULL DEFAULT '',
    timestamp   REAL NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    version     INTEGER NOT NULL DEFAULT 1,
    valid_from  REAL,
    valid_to    REAL,
    content_hash TEXT NOT NULL DEFAULT '',
    creation_method TEXT NOT NULL DEFAULT 'deterministic',
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edge_episodes (
    edge_id    TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    PRIMARY KEY (edge_id, episode_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    node_id    TEXT PRIMARY KEY,
    vector     BLOB NOT NULL,
    model_name TEXT NOT NULL DEFAULT '',
    dim        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    dataset      TEXT NOT NULL DEFAULT '',
    method       TEXT NOT NULL DEFAULT '',
    config_json  TEXT NOT NULL DEFAULT '{}',
    env_json     TEXT NOT NULL DEFAULT '{}',
    started_at   REAL NOT NULL,
    finished_at  REAL,
    status       TEXT NOT NULL DEFAULT 'running',
    metrics_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
"""


class Store:
    """Thin SQLite wrapper for all Memory City data."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._path = str(db_path)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(DDL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ── Episodes ────────────────────────────────────────────────────────────

    def upsert_episode(
        self,
        ep_id: str,
        session_id: str,
        turn_index: int,
        user_text: str,
        assistant_text: str,
        timestamp: float,
        metadata: dict | None = None,
    ) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO episodes
                   (id, session_id, turn_index, user_text, assistant_text, timestamp, metadata)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    ep_id, session_id, turn_index, user_text, assistant_text,
                    timestamp, json.dumps(metadata or {}),
                ),
            )
            full_text = f"{user_text} {assistant_text}"
            c.execute("DELETE FROM episodes_fts WHERE id=?", (ep_id,))
            c.execute("INSERT INTO episodes_fts(id, text_content) VALUES (?,?)",
                      (ep_id, full_text))

    def get_episode(self, ep_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM episodes WHERE id=?", (ep_id,)).fetchone()
        return dict(row) if row else None

    def all_episodes(self) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp ASC"
        ).fetchall()]

    def search_episodes_fts(self, query: str, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """SELECT e.* FROM episodes_fts f
               JOIN episodes e ON e.id = f.id
               WHERE episodes_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Nodes ────────────────────────────────────────────────────────────────

    def upsert_node(self, node: dict) -> None:
        meta = node.get("metadata", {})
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO nodes
                   (id, node_type, text, session_id, timestamp, confidence, version,
                    valid_from, valid_to, content_hash, creation_method, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    node["id"], node["node_type"], node.get("text", ""),
                    node.get("session_id", ""), node.get("timestamp", time.time()),
                    node.get("confidence", 1.0), node.get("version", 1),
                    node.get("valid_from"), node.get("valid_to"),
                    node.get("content_hash", ""), node.get("creation_method", "deterministic"),
                    json.dumps(meta),
                ),
            )
            c.execute("DELETE FROM nodes_fts WHERE id=?", (node["id"],))
            c.execute("INSERT INTO nodes_fts(id, text) VALUES (?,?)",
                      (node["id"], node.get("text", "")))
            for ep_id in node.get("source_episode_ids", []):
                c.execute(
                    "INSERT OR IGNORE INTO node_episodes(node_id, episode_id) VALUES (?,?)",
                    (node["id"], ep_id),
                )

    def get_node(self, node_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["source_episode_ids"] = [
            r[0] for r in self._conn.execute(
                "SELECT episode_id FROM node_episodes WHERE node_id=?", (node_id,)
            ).fetchall()
        ]
        return d

    def all_nodes(self, node_type: str | None = None) -> list[dict]:
        if node_type:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE node_type=? ORDER BY timestamp ASC", (node_type,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes ORDER BY timestamp ASC"
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["source_episode_ids"] = [
                r[0] for r in self._conn.execute(
                    "SELECT episode_id FROM node_episodes WHERE node_id=?", (d["id"],)
                ).fetchall()
            ]
            result.append(d)
        return result

    def search_nodes_fts(self, query: str, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """SELECT n.* FROM nodes_fts f
               JOIN nodes n ON n.id = f.id
               WHERE nodes_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Edges ────────────────────────────────────────────────────────────────

    def upsert_edge(self, edge: dict) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO edges
                   (id, src_id, dst_id, edge_type, weight, session_id, timestamp,
                    confidence, version, valid_from, valid_to, content_hash,
                    creation_method, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    edge["id"], edge["src_id"], edge["dst_id"], edge["edge_type"],
                    edge.get("weight", 1.0), edge.get("session_id", ""),
                    edge.get("timestamp", time.time()), edge.get("confidence", 1.0),
                    edge.get("version", 1), edge.get("valid_from"), edge.get("valid_to"),
                    edge.get("content_hash", ""), edge.get("creation_method", "deterministic"),
                    json.dumps(edge.get("metadata", {})),
                ),
            )
            for ep_id in edge.get("source_episode_ids", []):
                c.execute(
                    "INSERT OR IGNORE INTO edge_episodes(edge_id, episode_id) VALUES (?,?)",
                    (edge["id"], ep_id),
                )

    def get_edges(self, node_id: str | None = None, edge_type: str | None = None) -> list[dict]:
        clauses, params = [], []
        if node_id:
            clauses.append("(src_id=? OR dst_id=?)")
            params += [node_id, node_id]
        if edge_type:
            clauses.append("edge_type=?")
            params.append(edge_type)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM edges {where} ORDER BY timestamp ASC", params
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Embeddings ───────────────────────────────────────────────────────────

    def store_embedding(self, node_id: str, vector: list[float], model_name: str = "") -> None:
        import struct
        blob = struct.pack(f"{len(vector)}f", *vector)
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO embeddings(node_id, vector, model_name, dim) VALUES (?,?,?,?)",
                (node_id, blob, model_name, len(vector)),
            )

    def load_embedding(self, node_id: str) -> list[float] | None:
        import struct
        row = self._conn.execute(
            "SELECT vector, dim FROM embeddings WHERE node_id=?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return list(struct.unpack(f"{row['dim']}f", row["vector"]))

    def load_all_embeddings(self) -> dict[str, list[float]]:
        import struct
        rows = self._conn.execute("SELECT node_id, vector, dim FROM embeddings").fetchall()
        return {
            r["node_id"]: list(struct.unpack(f"{r['dim']}f", r["vector"]))
            for r in rows
        }

    # ── Runs ─────────────────────────────────────────────────────────────────

    def create_run(
        self, run_id: str, dataset: str, method: str,
        config: dict, env: dict, started_at: float,
    ) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR IGNORE INTO runs
                   (run_id, dataset, method, config_json, env_json, started_at, status)
                   VALUES (?,?,?,?,?,?,'running')""",
                (run_id, dataset, method, json.dumps(config), json.dumps(env), started_at),
            )

    def finish_run(self, run_id: str, metrics: dict) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE runs SET finished_at=?, status='done', metrics_json=? WHERE run_id=?",
                (time.time(), json.dumps(metrics), run_id),
            )

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC"
        ).fetchall()]
