"""Unit tests for Phase 4 contextual indexing.

A context-poor turn ("Yes, let's use that one.") is unretrievable by BM25 on its
own. With deterministic contextual indexing the previous turn's text is prefixed
*before indexing*, so the turn becomes findable — while the raw episode text (and
therefore RetrievedItem.text and any citation) is preserved untouched.

Fully deterministic: no model, no network.
"""

from __future__ import annotations

from memcity.datasets.corpus import apply_contextual_index
from memcity.retrieval.baselines import BM25Retriever, _index_text, _item_text


def _mk(eid, text, session, turn, ts):
    return {
        "id": eid, "node_type": "episode",
        "text": text, "user_text": text, "assistant_text": "",
        "session_id": session, "turn_index": turn, "timestamp": ts,
        "source_episode_ids": [eid],
    }


# Session s0: turn 0 names "Postgres", turn 1 refers to it obliquely. Several
# distractor turns in other sessions keep "postgres" a rare term so BM25's IDF is
# well-behaved (a term present in every document yields a degenerate negative IDF).
CORPUS = [
    _mk("t0", "Should we use Postgres for the analytics database?", "s0", 0, 100.0),
    _mk("t1", "Yes, let's use that one.", "s0", 1, 200.0),
    _mk("d0", "The weather in Paris was pleasant last spring.", "s1", 0, 300.0),
    _mk("d1", "I enjoy hiking on the weekend with friends.", "s2", 0, 400.0),
    _mk("d2", "The quarterly report is due next Friday afternoon.", "s3", 0, 500.0),
]


def _fresh():
    import copy
    return copy.deepcopy(CORPUS)


def test_raw_mode_is_noop():
    corpus = _fresh()
    out = apply_contextual_index(corpus, mode="raw")
    assert all("indexed_text" not in it for it in out)


def test_deterministic_prefix_added_but_raw_preserved():
    corpus = apply_contextual_index(_fresh(), mode="deterministic")
    t1 = next(it for it in corpus if it["id"] == "t1")
    # Prefix pulls the previous turn's content ("Postgres") into the index text.
    assert "postgres" in t1["indexed_text"].lower()
    # Raw fields are untouched — provenance and citations stay clean.
    assert t1["text"] == "Yes, let's use that one."
    assert t1["user_text"] == "Yes, let's use that one."
    # First turn of a session has no predecessor, so no context snippet leaks in.
    t0 = next(it for it in corpus if it["id"] == "t0")
    assert "context:" not in t0["indexed_text"].lower()


def test_index_text_prefers_indexed_but_item_text_stays_raw():
    corpus = apply_contextual_index(_fresh(), mode="deterministic")
    t1 = next(it for it in corpus if it["id"] == "t1")
    # Indexed text carries the prefix; the raw item text never does.
    assert _index_text(t1) == t1["indexed_text"]
    assert "postgres" not in _item_text(t1).lower()
    assert "session" not in _item_text(t1).lower()


def test_token_budget_caps_prefix():
    corpus = _fresh()
    apply_contextual_index(corpus, mode="deterministic", max_context_tokens=3)
    t1 = next(it for it in corpus if it["id"] == "t1")
    prefix = t1["indexed_text"].replace(t1["text"], "").strip()
    assert len(prefix.split()) <= 3


def test_bm25_finds_context_poor_turn_only_with_context():
    # Baseline: query about Postgres should NOT surface t1 in raw mode — it has no
    # lexical overlap with the query on its own.
    raw = BM25Retriever()
    raw.build(_fresh())
    raw_scores = {it.id: it.score for it in raw.query("Postgres analytics database", top_k=5).items}
    assert raw_scores.get("t1", 0.0) == 0.0

    # With contextual indexing t1 inherits "Postgres" from its predecessor and
    # becomes retrievable.
    ctx_corpus = apply_contextual_index(_fresh(), mode="deterministic")
    ctx = BM25Retriever()
    ctx.build(ctx_corpus)
    ctx_res = ctx.query("Postgres analytics database", top_k=5)
    ctx_scores = {it.id: it.score for it in ctx_res.items}
    assert ctx_scores.get("t1", 0.0) > 0.0
    # And the returned text is still the raw episode, not the prefixed index text.
    t1_item = next(it for it in ctx_res.items if it.id == "t1")
    assert "postgres" not in t1_item.text.lower()


def test_scope_isolation_prefix_does_not_cross_sessions():
    # A turn's context prefix only ever draws from its own session's prior turn.
    corpus = apply_contextual_index(_fresh(), mode="deterministic")
    d0 = next(it for it in corpus if it["id"] == "d0")
    # d0 is the first (and only) turn of session s1: no cross-session leakage.
    assert "postgres" not in d0["indexed_text"].lower()
