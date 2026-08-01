"""Scope-aware corpus construction.

Each QASample declares a ``corpus_scope_id`` identifying the retrieval scope it
belongs to. LongMemEval assigns one scope per question (each has its own
haystack). LoCoMo assigns one scope per conversation (all questions of a
conversation share that conversation's history). A retriever must only ever see
episodes from the querying sample's own scope — otherwise a query can retrieve
evidence from an unrelated conversation, which silently inflates or deflates
recall depending on the method.

An empty ``corpus_scope_id`` falls back to a single shared global scope. This
preserves the synthetic dataset's intended behaviour (one shared knowledge base)
and keeps legacy callers working.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from memcity.datasets.protocol import QASample

GLOBAL_SCOPE = "__global__"

ContextMode = Literal["raw", "deterministic", "llm"]


def scope_of(sample: QASample) -> str:
    """Return the canonical scope id for a sample, falling back to global."""
    return sample.corpus_scope_id or GLOBAL_SCOPE


def episode_to_corpus_item(ep) -> dict:
    """Convert one EpisodeTurn into a retriever corpus dict."""
    return {
        "id": ep.episode_id,
        "node_type": "episode",
        "text": f"{ep.user_text} {ep.assistant_text}".strip(),
        "user_text": ep.user_text,
        "assistant_text": ep.assistant_text,
        "session_id": ep.session_id,
        "turn_index": ep.turn_index,
        "timestamp": ep.timestamp,
        "source_episode_ids": [ep.episode_id],
    }


# ── Contextual indexing (Phase 4) ────────────────────────────────────────────

# Deterministic prefixes stay small so they help disambiguate context-poor turns
# without swamping the raw signal. ~100 tokens ≈ a few short clauses.
_MAX_PREV_CHARS = 160


def _fmt_ts(ts: float) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return ""


def _deterministic_prefix(item: dict, prev: dict | None, max_tokens: int) -> str:
    """Cheap, model-free context prefix built from local session structure.

    Sources: session id, turn timestamp, and a snippet of the previous turn in
    the same session. This is enough to resolve anaphora like "that one" back to
    the thing named a turn earlier, at zero model cost and fully deterministically.
    """
    bits: list[str] = []
    session = item.get("session_id") or ""
    if session:
        bits.append(f"[session {session}]")
    day = _fmt_ts(item.get("timestamp", 0.0))
    if day:
        bits.append(f"[{day}]")
    if prev is not None:
        prev_text = f"{prev.get('user_text', '')} {prev.get('assistant_text', '')}".strip()
        if prev_text:
            snippet = prev_text[:_MAX_PREV_CHARS].strip()
            bits.append(f"context: {snippet}")
    prefix = " ".join(bits).strip()
    if not prefix:
        return ""
    # Token budget guard (whitespace tokenisation is a fine proxy here).
    tokens = prefix.split()
    if len(tokens) > max_tokens:
        prefix = " ".join(tokens[:max_tokens])
    return prefix


def apply_contextual_index(
    corpus: list[dict],
    mode: ContextMode = "raw",
    max_context_tokens: int = 100,
) -> list[dict]:
    """Populate ``indexed_text`` on each corpus item according to ``mode``.

    ``indexed_text`` = context_prefix + raw text. It is what the index (BM25,
    embeddings, semantic edges) consumes; the raw ``text``/``user_text``/
    ``assistant_text`` fields are left untouched so retrieved evidence and
    citations always reflect the original episode. In ``raw`` mode this is a
    no-op. ``llm`` degrades to ``deterministic`` here (a summariser hook can be
    wired in later without changing the index contract).
    """
    if mode == "raw":
        return corpus

    by_session: dict[str, list[dict]] = defaultdict(list)
    for item in corpus:
        by_session[item.get("session_id", "")].append(item)

    prev_by_id: dict[str, dict | None] = {}
    for sess_items in by_session.values():
        sess_items.sort(key=lambda it: (it.get("turn_index", 0), it.get("timestamp", 0.0)))
        prev: dict | None = None
        for it in sess_items:
            prev_by_id[it["id"]] = prev
            prev = it

    for item in corpus:
        raw = item.get("text", "")
        prefix = _deterministic_prefix(item, prev_by_id.get(item["id"]), max_context_tokens)
        item["indexed_text"] = f"{prefix} {raw}".strip() if prefix else raw
    return corpus


def build_scope_corpus(samples: list[QASample]) -> dict[str, list[dict]]:
    """Group samples by scope and build one deduplicated corpus per scope.

    Deduplication is keyed by ``episode_id`` and is scoped: the same episode id
    appearing in two different scopes yields one corpus entry per scope. Within a
    scope the first occurrence of an episode id wins.
    """
    corpora: dict[str, list[dict]] = {}
    seen: dict[str, set[str]] = {}
    for sample in samples:
        scope = scope_of(sample)
        scope_seen = seen.setdefault(scope, set())
        scope_corpus = corpora.setdefault(scope, [])
        for ep in sample.history:
            if ep.episode_id in scope_seen:
                continue
            scope_seen.add(ep.episode_id)
            scope_corpus.append(episode_to_corpus_item(ep))
    return corpora


def group_samples_by_scope(samples: list[QASample]) -> dict[str, list[QASample]]:
    """Return samples grouped by canonical scope id, preserving order."""
    grouped: dict[str, list[QASample]] = {}
    for sample in samples:
        grouped.setdefault(scope_of(sample), []).append(sample)
    return grouped
