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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memcity.datasets.protocol import QASample

GLOBAL_SCOPE = "__global__"


def scope_of(sample: "QASample") -> str:
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
        "timestamp": ep.timestamp,
        "source_episode_ids": [ep.episode_id],
    }


def build_scope_corpus(samples: list["QASample"]) -> dict[str, list[dict]]:
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


def group_samples_by_scope(samples: list["QASample"]) -> dict[str, list["QASample"]]:
    """Return samples grouped by canonical scope id, preserving order."""
    grouped: dict[str, list["QASample"]] = {}
    for sample in samples:
        grouped.setdefault(scope_of(sample), []).append(sample)
    return grouped
