"""Deterministic evidence compression for the end-to-end reader (Phase 7).

Retrieval returns whole episodes, but a small reader has a tight context window
and pays (latency + tokens) for every word it reads. Most of an episode is
filler; the answer usually hinges on a few high-signal sentences. This module
compresses each retrieved passage *before* it reaches the reader by keeping only
the sentences most likely to carry the answer, using cheap deterministic
signals — no LLM, so it never contaminates retrieval metrics and stays
reproducible.

Signals a sentence is worth keeping (any one qualifies it):

* overlaps the query terms (topical relevance);
* contains a named entity / proper noun (who/what);
* contains a number, date, time, or money amount (when/how much);
* contains a negation ("not", "no", "never", "didn't") — negations flip meaning
  and are disproportionately answer-bearing, so they are never dropped.

The compressor preserves the passage's citation id and original ordering, so the
reader's ``evidence_ids`` stay valid and grounding is unaffected. It reports how
many tokens it removed so the benchmark can quantify the accuracy/coverage vs.
token/latency trade-off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from memcity.utils.helpers import tokenize

# Sentence splitter: break on ., !, ? followed by whitespace. Conservative so an
# abbreviation rarely splits a sentence in two (over-keeping is safe here).
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A number, date, time, percentage, or money amount — strong "when/how much"
# signal. Matches 2024, 3.5, 10%, $20, 12/05, 7:30.
_NUMERIC_RE = re.compile(r"\b\d[\d,.:/%$-]*\b|\$\d")

# Proper-noun / entity heuristic: a capitalised word not at sentence start, or a
# multi-word capitalised span. Deterministic and dependency-free.
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")

# Negations flip a sentence's meaning; always answer-relevant.
_NEGATIONS = frozenset(
    {
        "not", "no", "never", "none", "cannot", "can't", "don't", "doesn't",
        "didn't", "won't", "wouldn't", "isn't", "aren't", "wasn't", "weren't",
        "without", "nor", "neither",
    }
)


@dataclass
class CompressionResult:
    """One passage's compressed text plus its token accounting."""

    id: str
    text: str
    tokens_before: int
    tokens_after: int

    @property
    def tokens_removed(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


@dataclass
class CompressionStats:
    """Aggregate token accounting across a set of compressed passages."""

    passages: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    kept_sentences: int = 0
    total_sentences: int = 0
    per_passage: list[CompressionResult] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Fraction of tokens *removed* (0 = no compression, →1 = maximal)."""
        if self.tokens_before <= 0:
            return 0.0
        return 1.0 - (self.tokens_after / self.tokens_before)


class EvidenceCompressor:
    """Extractive, deterministic per-passage compressor.

    Args:
        max_sentences_per_passage: hard cap on sentences kept per passage. The
            most relevant sentences (by signal count, then original order) are
            kept up to this cap.
        keep_numeric: keep any sentence containing a number/date/money amount.
        keep_entities: keep any sentence containing a proper-noun entity.
        keep_negations: keep any sentence containing a negation (recommended:
            negations flip meaning and are cheap to keep).
        min_query_overlap: minimum count of query-term overlaps for a sentence to
            qualify on relevance alone.
    """

    def __init__(
        self,
        *,
        max_sentences_per_passage: int = 3,
        keep_numeric: bool = True,
        keep_entities: bool = True,
        keep_negations: bool = True,
        min_query_overlap: int = 1,
    ) -> None:
        self._max_sents = max(1, max_sentences_per_passage)
        self._keep_numeric = keep_numeric
        self._keep_entities = keep_entities
        self._keep_negations = keep_negations
        self._min_overlap = max(1, min_query_overlap)

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]

    def _sentence_signal(self, sentence: str, query_tokens: set[str]) -> tuple[bool, int]:
        """Return ``(keep, score)`` for one sentence.

        ``keep`` is True when the sentence trips any retention rule. ``score``
        ranks kept sentences so the strongest survive the per-passage cap: it
        sums query overlap, entity presence, numeric presence, and negation
        presence, so a sentence hitting several signals outranks one hitting one.
        """
        sent_tokens = set(tokenize(sentence))
        overlap = len(sent_tokens & query_tokens)
        has_numeric = bool(self._keep_numeric and _NUMERIC_RE.search(sentence))
        has_entity = bool(self._keep_entities and _ENTITY_RE.search(sentence))
        has_negation = bool(self._keep_negations and (sent_tokens & _NEGATIONS))

        score = overlap
        if has_entity:
            score += 1
        if has_numeric:
            score += 1
        if has_negation:
            score += 1

        keep = (
            overlap >= self._min_overlap
            or has_numeric
            or has_entity
            or has_negation
        )
        return keep, score

    def compress_passage(self, text: str, query: str) -> tuple[str, int, int]:
        """Compress one passage. Returns ``(compressed_text, n_kept, n_total)``.

        Sentences that trip a retention rule are kept, ranked by signal score,
        capped at ``max_sentences_per_passage``, then re-emitted in their original
        order so the passage still reads coherently. When nothing qualifies (a
        context-poor turn), the leading sentence is kept so a passage is never
        emptied — dropping it entirely would silently reduce recall.
        """
        sentences = self._split_sentences(text)
        n_total = len(sentences)
        if n_total <= 1:
            return text.strip(), n_total, n_total

        query_tokens = set(tokenize(query))
        scored: list[tuple[int, int]] = []  # (original_index, score)
        for idx, sent in enumerate(sentences):
            keep, score = self._sentence_signal(sent, query_tokens)
            if keep:
                scored.append((idx, score))

        if not scored:
            # Never emit an empty passage; fall back to the first sentence.
            return sentences[0], 1, n_total

        # Keep the strongest sentences (by score, then earliest) up to the cap.
        scored.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        kept_indices = sorted(idx for idx, _ in scored[: self._max_sents])
        compressed = " ".join(sentences[i] for i in kept_indices)
        return compressed, len(kept_indices), n_total

    def compress_evidence(
        self, evidence: list[dict], query: str
    ) -> tuple[list[dict], CompressionStats]:
        """Compress every passage, preserving ids/order and reporting token stats.

        The returned evidence dicts are shallow copies with a compressed ``text``;
        every other field (notably ``id`` and ``source_episode_ids``) is carried
        through unchanged so citation ids stay valid.
        """
        stats = CompressionStats()
        out: list[dict] = []
        for ep in evidence:
            original = ep.get("text", "")
            tokens_before = len(tokenize(original))
            compressed, n_kept, n_total = self.compress_passage(original, query)
            tokens_after = len(tokenize(compressed))

            new_ep = dict(ep)
            new_ep["text"] = compressed
            out.append(new_ep)

            stats.passages += 1
            stats.tokens_before += tokens_before
            stats.tokens_after += tokens_after
            stats.kept_sentences += n_kept
            stats.total_sentences += n_total
            stats.per_passage.append(
                CompressionResult(
                    id=ep.get("id", ""),
                    text=compressed,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                )
            )
        return out, stats
