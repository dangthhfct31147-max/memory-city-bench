"""Heuristic query coordinator — no LLM, fully traceable."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Route(str, Enum):
    LEXICAL = "LEXICAL"
    VECTOR = "VECTOR"
    TEMPORAL = "TEMPORAL"
    LOCAL_GRAPH = "LOCAL_GRAPH"
    COMMUNITY_GLOBAL = "COMMUNITY_GLOBAL"
    HYBRID = "HYBRID"


TEMPORAL_PATTERNS = re.compile(
    r"\b(before|after|latest|previously|recently|last|first|when|"
    r"originally|updated|changed|current|now|today|yesterday|"
    r"\d{4}|\d{1,2}/\d{1,2})\b",
    re.I,
)

GLOBAL_PATTERNS = re.compile(
    r"\b(themes?|topics?|summary|overview|pattern|generally|overall|"
    r"all|most|main|common|history|everything|broadly)\b",
    re.I,
)

ENTITY_RE = re.compile(r"\b[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?\b")

# Sentence-initial question/auxiliary words that ENTITY_RE would otherwise
# misread as proper-noun entities (e.g. "What did Caroline buy?" → "What").
QUESTION_WORDS = frozenset(
    {
        "What", "Who", "Where", "When", "Why", "How", "Which",
        "Can", "Is", "Are", "Do", "Does", "Did", "Was", "Were",
        "Tell", "Amid", "Any", "Could", "Would", "Should", "Will",
    }
)


@dataclass
class CoordinatorDecision:
    routes: list[Route] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "routes": [r.value for r in self.routes],
            "features": self.features,
            "weights": self.weights,
            "confidence": self.confidence,
        }


# Temporal query intent (Phase 3). A temporal query is parsed into a *constraint*
# on event time, not merely a "sort oldest/newest" hint:
#   "currently" / "now"           → CURRENT  (valid_to is None)
#   "originally" / "first"        → ORIGINAL (earliest valid fact)
#   "before <ref>"                → BEFORE   (valid at < ref)
#   "after <ref>"                 → AFTER    (valid at > ref)
# ``kind == NONE`` means the query carries no resolvable temporal constraint.
_CURRENT_RE = re.compile(r"\b(current(?:ly)?|now|today|latest|these days)\b", re.I)
_ORIGINAL_RE = re.compile(r"\b(originally|first|initially|at first|used to|before)\b", re.I)
_BEFORE_RE = re.compile(r"\bbefore\b", re.I)
_AFTER_RE = re.compile(r"\bafter\b", re.I)


@dataclass
class TemporalConstraint:
    kind: str = "none"  # none | current | original | before | after
    # Reference event-time boundary for before/after when one can be resolved
    # from the query (e.g. an explicit year). None when unresolved — the retriever
    # then relies on fact ordering rather than an absolute timestamp.
    reference: float | None = None

    @property
    def is_active(self) -> bool:
        return self.kind != "none"


def parse_temporal_constraint(query: str) -> TemporalConstraint:
    """Parse a query into an event-time constraint (deterministic, no LLM).

    Kept intentionally conservative: only clear cues produce a constraint so a
    non-temporal query is never forced onto the temporal path.
    """
    q = query.strip()
    # "current" wins over "before/original" when both appear ("current vs original").
    if _CURRENT_RE.search(q) and not _BEFORE_RE.search(q):
        return TemporalConstraint(kind="current")
    if _BEFORE_RE.search(q):
        return TemporalConstraint(kind="before")
    if _AFTER_RE.search(q):
        return TemporalConstraint(kind="after")
    if _ORIGINAL_RE.search(q):
        return TemporalConstraint(kind="original")
    return TemporalConstraint(kind="none")


class Coordinator:
    """Rule-based query router.

    Records every feature and decision for auditing.
    Never uses an LLM.
    """

    def route(self, query: str) -> CoordinatorDecision:
        q = query.strip()
        tokens = q.split()
        n_tokens = len(tokens)
        has_upper = sum(1 for t in tokens if t and t[0].isupper() and t not in ("What", "Who",
            "Where", "When", "Why", "How", "Which", "Can", "Is", "Are", "Do", "Does",
            "Tell", "Amid", "Any"))
        has_temporal = bool(TEMPORAL_PATTERNS.search(q))
        has_global = bool(GLOBAL_PATTERNS.search(q))
        entities = [e for e in ENTITY_RE.findall(q) if e not in QUESTION_WORDS]
        is_short = n_tokens <= 6

        features = {
            "n_tokens": n_tokens,
            "has_temporal": has_temporal,
            "has_global": has_global,
            "entity_count": len(entities),
            "is_short": is_short,
            "has_proper_nouns": has_upper > 0,
        }

        routes: list[Route] = []
        weights: dict[str, float] = {"lexical": 0.5, "vector": 0.5}

        if has_temporal:
            routes.append(Route.TEMPORAL)
            weights["temporal"] = 0.8

        if has_global:
            routes.append(Route.COMMUNITY_GLOBAL)
            weights["community"] = 0.7
            weights["vector"] = 0.6

        if entities and not has_global:
            routes.append(Route.LOCAL_GRAPH)
            weights["lexical"] = min(0.9, 0.5 + 0.1 * len(entities))

        if is_short or has_upper:
            weights["lexical"] = max(weights.get("lexical", 0.5), 0.7)

        if n_tokens > 10:
            weights["vector"] = max(weights.get("vector", 0.5), 0.7)

        if not routes:
            routes.append(Route.HYBRID)

        # Ensure at least LEXICAL and VECTOR are always in play
        if Route.LEXICAL not in routes and Route.HYBRID not in routes:
            routes.append(Route.LEXICAL)
        if Route.VECTOR not in routes and Route.HYBRID not in routes:
            routes.append(Route.VECTOR)

        # Confidence reflects how much positive routing signal the query carries.
        # A bare query with no entities / temporal / global cue is low-confidence:
        # the retriever should fall back to plain Hybrid RRF rather than trust a
        # weakly-motivated graph/temporal route.
        signal = 0.0
        if entities:
            signal += min(0.4, 0.2 * len(entities))
        if has_temporal:
            signal += 0.2
        if has_global:
            signal += 0.2
        if has_upper:
            signal += 0.1
        confidence = min(1.0, 0.3 + signal)
        features["confidence"] = confidence

        return CoordinatorDecision(
            routes=routes, features=features, weights=weights, confidence=confidence
        )
