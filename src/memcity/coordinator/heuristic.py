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


@dataclass
class CoordinatorDecision:
    routes: list[Route] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "routes": [r.value for r in self.routes],
            "features": self.features,
            "weights": self.weights,
        }


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
        entities = ENTITY_RE.findall(q)
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

        return CoordinatorDecision(routes=routes, features=features, weights=weights)
