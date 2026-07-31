"""Retriever registry — maps method names to retriever factories.

Ablation variants are real: each builds a ``MemoryCityRetriever`` with specific
components disabled, so the difference between two rows is exactly one component
evaluated against the identical index.
"""

from __future__ import annotations

from typing import Callable

from memcity.retrieval.baselines import (
    BASELINE_REGISTRY,
    BM25Retriever,
    HybridRRFRetriever,
    OracleRetriever,
    RandomRetriever,
    RecencyRetriever,
    VectorRetriever,
)
from memcity.retrieval.memory_city import MemoryCityRetriever

# Ablation presets.
#
# Two disciplined designs are provided (review issue #8):
#
#  * INDEPENDENT — every variant starts from the SAME hybrid baseline
#    (coordinator OFF so it never adds noise) and enables exactly ONE component.
#    The delta of ``hybrid_<x>`` minus ``hybrid`` isolates component x's effect.
#
#  * CUMULATIVE ("ladder") — each step adds exactly ONE component on top of the
#    previous step, ending at full Memory City. Adjacent deltas attribute gain
#    to the single component that was switched on at that step.
#
# The key correctness fix: with ``coordinator_enabled=False`` the retriever fires
# every enabled component unconditionally (see MemoryCityRetriever._route_active),
# so a flag genuinely turns its component on rather than depending on the
# coordinator's routing decision. B/E enables graph traversal so it can actually
# be exercised (the flag alone was previously a no-op).

# Baseline shared by the independent variants: everything off, coordinator off.
_HYBRID_BASE = dict(
    enable_graph_expansion=False, enable_temporal=False,
    enable_community=False, enable_provenance=False, enable_be=False,
    coordinator_enabled=False,
)


def _with(**overrides: object) -> dict:
    cfg = dict(_HYBRID_BASE)
    cfg.update(overrides)
    return cfg


_ABLATION_CONFIGS: dict[str, dict] = {
    # ── Independent ablation (each adds ONE component to the hybrid base) ──────
    "hybrid": _with(),  # BM25 + vector RRF only — the common baseline
    "hybrid_mc": _with(),  # legacy alias for the hybrid baseline
    "hybrid_coordinator": _with(coordinator_enabled=True),
    "hybrid_temporal": _with(enable_temporal=True),
    "hybrid_graph": _with(enable_graph_expansion=True),
    "hybrid_community": _with(enable_community=True),
    # B/E needs graph traversal to matter, so it turns graph expansion on and
    # builds B/E nodes; compare against hybrid_graph to isolate the B/E effect.
    "hybrid_be": _with(enable_graph_expansion=True, enable_be=True),
    "hybrid_provenance": _with(enable_provenance=True),
    "hybrid_graph_temporal": _with(enable_graph_expansion=True, enable_temporal=True),

    # ── Cumulative ladder (each step adds exactly one component) ──────────────
    "ladder_0_hybrid": _with(),
    "ladder_1_coordinator": _with(coordinator_enabled=True),
    "ladder_2_temporal": _with(coordinator_enabled=True, enable_temporal=True),
    "ladder_3_graph": _with(
        coordinator_enabled=True, enable_temporal=True, enable_graph_expansion=True,
    ),
    "ladder_4_community": _with(
        coordinator_enabled=True, enable_temporal=True, enable_graph_expansion=True,
        enable_community=True,
    ),
    "ladder_5_be": _with(
        coordinator_enabled=True, enable_temporal=True, enable_graph_expansion=True,
        enable_community=True, enable_be=True,
    ),
    "ladder_6_provenance": _with(
        coordinator_enabled=True, enable_temporal=True, enable_graph_expansion=True,
        enable_community=True, enable_be=True, enable_provenance=True,
    ),

    "memory_city_full": {},  # all components on (defaults, coordinator on)
}


def _mc_factory(method: str, config: dict) -> Callable[[], MemoryCityRetriever]:
    def factory() -> MemoryCityRetriever:
        return MemoryCityRetriever(name=method, **config)

    return factory


_BASELINE_FACTORIES: dict[str, Callable[[], object]] = {
    name: (lambda cls=cls: cls()) for name, cls in BASELINE_REGISTRY.items()
}


def get_retriever(method: str) -> object:
    """Instantiate a single retriever by name (legacy single-instance path)."""
    return get_retriever_factory(method)()


def get_retriever_factory(method: str) -> Callable[[], object]:
    """Return a zero-arg factory that builds a fresh retriever each call.

    The runner uses this to build one independent index per corpus scope.
    """
    if method in _ABLATION_CONFIGS:
        return _mc_factory(method, _ABLATION_CONFIGS[method])
    if method in _BASELINE_FACTORIES:
        return _BASELINE_FACTORIES[method]
    available = sorted(set(_ABLATION_CONFIGS) | set(_BASELINE_FACTORIES))
    raise ValueError(f"Unknown retriever '{method}'. Available: {available}")


def list_methods() -> list[str]:
    return sorted(set(_ABLATION_CONFIGS) | set(_BASELINE_FACTORIES))


def ablation_methods() -> list[str]:
    """Cumulative ablation ladder: each step adds exactly one component.

    Adjacent deltas attribute gain/loss to the single component switched on at
    that step, ending at full Memory City. Baselines are included at the top
    for reference.
    """
    return [
        "bm25",
        "vector",
        "hybrid_rrf",
        "ladder_0_hybrid",
        "ladder_1_coordinator",
        "ladder_2_temporal",
        "ladder_3_graph",
        "ladder_4_community",
        "ladder_5_be",
        "ladder_6_provenance",
        "memory_city_full",
    ]


def independent_ablation_methods() -> list[str]:
    """Independent ablation: each variant adds one component to the same baseline.

    ``hybrid_<x>`` minus ``hybrid`` isolates component x against a fixed baseline
    (coordinator off, so routing never confounds the delta).
    """
    return [
        "hybrid",
        "hybrid_coordinator",
        "hybrid_temporal",
        "hybrid_graph",
        "hybrid_community",
        "hybrid_be",
        "hybrid_provenance",
    ]
