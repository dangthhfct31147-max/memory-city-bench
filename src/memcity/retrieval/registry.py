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

# Ablation flag presets. Each disables exactly the components not named, so the
# reported delta between two variants isolates a single component's effect.
_ABLATION_CONFIGS: dict[str, dict] = {
    # Hybrid = BM25 + vector RRF only (graph/temporal/community/provenance off).
    "hybrid_mc": dict(
        enable_graph_expansion=False, enable_temporal=False,
        enable_community=False, enable_provenance=False, enable_be=False,
        coordinator_enabled=False,
    ),
    "hybrid_temporal": dict(
        enable_graph_expansion=False, enable_temporal=True,
        enable_community=False, enable_provenance=False, enable_be=False,
    ),
    "hybrid_graph": dict(
        enable_graph_expansion=True, enable_temporal=False,
        enable_community=False, enable_provenance=False, enable_be=False,
    ),
    "hybrid_community": dict(
        enable_graph_expansion=False, enable_temporal=False,
        enable_community=True, enable_provenance=False, enable_be=False,
    ),
    "hybrid_be": dict(
        enable_graph_expansion=False, enable_temporal=False,
        enable_community=False, enable_provenance=False, enable_be=True,
    ),
    "hybrid_graph_temporal": dict(
        enable_graph_expansion=True, enable_temporal=True,
        enable_community=False, enable_provenance=False, enable_be=False,
    ),
    "memory_city_full": {},  # all components on (defaults)
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
    """Ordered ablation ladder from plain hybrid up to full Memory City."""
    return [
        "bm25",
        "vector",
        "hybrid_rrf",
        "hybrid_mc",
        "hybrid_temporal",
        "hybrid_graph",
        "hybrid_community",
        "hybrid_be",
        "hybrid_graph_temporal",
        "memory_city_full",
    ]
