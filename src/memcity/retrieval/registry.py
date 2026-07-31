"""Retriever registry — maps method names to retriever classes."""

from __future__ import annotations

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

_REGISTRY: dict[str, type] = {
    **BASELINE_REGISTRY,
    "memory_city_full": MemoryCityRetriever,
    "graph_only": MemoryCityRetriever,       # same, coordinator restricts to graph route
    "community_only": MemoryCityRetriever,   # same, coordinator restricts to community route
    "graph_vector": MemoryCityRetriever,
}


def get_retriever(method: str) -> object:
    """Instantiate a retriever by name."""
    cls = _REGISTRY.get(method)
    if cls is None:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown retriever '{method}'. Available: {available}")
    return cls()


def list_methods() -> list[str]:
    return sorted(_REGISTRY.keys())
