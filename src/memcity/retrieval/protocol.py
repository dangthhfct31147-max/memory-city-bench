"""Retriever protocol and result schemas."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RetrievedItem(BaseModel):
    id: str
    text: str
    score: float = 0.0
    source_episode_ids: list[str] = Field(default_factory=list)
    node_type: str = "episode"
    stage_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    query: str
    routes: list[str] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    candidate_count_before: int = 0
    candidate_count_after: int = 0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Per-source candidate lists, each capped at candidate_k so every method is
    # measured at the same retrieval depth (the review's #1 fairness fix).
    bm25_candidates: list[str] = Field(default_factory=list)
    vector_candidates: list[str] = Field(default_factory=list)
    graph_candidates: list[str] = Field(default_factory=list)
    community_candidates: list[str] = Field(default_factory=list)
    # Unranked union of every candidate the pipeline saw (answers "is the
    # evidence anywhere in the pool?" — candidate_pool_recall).
    candidate_pool: list[str] = Field(default_factory=list)
    # Ranked union after fusion (answers candidate_ranked_recall). Kept under the
    # legacy name ``union_candidates`` for back-compat with old artifacts.
    union_candidates: list[str] = Field(default_factory=list)
    # Stage-by-stage ranked snapshots of raw episodes, enabling per-stage loss
    # attribution (fusion → graph → temporal → provenance → final).
    post_fusion_ranking: list[str] = Field(default_factory=list)
    pre_graph: list[str] = Field(default_factory=list)
    post_graph: list[str] = Field(default_factory=list)
    post_temporal: list[str] = Field(default_factory=list)
    post_provenance: list[str] = Field(default_factory=list)
    pre_rerank: list[str] = Field(default_factory=list)
    post_rerank: list[str] = Field(default_factory=list)
    final_ranking: list[str] = Field(default_factory=list)
    candidate_k: int = 0
    ground_truth_ranks: dict[str, int] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    items: list[RetrievedItem] = Field(default_factory=list)
    top_k: int = 10
    latency_ms: float = 0.0
    trace: RetrievalTrace | None = None
    error: str | None = None

    def ids(self) -> list[str]:
        return [item.id for item in self.items]

    def episode_ids(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in self.items:
            for ep in item.source_episode_ids:
                if ep not in seen:
                    seen.add(ep)
                    out.append(ep)
            if item.node_type == "episode" and item.id not in seen:
                seen.add(item.id)
                out.append(item.id)
        return out


class IndexStats(BaseModel):
    method: str
    node_count: int = 0
    edge_count: int = 0
    community_count: int = 0
    embedding_count: int = 0
    index_size_bytes: int = 0
    build_wall_time_s: float = 0.0
    build_cpu_time_s: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    name: str

    def build(self, corpus: list[dict], config: Any) -> IndexStats: ...
    def query(
        self,
        query: str,
        top_k: int = 10,
        trace: bool = False,
        candidate_k: int = 100,
    ) -> RetrievalResult: ...
    def close(self) -> None: ...
