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
    
    bm25_candidates: list[str] = Field(default_factory=list)
    vector_candidates: list[str] = Field(default_factory=list)
    graph_candidates: list[str] = Field(default_factory=list)
    community_candidates: list[str] = Field(default_factory=list)
    union_candidates: list[str] = Field(default_factory=list)
    pre_rerank: list[str] = Field(default_factory=list)
    post_rerank: list[str] = Field(default_factory=list)
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
    def query(self, query: str, top_k: int = 10, trace: bool = False) -> RetrievalResult: ...
    def close(self) -> None: ...
