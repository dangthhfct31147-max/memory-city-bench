"""Pydantic configuration schemas for Memory City Bench."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 64
    cache_dir: str = "data/embedding_cache"
    normalize: bool = True


class IndexConfig(BaseModel):
    semantic_threshold: float = 0.75
    max_edges_per_node: int = 10
    temporal_edge: bool = True
    session_edge: bool = True
    community_algorithm: Literal["louvain", "girvan_newman", "greedy"] = "greedy"
    num_topic_hubs: int = 8
    readme_max_terms: int = 20
    readme_max_entities: int = 10


class RetrieverConfig(BaseModel):
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    vector_backend: Literal["numpy", "hnsw"] = "numpy"
    graph_hop_limit: int = 2
    temporal_decay_days: float = 365.0
    contradiction_penalty: float = 0.2
    provenance_boost: float = 0.1
    coordinator_enabled: bool = True


class ReaderConfig(BaseModel):
    provider: Literal["openai_compatible", "none"] = "none"
    base_url: str = "http://localhost:20128/v1"
    model: str = "qwen3-0.6b"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 256
    context_token_budget: int = 4096
    timeout_s: float = 60.0
    # Optional OmniRoute-style gateway settings. All opt-in; retrieval-only
    # benchmarks ignore these entirely. The API key is never stored in config —
    # only the name of the environment variable that holds it.
    api_key_env: str = "MEMCITY_LLM_API_KEY"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    # Bypass the gateway's own (semantic) cache. Default True so published runs
    # never score a stale generation.
    disable_gateway_cache: bool = True
    # Local response-cache mode: off | read-only | read-write.
    cache_mode: Literal["off", "read-only", "read-write"] = "read-write"
    # Optional OmniRoute provider id for a provider-locked chat route.
    omniroute_provider: str | None = None


class BenchmarkConfig(BaseModel):
    dataset: str = "synthetic-tiny"
    methods: list[str] = Field(default_factory=lambda: ["bm25", "vector", "memory_city_full"])
    top_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    seeds: list[int] = Field(default_factory=lambda: [42])
    warmup: int = 5
    limit: int | None = None
    output_dir: str = "runs"
    resume: bool = True


class MemCityConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reader: ReaderConfig = Field(default_factory=ReaderConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    data_dir: str = "data"
    runs_dir: str = "runs"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path) -> MemCityConfig:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)
