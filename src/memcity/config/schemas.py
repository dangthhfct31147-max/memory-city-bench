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
    # Graph propagation (Phase 2). "bfs" is the legacy flat expansion; "weighted"
    # and "ppr" answer "which path fits the query" via edge-type weights.
    graph_mode: Literal["bfs", "weighted", "ppr"] = "bfs"
    degree_penalty: bool = False
    # Cap on how much score a single episode can accumulate from graph paths, so a
    # super-node cannot hoard relevance from many routes.
    max_graph_contribution_per_episode: float = 1.0
    # Optional per-edge-type weight overrides (defaults live in memory_city.py).
    edge_type_weights: dict[str, float] = Field(default_factory=dict)


class TemporalConfig(BaseModel):
    """Bi-temporal fact layer (Phase 3).

    When ``enable_facts`` is on the builder emits FactNodes with event-time
    validity (``valid_from``/``valid_to``) and SUPERSEDES/CONTRADICTS edges, and
    temporal queries are resolved against event time rather than a recency nudge.
    Extraction is deterministic by default (regex); ``spacy`` is opt-in.
    """

    enabled: bool = False
    enable_facts: bool = False
    fact_extractor: Literal["regex", "spacy"] = "regex"


class ContextualIndexConfig(BaseModel):
    """Contextual indexing for BM25 and embeddings (Phase 4).

    A short context prefix is prepended to each episode *before indexing* so that
    context-poor turns ("Yes, let's use that one.") become retrievable. The prefix
    only ever feeds the index; the raw episode text is preserved for provenance
    and is what ``RetrievedItem.text`` returns. Modes:

    * ``raw`` — no prefix (baseline behaviour).
    * ``deterministic`` — cheap prefix from session id, previous turn, timestamp;
      no LLM, no per-episode write cost that depends on a model.
    * ``llm`` — optional model-generated prefix (extra ``readers``); degrades to
      ``deterministic`` when no summariser is supplied.
    """

    mode: Literal["raw", "deterministic", "llm"] = "raw"
    max_context_tokens: int = 100


class HierarchicalConfig(BaseModel):
    """Hierarchical summary tier (Phase 6, RAPTOR-style; no LLM required).

    When ``enabled`` the builder condenses each session's episodes into an
    extractive SUMMARY node (top TF-IDF sentences), then optionally stacks further
    summary levels up to a root. A global/overview query is routed to the summary
    tree first and drilled down to the raw episodes beneath the best-matching
    summaries. Summaries only *route*; they never become final evidence, so
    retrieval metrics stay grounded in raw episodes.

    * ``max_levels`` — how many summary tiers to build (1 = session summaries
      only; 2 adds a tier over those; capped small for laptop budgets).
    * ``summary_max_sentences`` — extractive budget per summary node.
    * ``branching_factor`` — children grouped per higher-level summary.
    * ``summary_extractor`` — ``tfidf`` (deterministic, default) or ``llm``
      (opt-in; degrades to ``tfidf`` when no summariser is supplied).
    """

    enabled: bool = False
    max_levels: int = 1
    summary_max_sentences: int = 3
    branching_factor: int = 5
    summary_extractor: Literal["tfidf", "llm"] = "tfidf"


class CompressionConfig(BaseModel):
    """Deterministic evidence compression before the reader (Phase 7).

    Retrieval returns whole episodes; a small reader pays latency and tokens for
    every word. When ``enabled`` the compressor keeps only high-signal sentences
    (query overlap, named entities, numbers/dates, negations) per passage,
    preserving citation ids and ordering. No LLM: retrieval metrics are never
    affected and the transform is reproducible. The benchmark records the
    accuracy / coverage vs. token / latency trade-off.
    """

    enabled: bool = False
    max_sentences_per_passage: int = 3
    keep_numeric: bool = True
    keep_entities: bool = True
    keep_negations: bool = True
    min_query_overlap: int = 1


class RerankConfig(BaseModel):
    """Cross-encoder reranker (optional dependency: extra ``embeddings``).

    A two-stage retrieve-then-rerank design. When disabled, or when
    ``sentence-transformers`` is not installed, the pipeline is a pass-through and
    retrieval metrics are unaffected.
    """

    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: int = 50
    rerank_k: int = 30
    final_k: int = 10
    device: str = "cpu"


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
    # Phase 5: KV/prefix-cache metrics. When True the reader extracts per-call
    # timing and cached-token counts from the usage object and includes them in
    # ReaderResult. When False (default) these fields stay None to avoid
    # misleading zeros on backends that do not report them.
    capture_cache_metrics: bool = True
    # Stable prompt-prefix ordering. When True build_prompt places the static
    # system prompt first so KV-cache-aware backends can reuse the prefix across
    # queries. True by default; set False only to reproduce an old run that used
    # a different prompt order.
    prompt_prefix_stable_order: bool = True


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
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    contextual_index: ContextualIndexConfig = Field(default_factory=ContextualIndexConfig)
    hierarchical: HierarchicalConfig = Field(default_factory=HierarchicalConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
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
