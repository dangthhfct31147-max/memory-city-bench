# Memory City Bench — Implementation Plan

## Status: Requested completion gates complete on official data

## Completion scope — 2026-07-31

Work proceeds through one verified gate at a time:

1. Fetch and validate the official cleaned LongMemEval-S and LoCoMo datasets.
2. Run a 20-query retrieval smoke on each real dataset.
3. Run the same samples end-to-end through a local OpenAI-compatible reader.
4. Add paired per-query bootstrap comparison to `memcity report compare`.
5. Evaluate optional HNSWLib and spaCy extras only after the core gates pass.

Decisions:

- External adapters must parse the official upstream files directly; no hand-written
  intermediate QA file is required.
- Dataset selection is shared by `validate`, `index`, retrieval, and e2e commands so a
  requested real dataset can never silently fall back to synthetic data.
- Comparisons operate on per-query artifacts, reject incompatible runs, and retain
  failed-query counts instead of dropping failures silently.
- Exact NumPy retrieval and regex entity extraction remain the supported core fallback;
  HNSWLib and spaCy stay opt-in.

---

## Phase 1 — Foundation (DONE)

- [x] `pyproject.toml` with `uv` and Hatchling
- [x] Source scaffold: `src/memcity/` with all subpackages
- [x] `MemCityConfig` (Pydantic) + config loading from YAML
- [x] Utility helpers: `stable_hash`, `make_id`, `run_id`, `seed_everything`, `tokenize`, JSONL I/O
- [x] Memory City schema: `NodeType`, `EdgeType`, `CreationMethod`, all node/edge Pydantic models
- [x] SQLite `Store`: episodes, FTS5, nodes, edges, embeddings, runs — with WAL mode
- [x] Dataset `BaseDataset` protocol + `QASample`, `EpisodeTurn`, `DatasetManifest`
- [x] `SyntheticDataset` — deterministic generator for tiny/small/medium/stress, covering 8+ question categories
- [x] Evaluation `metrics.py` — recall@k, precision@k, any-hit@k, all-evidence@k, MRR, nDCG@k, evidence-F1, RRF, bootstrap CI
- [x] Retriever `Protocol` with `RetrievedItem`, `RetrievalResult`, `RetrievalTrace`, `IndexStats`
- [x] `BM25Retriever`, `RandomRetriever`, `RecencyRetriever`
- [x] Typer CLI: `memcity doctor`, `datasets generate/validate/list/fetch`, `benchmark retrieval`, `report show/leaderboard/export`, `index build`
- [x] Unit and integration suite (84 passing)
- [x] `uv run memcity doctor` works
- [x] `uv run memcity datasets generate --scale tiny` works
- [x] `uv run memcity benchmark retrieval --methods bm25,hybrid_rrf,memory_city_full` works

## Phase 2 — Vector & Hybrid (DONE)

- [x] `VectorRetriever` — sentence-transformers / numpy cosine (optional import)
- [x] `HybridRRFRetriever` — BM25 + Vector fused with RRF
- [x] `OracleRetriever` — ground-truth ceiling
- [x] `LatencyStats` + `ResourceSnapshot` instrumentation
- [x] `collect_environment()` — captures Python, OS, CPU, RAM, GPU, git commit, package versions
- [x] `RetrievalBenchmarkRunner` — warmup, per-seed runs, artifact writing
- [x] Rich tables: `print_retrieval_table`, `print_category_table`
- [x] Artifacts: `metrics.json`, `environment.json`, `config.resolved.yaml`, `retrieval_results_{seed}.jsonl`, `failures_{seed}.jsonl`
- [x] Markdown + CSV export

## Phase 3 — Memory City Graph (DONE)

- [x] `MemoryCityGraphBuilder` — deterministic, no LLM
  - Episode nodes + NEXT/SAME_SESSION edges
  - Begin/End nodes (B→E journeys) + BEGINS/ENDS_WITH edges
  - Entity extraction (regex) + MENTIONS edges
  - Semantic edges from embedding cosine similarity (optional)
  - Topic hubs via K-Means + TF-IDF term labeling (optional)
  - Communities via NetworkX `greedy_modularity_communities`
  - Extractive README per community: top TF-IDF terms, entities, time range, source episodes
- [x] `MemoryCityRetriever` — integrates BM25 + vector + graph expansion + community routing + temporal reordering + provenance reranking
- [x] Heuristic `Coordinator` — rule-based routing (TEMPORAL, COMMUNITY_GLOBAL, LOCAL_GRAPH, HYBRID); fully traceable, no LLM
- [x] Retriever `registry` — `get_retriever(name)` for all 9 methods

## Phase 4 — Benchmark Adapters (IN PROGRESS)

- [x] Synthetic dataset (fully implemented)
- [x] LongMemEval cleaned-data fetch, parser, full validation, and retrieval smoke
- [x] LoCoMo official-file fetch, parser, full validation, and retrieval smoke
- [ ] LongMemEval-V2 skeleton

## Phase 5 — End-to-End Reader (IN PROGRESS)

- [x] `ReaderConfig` Pydantic schema
- [x] OpenAI-compatible reader implementation (httpx)
- [x] JSON schema validation + one deterministic repair attempt
- [x] Reader prompt with abstention + evidence citation
- [x] `memcity benchmark e2e` command with oracle reader
- [x] Real local llama.cpp/Qwen3-0.6B Q8_0 smoke with environment and artifact verification

## Phase 6 — Ablation & Statistics (PARTIAL)

- [x] `memcity benchmark ablation` command (delegates to `benchmark retrieval`)
- [x] Multi-seed support in runner
- [x] Bootstrap CI in `metrics.py`
- [x] Paired bootstrap delta test (`memcity report compare`)
- [ ] Pareto frontier report

## Optional runtime extras (COMPLETE)

- [x] HNSWLib 0.8.0 built with the existing Visual Studio Build Tools workload
- [x] spaCy 3.8 with pinned `en_core_web_sm` 3.8.0 model
- [x] Exact NumPy and regex fallbacks retained as the default MVP path

---

## ADRs (Architectural Decision Records)

### ADR-001: In-memory Store for MemoryCityRetriever
Decision: `MemoryCityRetriever.build()` creates an in-memory SQLite store, not a persistent one.
Rationale: Keeps the retriever self-contained. Persistent indexing is deferred to `memcity index build`.
Trade-off: Graph is rebuilt each benchmark run unless cached externally.

### ADR-002: FTS5 as standalone tables (not external content)
Decision: `episodes_fts` and `nodes_fts` are standalone FTS5 tables, not external-content tables.
Rationale: External-content FTS5 requires row IDs to match the parent table — fragile with custom IDs.
Trade-off: Slight storage duplication; acceptable for research use.

### ADR-003: Greedy modularity communities as default
Decision: Default community algorithm is `networkx.greedy_modularity_communities`, not Louvain.
Rationale: Louvain (`python-louvain`) is an optional dependency; greedy is always available.
Trade-off: Louvain may produce better communities on larger graphs; configurable.

### ADR-004: aggregate_metrics skips non-numeric keys
Decision: `aggregate_metrics()` silently skips non-numeric values (e.g. `sample_id`, `category`).
Rationale: Per-query dicts carry metadata keys for debugging; averaging them makes no sense.
Consequence: Callers must not rely on metadata keys appearing in the aggregate.

### ADR-005: No LLM for retrieval metrics or graph building
Decision: LLMs are never invoked by default in the retrieval or graph-build pipeline.
Rationale: LLM outputs are non-deterministic and would contaminate retrieval metrics.
Trade-off: Graph quality is bounded by heuristic entity extraction and clustering.
