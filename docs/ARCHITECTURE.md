# Architecture

## Overview

```
memcity/
├── cli/          ← Typer CLI entry-point (doctor, datasets, benchmark, report, index)
├── config/       ← Pydantic schemas for all configuration
├── datasets/     ← Dataset protocol, synthetic generator, external adapters
├── memory/       ← Node/Edge schemas (Pydantic), SQLite Store
├── graph/        ← MemoryCityGraphBuilder (deterministic, no LLM)
├── retrieval/    ← Protocol, baselines (random/recency/bm25/vector/rrf/oracle), MemoryCityRetriever, registry
├── coordinator/  ← Heuristic query router (no LLM)
├── evaluation/   ← Metrics, RetrievalBenchmarkRunner
├── instrumentation/ ← LatencyStats, resource measurement, environment capture
├── reporting/    ← Rich tables, markdown/CSV export
├── readers/      ← Optional LLM reader (OpenAI-compatible endpoint)
└── utils/        ← Hashing, IDs, seeds, JSONL, tokenizer
```

## Data flow

```
[Corpus (QASample history)]
    │
    ▼
[Store (SQLite)]
    │
    ├─ Episode nodes
    ├─ Begin/End nodes (B→E journeys per session)
    ├─ Entity nodes (regex extraction)
    ├─ NEXT / SAME_SESSION / BEGINS / ENDS_WITH / MENTIONS edges
    ├─ SEMANTICALLY_RELATED edges (embedding cosine, optional)
    ├─ Topic Hub nodes (K-Means on embeddings, optional)
    └─ Community nodes + extractive README (greedy modularity)

[Query]
    │
    ▼
[Coordinator]  ← heuristic rule engine
    │ routes: TEMPORAL / LOCAL_GRAPH / COMMUNITY_GLOBAL / HYBRID
    ▼
[BM25]  +  [Vector]  +  [Community search via README]
    │         │               │
    └─────────┴───────────────┘
               │ RRF fusion
               ▼
         [Graph expansion] (BFS, optional)
               │
         [Temporal reordering] (optional)
               │
         [Provenance reranking]
               │
               ▼
         [RetrievalResult with trace]
```

## Memory City node types

| Type | Role |
|------|------|
| `episode` | Raw conversation turn (user + assistant) |
| `begin` | Extractive summary of session start (B node) |
| `end` | Extractive summary of session end (E node) |
| `entity` | Named entity from regex/spaCy extraction |
| `topic_hub` | K-Means cluster centre for a topic |
| `community` | Group of related episodes with README |
| `fact` | Optional explicit fact triple (subject, predicate, object) |

## Edge types

`BEGINS`, `ENDS_WITH`, `NEXT`, `SAME_SESSION`, `SEMANTICALLY_RELATED`,
`MENTIONS`, `BELONGS_TO_TOPIC`, `BELONGS_TO_COMMUNITY`, `SUPPORTS`,
`CONTRADICTS`, `SUPERSEDES`, `DERIVED_FROM`, `CAUSED_BY`, `DEPENDS_ON`

## Retriever interface

```python
class Retriever(Protocol):
    name: str
    def build(self, corpus: list[dict], config: Any) -> IndexStats: ...
    def query(self, query: str, top_k: int = 10, trace: bool = False) -> RetrievalResult: ...
    def close(self) -> None: ...
```

All 9 methods share the same `corpus` (episode dicts), the same embedding model
when applicable, and the same `top_k`. No method receives extra information that
others do not.

## Fairness guarantees

- Ground-truth labels are never passed to non-oracle retrievers.
- The oracle method is clearly labelled and only used to measure reader ceiling.
- All methods use the same tokenizer, the same embedding model, and the same
  context token budget.
- Samples with no evidence are counted in the denominator (not silently dropped).

## Reproducibility

Every run writes `environment.json` capturing:
Python version, OS, CPU, RAM, GPU/VRAM, git commit, all package versions,
random seeds, and the original command.

Retrieval with BM25 is fully deterministic. Vector retrieval depends on
sentence-transformers; embeddings are cached in SQLite after the first build.
