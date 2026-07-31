# memory-city-bench

A CLI research harness for evaluating **long-term AI memory retrieval** architectures.

`memory-city-bench` (CLI: `memcity`) empirically tests the **Memory City** memory
architecture — a city metaphor for AI memory (Begin/End nodes, roads, topic hubs,
communities, READMEs, coordinator routing) — against strong, transparent baselines.

It is a **benchmark**, not a chatbot or a trained model. It separates two claims:

- **Claim A — Retrieval-only:** Memory City finds correct evidence, ranks it well,
  and handles time better than baselines — *without any generative LLM*.
- **Claim B — End-to-end:** Given Memory City evidence, a small fixed reader LLM
  answers more accurately, with grounding and better abstention.

Retrieval and end-to-end results are reported in **separate tables**. LLMs are
never a required dependency and are never used to compute retrieval metrics or to
build the graph by default.

## Design constraints

- Windows 10/11, NVIDIA RTX 5060 Laptop (8 GB VRAM), CPU fallback.
- No Neo4j, no Docker, no paid APIs. CLI only.
- Every model can be fully disabled. Datasets, embeddings, inference are cached.
- Reproducible: fixed seeds, environment capture, resumable runs.

## Install

```bash
uv sync                 # core (no ML/LLM)
uv sync --extra dev     # + pytest, ruff, mypy
uv sync --extra embeddings   # + sentence-transformers (vector retrieval)
uv sync --extra gpu     # + nvidia-ml-py (VRAM measurement)
uv sync --extra readers # + httpx (OpenAI-compatible local reader)
```

Optional native-build deps are opt-in: `uv sync --extra ann` for HNSWLib and
`uv sync --extra nlp` for spaCy. Exact NumPy retrieval and regex extraction remain
the core fallbacks.

## Quick start / smoke test

```bash
uv run memcity doctor
uv run memcity datasets generate synthetic --scale tiny --seed 42
uv run memcity datasets validate synthetic-tiny
uv run memcity benchmark retrieval \
  --dataset synthetic-tiny \
  --methods random,recency,bm25,hybrid_rrf,memory_city_full \
  --top-k 1,3,5,10 \
  --seeds 42 \
  --warmup 3
```

`bm25`, `random`, `recency`, `hybrid_rrf`, `memory_city_full` run **without any ML deps**
(vector stages are skipped gracefully). Install `--extra embeddings` to enable
`vector` and the semantic stages of `memory_city_full`.

## First real benchmark (with vector retrieval)

```bash
uv sync --extra embeddings --extra dev
uv run memcity benchmark retrieval \
  --dataset synthetic-small \
  --methods bm25,vector,hybrid_rrf,graph_only,community_only,memory_city_full \
  --top-k 1,3,5,10 \
  --seeds 42,43,44 \
  --warmup 10 \
  --output runs/
```

## Official datasets and run comparison

```bash
uv run memcity datasets fetch longmemeval --variant s
uv run memcity datasets fetch locomo
uv run memcity datasets validate longmemeval-s
uv run memcity datasets validate locomo

uv run memcity benchmark retrieval --dataset longmemeval-s \
  --methods bm25,vector,hybrid_rrf,memory_city_full \
  --limit 20 --top-k 1,5,10 --seeds 42

uv run memcity report compare RUN_ID_1 RUN_ID_2 \
  --metric recall_at_10 --bootstrap-resamples 10000 \
  --confidence 0.95 --seed 42
```

`report compare` requires matching dataset hashes, query IDs, top-k values, and
evaluation protocols. It writes `comparison.csv`, `comparison.json`, and `report.md`.

## End-to-end (optional reader LLM)

Run a local OpenAI-compatible endpoint (for example, the official Qwen3-0.6B
Q8_0 GGUF with llama.cpp) then:

```powershell
llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 `
  --alias qwen3-0.6b -c 4096 -ngl 99 -np 1 -cram 0 `
  --no-cache-idle-slots --reasoning off --host 127.0.0.1 --port 8080
```

Choose another free local port if `8080` is already in use.

```bash
uv run memcity benchmark e2e \
  --dataset longmemeval-s \
  --retrievers bm25,hybrid_rrf,memory_city_full,oracle \
  --reader-provider openai-compatible \
  --reader-base-url http://127.0.0.1:8080/v1 \
  --reader-model qwen3-0.6b \
  --context-token-budget 4096 \
  --limit 20
```

See [docs/](docs/) for architecture, experiment design, metrics, reproducibility, and limitations.

## Methods compared

`random`, `recency`, `bm25`, `vector`, `hybrid_rrf`, `graph_only`, `community_only`,
`graph_vector`, `memory_city_full`, and `oracle` (when the dataset has evidence labels).

## License

Apache-2.0. See [LICENSE](LICENSE).
