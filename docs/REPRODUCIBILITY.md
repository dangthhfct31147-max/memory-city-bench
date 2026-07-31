# Reproducibility Guide

## Reproducible benchmark runs

Every benchmark run captures a full environment snapshot in `runs/<run-id>/environment.json`:

```json
{
  "python_version": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "cpu": "...",
  "ram_gb": 15.3,
  "gpu": "NVIDIA RTX 5060 Laptop GPU",
  "vram_gb": 8.0,
  "git_commit": "abc1234",
  "packages": {"rank-bm25": "0.2.2", ...},
  "seeds": [42],
  "command": "memcity benchmark retrieval ..."
}
```

## Seeds

All stochastic components respect `--seeds`:

```bash
memcity benchmark retrieval --seeds 42,43,44 --methods bm25,vector,memory_city_full
```

This sets Python `random`, `numpy.random`, and optionally `torch` seeds before each run.

BM25 retrieval is fully deterministic (no random component). Vector retrieval is
deterministic given the same embedding model and cached embeddings. Community
detection is deterministic given fixed seeds for K-Means.

Latency measurements inherently vary between runs. Only non-timing metrics
(recall, precision, MRR, nDCG) must match exactly between same-seed runs.

## Embedding cache

Embeddings are computed once and cached in the run's SQLite store. Subsequent
runs with the same corpus and model skip re-computation.

## Dataset hashing

`DatasetManifest.corpus_hash` is a SHA-256 hash of all episode texts, computed
before indexing. If the hash changes, the manifest logs it.

## Re-running from a manifest

```bash
memcity benchmark retrieval \
  --dataset synthetic-tiny \
  --methods bm25 \
  --seeds 42
```

Loads the same dataset, rebuilds the index from scratch, and produces identical
retrieval rankings (for deterministic methods).

## What can legitimately differ between runs

- Query latency (`latency_ms`, `p50_ms` etc.) due to OS scheduling.
- VRAM usage if the GPU is shared.
- Model loading time.

## Checking reproducibility manually

```bash
# Run twice with same seed
uv run memcity benchmark retrieval --dataset synthetic-tiny --methods bm25 --seeds 42
uv run memcity benchmark retrieval --dataset synthetic-tiny --methods bm25 --seeds 42

# Compare metrics (latency excluded)
uv run memcity report compare <run-id-1> <run-id-2>
```

## Pinned dependencies

Dependencies are pinned via `uv.lock`. To reproduce on a new machine:

```bash
uv sync --frozen
```

Never run `uv sync` without `--frozen` in a reproducibility-critical environment.
