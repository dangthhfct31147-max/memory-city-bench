# AGENTS.md — Context for AI coding sessions

Read this before making any change. It encodes the non-negotiable invariants of
this project so a fresh session does not drift off course. When a request seems to
conflict with these rules, stop and confirm with the user rather than guessing.

## What this project is

`memory-city-bench` (CLI: `memcity`) is a **research benchmark harness** for
evaluating long-term AI memory *retrieval* architectures. It empirically tests the
**Memory City** memory architecture against transparent baselines.

It is **NOT** a chatbot, an agent, a trained model, or a product. It is a
measurement instrument. Every design decision serves reproducibility and honest
measurement, not features or UX.

## The two claims (never conflate them)

- **Claim A — Retrieval-only:** Memory City finds and ranks correct evidence, and
  handles time, better than baselines — **without any generative LLM**.
- **Claim B — End-to-end:** Given retrieved evidence, a small fixed reader LLM
  answers more accurately, with grounding and better abstention.

Retrieval and end-to-end results are reported in **separate tables**. Retrieval
metrics must **never** depend on an LLM.

## Hard invariants (do not violate)

1. **LLMs are always optional.** No model may become a required dependency. An LLM
   is never used to compute retrieval metrics or to build the graph by default.
   Any feature must degrade gracefully when ML/LLM extras are not installed.
2. **Reproducibility first.** Fixed seeds, environment capture, resumable runs. A
   change that makes runs non-deterministic (given a seed) is a bug.
3. **No stale results scored.** Benchmark reads bypass the gateway semantic cache
   by default; latency stats never mix cache hits with real upstream calls.
4. **Secret hygiene.** API keys are read from an env var only. A key value must
   never appear in a cache key, cache file, `repr`, log line, error string, or any
   run artifact. Only the env-var *name* and an auth-enabled bool may be recorded.
5. **Scope isolation.** Retrieval is scope-isolated: one index per corpus scope
   (one per LoCoMo conversation, one per LongMemEval haystack). Cross-scope
   contamination must be zero by construction. The oracle also respects scope.
6. **Constrained environment.** Windows 10/11, NVIDIA RTX 5060 Laptop (8 GB VRAM),
   CPU fallback. No Neo4j, no Docker, no paid APIs. CLI only. Keep it that way.

## Repository layout

- `src/memcity/cli/main.py` — Typer CLI entry (`memcity`). All commands live here.
- `src/memcity/datasets/` — loaders + protocols: `synthetic`, `locomo`,
  `longmemeval`, `corpus` (scope grouping), `loader`.
- `src/memcity/retrieval/` — `baselines`, `memory_city`, `protocol`, `registry`.
- `src/memcity/graph/`, `memory/`, `coordinator/` — Memory City internals.
- `src/memcity/evaluation/` — `metrics`, `e2e_metrics`, `runner`, `diagnostics`.
- `src/memcity/readers/reader.py` — `OracleReader` (default, offline) and
  `OpenAICompatibleReader` (opt-in gateway/LLM path).
- `src/memcity/reporting/` — `tables`, `comparison`.
- `src/memcity/instrumentation/resources.py` — env + VRAM capture.
- `configs/memory_city.yaml` — resolved config schema mirror (`config/schemas.py`).
- `tests/unit/`, `tests/integration/` — pytest suite.
- `docs/` — `ARCHITECTURE.md`, `EXPERIMENT_DESIGN.md`, `METRICS.md`,
  `REPRODUCIBILITY.md`, `LIMITATIONS.md`. Read the relevant one before changing
  behavior it documents.

## Toolchain and commands

Package manager is **uv**. Python >= 3.11.

```bash
uv sync --extra dev                 # tests, ruff, mypy
uv sync --extra readers             # httpx (OpenAI-compatible reader)
uv sync --extra embeddings          # sentence-transformers (vector retrieval)

uv run pytest -q -rs                # full suite; always check the skipped count
uv run pytest -q -rs tests/unit/... # a single module
uv run ruff check .                 # lint (line-length 100; rules E,F,I,UP,B,W)
uv run mypy src                     # type check
```

## Conventions

- Style: ruff, line-length 100. Imports at top of file (no inline imports).
- Types: annotate public functions; `no_implicit_optional` is on.
- Keep changes minimal and scoped. This is a benchmark — do not add features,
  abstractions, or "improvements" the task did not ask for.
- Match existing patterns (Typer options, Pydantic schemas, protocol classes).
- Comments explain *why*, never narrate *what*.

## Verification before you say "done"

1. `uv run ruff check .` on touched files is clean.
2. `uv run pytest -q -rs` passes **and** has the expected skipped count. Optional
   deps (`httpx`, etc.) live behind extras; `importorskip` will silently skip a
   whole module if the extra is missing. Always verify **0 skipped** in the module
   you changed, not just "all passed".
3. For gateway/reader changes, confirm no secret leaks into artifacts.

## Datasets

Selected by name via `memcity.datasets.loader.get_dataset`:

- **`synthetic-<scale>`** — generated on the fly, seeded. Scales: `tiny`, `small`
  (and larger). Use for smoke tests. Generate with
  `memcity datasets generate synthetic --scale tiny --seed 42`.
- **`longmemeval-s`** / **`longmemeval-oracle`** — official LongMemEval (cleaned).
  `fetch` downloads from HuggingFace (`xiaowu0162/longmemeval-cleaned`). Scope =
  one haystack per query.
- **`locomo`** — official LoCoMo (`snap-research/locomo`, `locomo10.json`). Scope =
  one conversation.

Rules: never commit downloaded dataset files (they live under `data/`, gitignored).
Validate after fetching (`memcity datasets validate <name>`). `oracle` retriever is
only meaningful when the dataset carries evidence labels.

## Runs and artifacts

- Run IDs are time-sortable: `run_YYYYMMDD_HHMMSS_<6hex>` (`utils.helpers.run_id`).
  E2E runs use `e2e_<retriever>_<run_id>/`. Everything lands under `runs/`.
- Per-run artifacts: `config.resolved.*` (resolved config, secret-free),
  `environment.json` (env + versions), `metrics.json`, per-query and retrieval
  JSONL, `failures_*.jsonl`. `runs/` is gitignored — do not commit run outputs.
- IDs and hashes are deterministic (SHA-256 via `stable_hash`); JSON is written
  sorted for stable diffs. Keep it that way.

## Metrics (names are contractual — don't rename casually)

- Retrieval (`evaluation/metrics.py`): `recall@k`, `all_evidence_recall@k`,
  `ndcg@k`, `mrr`. Reported per top-k.
- End-to-end (`evaluation/e2e_metrics.py`): exact-match / correctness,
  `json_schema_success_rate`, and abstention (`abstention_precision`,
  `_recall`, `_f1`). Answerable vs. abstention queries are aggregated separately.
- `report compare` requires matching dataset hashes, query IDs, top-k, and
  evaluation protocol; it bootstraps CIs. Don't compare runs that disagree on these.

## Gotchas learned the hard way

- Shell is PowerShell on Windows; there is no heredoc. For multi-line git commit
  messages, use repeated `-m` flags.
- `pytest.importorskip("httpx")` means the gateway tests vanish (not fail) without
  `--extra readers`. A green "all passed" can hide skipped modules.
- The gateway-cache flag and its recorded artifact value must always agree — pass
  the flag through honestly, never hardcode the header while recording the flag.
- `X-OmniRoute-Provider` is best-effort (not in OmniRoute's documented header set);
  treat it as informational, fall back to the pinned provider id.
