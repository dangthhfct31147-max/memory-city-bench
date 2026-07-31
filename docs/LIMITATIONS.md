# Known Limitations

## Graph Construction

**Entity extraction quality**
- The default entity extractor uses regex patterns (CAPITALISED words, quoted phrases).
  It does not perform coreference resolution, so "Alice" and "she" are not linked.
- Optional spaCy adapter improves NER but adds a large dependency and download.
- Without high-quality entity extraction, `LOCAL_GRAPH` queries that traverse entity
  nodes will underperform compared to a production system with full NER.

**Semantic edges may introduce noise**
- SEMANTICALLY_RELATED edges are added when cosine similarity exceeds a threshold.
  Lexically similar but logically unrelated episodes will be connected.
- This can degrade multi-hop precision by creating spurious traversal paths.

**Community detection on small corpora**
- Greedy modularity produces trivial communities (all episodes in one cluster) when
  the corpus is small and well-connected.
- Community-based routing benefits only when there are ≥50+ diverse episodes.

**Community READMEs drift over time**
- READMEs are generated once per build. Incrementally adding episodes requires
  rebuilding communities (or at minimum, regenerating the README).
- READMEs are navigational aids, not ground truth. Always trace back to raw episodes.

## Retrieval

**Coordinator is heuristic, not learned**
- The coordinator uses manually crafted rules. It can misroute queries that do not
  match the patterns (e.g. temporal queries phrased without explicit time words).
- A learned router could improve routing quality at the cost of adding a training
  dependency (out of scope for MVP).

**No cross-session coreference**
- The same person or concept across different sessions is not unified unless entity
  names match exactly. Cross-session multi-hop reasoning will fail for paraphrased
  entity references.

**Graph expansion is breadth-first with a fixed depth**
- Deep chains of reasoning (>3 hops) are not reliably recovered.
- Deeper BFS increases latency super-linearly on large graphs.

## End-to-End Reader

**Small LLM JSON compliance**
- Qwen3-0.6B sometimes generates invalid JSON or fails to follow the abstention
  instruction, especially for complex multi-step questions.
- The pipeline retries once and then marks the sample as a schema failure.
- Schema failure rate is always reported; do not filter these samples.

**Reader model cannot be the judge**
- We explicitly prohibit using the reader LLM to evaluate its own answers.
- All scoring uses official evaluators, exact match, or token F1.

**Context window pressure**
- At `context_token_budget=4096`, only the top few retrieved episodes fit.
  The reader may miss evidence that is correctly retrieved but truncated from context.

## Benchmarks

**Synthetic ≠ real data**
- The synthetic dataset tests specific capabilities in isolation with clean, unambiguous
  evidence. Real-world conversations are noisier, more ambiguous, and harder to evaluate.
- Always benchmark on LongMemEval and LoCoMo before making claims.

**LoCoMo evidence is not available for every QA row**
- The official file includes some non-adversarial questions without evidence dialog IDs.
- The adapter keeps these rows and records an empty evidence list; it does not silently
  drop them. Retrieval metrics therefore follow the project's existing empty-evidence
  definitions, while end-to-end results should be interpreted separately.

**Small sample counts inflate variance**
- Benchmarks with <100 samples have wide confidence intervals. Bootstrap CI is always
  reported. Do not cite means from n<50 as definitive results.

**No production scalability guarantee**
- This tool is validated on tiny/small/medium synthetic corpora and the small variants
  of LongMemEval and LoCoMo.
- Performance on millions of episodes requires HNSWLib (optional) and a persistent
  graph store beyond SQLite.

## Platform

**Windows path handling**
- All paths use `pathlib.Path` to avoid cross-platform issues. If you encounter a path
  error, please file a bug with the full path and OS version.

**VRAM measurement**
- VRAM usage is measured via `pynvml`. On systems without an NVIDIA driver or `pynvml`
  installed, VRAM is always reported as 0.

**HNSWLib on Windows**
- HNSWLib requires a C++ build toolchain (MSVC). Install Visual Studio Build Tools
  before running `uv sync --extra hnsw`.
