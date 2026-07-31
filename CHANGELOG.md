# Changelog

## 2026-07-31 — Complete the real-data benchmark path

- Adopt the official cleaned LongMemEval-S and official LoCoMo source files.
- Require all dataset-aware CLI commands to use the selected adapter rather than
  implicitly treating every dataset as synthetic.
- Complete the local OpenAI-compatible reader gate before optional native dependencies.
- Compare runs with a paired per-query bootstrap and strict compatibility checks.
- Keep HNSWLib and spaCy optional; the MVP retains exact NumPy and regex fallbacks.
- Implement official-source atomic fetch and adapters for cleaned LongMemEval and LoCoMo.
- Add strict paired bootstrap comparison with CSV, JSON, and Markdown artifacts.
- Verify 20-query retrieval runs for BM25, vector, hybrid RRF, and Memory City on both datasets.
- Verify Qwen3-0.6B Q8_0 through a local llama.cpp Vulkan server on LongMemEval-S.
- Add reproducible `readers`, `ann`, `nlp`, and `gpu` extras; pin `en_core_web_sm` in `uv.lock`.
