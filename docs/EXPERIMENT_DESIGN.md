# Experiment Design

This document explains how Memory City Bench is designed and how it relates to prior work.

## Research questions

### Claim A — Retrieval quality (no LLM)

> "Memory City finds correct evidence, ranks it well, uses fewer resources, and
> handles temporal updates better than strong baselines."

Measured by: Recall@k, Precision@k, MRR, nDCG@k, Any-hit@k, All-evidence recall@k,
latency percentiles, and RAM/VRAM usage — **without any generative LLM**.

### Claim B — End-to-end quality (fixed reader LLM)

> "When Memory City evidence is fed to a small, fixed reader LLM, the model
> answers more accurately, with better grounding and abstention."

Measured by: Exact Match, Token F1, abstention metrics, grounding rate, JSON
schema success — using the **same** reader model, prompt, and context budget for
every retriever.

These two claims are reported in **separate tables**. Never merge them.

---

## How this project relates to prior work

### LongMemEval (Wu et al. 2024)

Key ideas borrowed:
- Multi-capability evaluation framework for memory (information extraction,
  multi-session reasoning, temporal reasoning, knowledge update, abstention).
- Evaluator design: test whether the full pipeline can answer questions that
  require going back through long conversation histories.
- Evidence-aware evaluation: ground-truth episode IDs per sample.

**Not** copied:
- We re-implement all indexing and retrieval from scratch.
- We benchmark retrieval separately from end-to-end.

### LoCoMo (Maharana et al. 2024)

Key ideas borrowed:
- Long-form multi-session conversations as a testbed for memory.
- Temporal and causal reasoning categories.
- Session-summary vs. raw-turn retrieval comparison.

**Not** copied:
- Adapters are written to load LoCoMo data into our generic `QASample` schema.

### GraphRAG (Edge et al. 2024)

Key ideas borrowed:
- Community reports as a coarse-grained index over large corpora (global queries).
- Local vs. global retrieval routing.
- Hierarchical traversal from community → member → raw document.

**Not** copied:
- We do not run GPT-4 for entity extraction or community summarisation.
- Community READMEs are built extractively (TF-IDF + entity regex).
- The graph is built deterministically; no API calls.

### Graphiti / Zep (Falkor et al. 2024)

Key ideas borrowed:
- Episode provenance: every node traces back to original conversation turns.
- Temporal validity: `valid_from`/`valid_to` + `SUPERSEDES` edges.
- Incremental graph construction as new conversations arrive.
- Hybrid lexical + semantic + graph retrieval.

**Not** copied:
- No LLM for edge extraction in the default pipeline.
- We build the graph once per benchmark run (batch mode).

### LongMemEval-V2

Key ideas acknowledged:
- Context-gathering pipeline (multi-step retrieval to build dense context).
- Compact evidence representation for token efficiency.
- Accuracy–latency Pareto frontier analysis.

**Status**: Adapter skeleton only. Not evaluated at 115M-token tier.

---

## Limitations

### Graph quality

- Entity extraction is regex-based. Named entity recognition is optional (spaCy
  adapter). Graph quality degrades on domains not covered by the default patterns.
- Semantic edge threshold (default 0.75) is tuned on synthetic data; may not
  generalise.
- Community quality depends on graph connectivity. Small corpora produce trivial
  communities.

### Community README

- READMEs are extractive summaries, not curated descriptions. They may drift from
  the actual content of the community as it grows.
- README provenance always links to source episodes; it is never treated as ground truth.

### Temporal handling

- Temporal ordering is heuristic (prefer newer by default, prefer older if query
  contains "originally"/"before"). It does not model fine-grained temporal reasoning.
- `valid_to` / `SUPERSEDES` edges require explicit update episodes in the corpus;
  they are not inferred automatically.

### Embedding similarity ≠ causal relation

- Cosine similarity between embeddings captures lexical/topical proximity, not
  causal or logical dependency. Multi-hop reasoning via semantic edges can introduce
  spurious paths.

### Small LLM compliance

- Qwen3-0.6B may not reliably follow the JSON schema or the abstention instruction.
  Schema failure rate is always reported; do not silently discard failures.

### Synthetic vs. real data

- The synthetic benchmark is designed to test specific capabilities in isolation.
  Strong performance on synthetic does not guarantee generalisation.
- Real datasets (LongMemEval, LoCoMo) should always be run before claiming
  production-level performance.

### Retrieval quality does not guarantee reader quality

- A retriever that achieves perfect Recall@10 provides no guarantee that the reader
  will use the evidence correctly. Always run `oracle → reader` as a ceiling check.

### Scale

- The default tiny/small synthetic datasets are suitable for rapid iteration.
  Production-scale claims require stress testing (≥10,000 episodes) and real datasets.

---

## Fairness checklist

Before publishing results, verify:

- [ ] Ground-truth labels are NOT indexed (only the raw episode text).
- [ ] Category labels are NOT passed to retrievers.
- [ ] Evidence labels are NOT used for reranking (oracle excluded).
- [ ] All methods use the same corpus, same top-k, same token budget.
- [ ] All methods using embeddings use the same model.
- [ ] All e2e methods use the same reader, prompt, and generation config.
- [ ] Failure samples are included in the denominator.
- [ ] Hyperparameters (threshold, RRF k) are tuned on a held-out dev split, not the eval split.
- [ ] Results include confidence intervals and win/loss/tie rates, not just means.
