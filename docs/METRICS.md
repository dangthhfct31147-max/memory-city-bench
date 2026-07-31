# Metrics Reference

## Retrieval Metrics

All retrieval metrics operate on **episode IDs**, not raw text.
A "hit" means the retrieved episode was listed in the ground-truth `evidence_episode_ids` for that sample.

### Core metrics

| Metric | Definition |
|--------|-----------|
| `Recall@k` | Fraction of required evidence IDs appearing in top-k |
| `Precision@k` | Fraction of top-k items that are relevant |
| `Any-hit@k` | 1 if ≥1 relevant item in top-k, 0 otherwise |
| `All-evidence recall@k` | Fraction of **all** required evidence found in top-k |
| `MRR` | Mean Reciprocal Rank of the first relevant item |
| `nDCG@k` | Binary-relevance normalised discounted cumulative gain |
| `Evidence F1@k` | Harmonic mean of Evidence Precision@k and Evidence Recall@k |
| `Mean rank of first relevant` | Average rank position of the first correct evidence |

### Temporal & provenance metrics

| Metric | Definition |
|--------|-----------|
| `Temporal current-fact accuracy` | For `fact_update` samples: was the current (non-stale) evidence ranked above stale evidence? |
| `Stale-fact retrieval rate` | Fraction of `fact_update` samples where stale evidence ranked in top-k |
| `Abstention retrieval precision` | For `abstention` samples: fraction of retrieved items that are irrelevant (should be high for good abstention) |
| `Multi-hop path coverage` | Fraction of multi-hop evidence chains fully recovered |
| `Community hit rate` | Fraction of `global_community` samples where ≥1 community member retrieved |
| `Provenance completeness` | Fraction of retrieved items with non-empty `source_episode_ids` |

### Do NOT call `Any-hit@5` "accuracy"

These are distinct concepts:
- `Any-hit@5` = at least one relevant item in top 5
- `All-evidence recall@5` = all required evidence in top 5
- `Evidence F1@5` = harmonic mean of precision and recall at 5

Reporting only one number hides failures.

## System Performance Metrics

| Metric | Description |
|--------|-----------|
| `build_wall_time_s` | Wall-clock time to build the full index |
| `p50_ms / p90_ms / p95_ms / p99_ms` | Query latency percentiles (warm, after warmup) |
| `qps` | Queries per second (warm) |
| `rss_peak_mb` | Peak process RSS during queries |
| `vram_peak_mb` | Peak VRAM during queries (requires `pynvml`) |
| `index_disk_size_bytes` | Serialised index size |
| `node_count / edge_count / community_count` | Graph structure summary |

Latency is measured **after** a configurable warmup (default 5 queries).
Model cold-start time is **not** included in query latency.

## End-to-End Metrics

Reported only in `memcity benchmark e2e`. Listed in a **separate table** from retrieval.

| Metric | Description |
|--------|-----------|
| `exact_match` | Case-normalised exact match of answer vs. ground truth |
| `token_f1` | Token-level F1 between predicted and reference answer |
| `abstention_precision` | TP / (TP + FP) for `INSUFFICIENT_EVIDENCE` predictions |
| `abstention_recall` | TP / (TP + FN) for `INSUFFICIENT_EVIDENCE` predictions |
| `abstention_f1` | Harmonic mean of abstention precision and recall |
| `evidence_citation_precision` | Fraction of cited episode IDs that are actually relevant |
| `evidence_citation_recall` | Fraction of relevant episodes that were cited |
| `grounded_answer_rate` | Fraction of non-abstentions with ≥1 cited evidence |
| `unsupported_answer_rate` | Fraction of non-abstentions with 0 cited evidence |
| `json_schema_success_rate` | Fraction of reader responses that parse as valid JSON |
| `reader_latency_p50_ms` | Reader inference latency p50 |
| `prompt_tokens / completion_tokens` | Token counts per sample |
| `peak_vram_mb` | Peak VRAM during reader inference |

The `oracle → reader` row sets an **upper bound**: if oracle evidence still gives
poor answers, the problem is the reader, not the retriever.
