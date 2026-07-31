"""Benchmark runner — orchestrates scope-isolated retrieval across methods and seeds.

Retrieval is scope-isolated: samples are grouped by ``corpus_scope_id`` and one
index is built per scope. A query therefore only ever sees episodes from its own
scope (its own LongMemEval haystack, or its own LoCoMo conversation). This is a
correctness requirement of both benchmarks, not a tuning knob.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from memcity.datasets.corpus import build_scope_corpus, group_samples_by_scope, scope_of
from memcity.datasets.protocol import QASample
from memcity.evaluation.diagnostics import compute_diagnostics
from memcity.evaluation.metrics import aggregate_metrics, compute_all_metrics
from memcity.instrumentation.resources import (
    LatencyStats,
    collect_environment,
    get_process_rss_mb,
    get_vram_mb,
)
from memcity.retrieval.protocol import IndexStats, Retriever
from memcity.utils.helpers import run_id, seed_everything, write_jsonl

EVALUATION_PROTOCOL = "retrieval-episode-id-v2-scoped"


def dataset_hash(samples: list[QASample], *, source_hash: str = "", extra: dict | None = None) -> str:
    """Strong dataset hash covering evidence protocol, not just corpus text.

    Includes sample ids, queries, evidence labels, scope ids, and per-scope
    episode ids/text so two runs are only judged comparable when they evaluate
    the identical question/evidence/scope protocol.
    """
    digest = hashlib.sha256()
    if source_hash:
        digest.update(b"source:")
        digest.update(source_hash.encode("utf-8"))
        digest.update(b"\n")
    if extra:
        digest.update(b"extra:")
        digest.update(json.dumps(extra, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    for sample in samples:
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(scope_of(sample).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.query.encode("utf-8"))
        digest.update(b"\0")
        digest.update("|".join(sorted(sample.evidence_episode_ids)).encode("utf-8"))
        digest.update(b"\0")
        for ep in sample.history:
            digest.update(ep.episode_id.encode("utf-8"))
            digest.update(b"\1")
            digest.update((f"{ep.user_text} {ep.assistant_text}").encode("utf-8"))
            digest.update(b"\2")
        digest.update(b"\n")
    return digest.hexdigest()


class RetrievalBenchmarkRunner:
    """Runs scope-isolated retrieval benchmarks with warmup, seeds, and artifacts."""

    def __init__(
        self,
        output_dir: str | Path = "runs",
        warmup: int = 5,
        top_ks: tuple[int, ...] = (1, 3, 5, 10),
        seeds: list[int] | None = None,
        candidate_ks: tuple[int, ...] = (20, 50, 100),
    ) -> None:
        self._output_dir = Path(output_dir)
        self._warmup = warmup
        self._top_ks = top_ks
        self._seeds = seeds or [42]
        self._candidate_ks = candidate_ks

    def run(
        self,
        retriever: Retriever | None = None,
        corpus: list[dict] | None = None,
        samples: list[QASample] | None = None,
        config: Any = None,
        dataset_name: str = "unknown",
        limit: int | None = None,
        resume: bool = True,
        retriever_factory: Callable[[], Retriever] | None = None,
        source_hash: str = "",
    ) -> dict:
        """Run full benchmark for one retriever over all seeds and scopes.

        ``retriever_factory`` is preferred: it builds a fresh retriever per scope.
        A single ``retriever`` instance is also accepted (rebuilt per scope) for
        backward compatibility. ``corpus`` is ignored when samples carry scopes;
        it is retained only for legacy call sites.
        """
        if samples is None:
            raise ValueError("samples is required")
        if retriever_factory is None:
            if retriever is None:
                raise ValueError("either retriever or retriever_factory is required")
            method_name = retriever.name

            def retriever_factory() -> Retriever:  # reuse the single instance
                return retriever
        else:
            method_name = retriever_factory().name

        eff_samples = samples[:limit] if limit else samples

        rid = run_id()
        run_dir = self._output_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        env = collect_environment()
        (run_dir / "environment.json").write_text(
            json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "config.resolved.yaml").write_text(
            f"method: {method_name}\ndataset: {dataset_name}\ntop_ks: {list(self._top_ks)}\n"
            f"candidate_ks: {list(self._candidate_ks)}\nwarmup: {self._warmup}\nseeds: {self._seeds}\n",
            encoding="utf-8",
        )

        all_seed_metrics: list[dict] = []
        for seed in self._seeds:
            seed_everything(seed)
            all_seed_metrics.append(
                self._run_seed(
                    retriever_factory=retriever_factory,
                    samples=eff_samples,
                    seed=seed,
                    run_dir=run_dir,
                    dataset_name=dataset_name,
                    config=config,
                )
            )

        final_metrics = self._aggregate_seeds(all_seed_metrics)
        final_metrics["run_id"] = rid
        final_metrics["method"] = method_name
        final_metrics["dataset"] = dataset_name
        final_metrics["dataset_hash"] = dataset_hash(
            eff_samples, source_hash=source_hash,
            extra={"top_ks": list(self._top_ks), "limit": limit},
        )
        final_metrics["top_ks"] = list(self._top_ks)
        final_metrics["candidate_ks"] = list(self._candidate_ks)
        final_metrics["evaluation_protocol"] = EVALUATION_PROTOCOL
        final_metrics["seeds"] = self._seeds

        (run_dir / "metrics.json").write_text(
            json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return final_metrics

    def _run_seed(
        self,
        retriever_factory: Callable[[], Retriever],
        samples: list[QASample],
        seed: int,
        run_dir: Path,
        dataset_name: str,
        config: Any,
    ) -> dict:
        scope_corpora = build_scope_corpus(samples)
        scoped_samples = group_samples_by_scope(samples)

        latency_stats = LatencyStats()
        per_query_metrics: list[dict] = []
        raw_results: list[dict] = []
        failures: list[dict] = []
        rss_samples: list[float] = []
        vram_samples: list[float] = []
        build_time_total = 0.0
        agg_index_stats: dict[str, int] = {
            "node_count": 0, "edge_count": 0, "community_count": 0, "embedding_count": 0,
        }
        cross_scope_events = 0
        cross_scope_total = 0

        for scope_id, scope_samples in scoped_samples.items():
            corpus = scope_corpora.get(scope_id, [])
            scope_ep_ids = {c["id"] for c in corpus}
            retriever = retriever_factory()

            t_build = time.perf_counter()
            stats: IndexStats = retriever.build(corpus, config)
            build_time_total += time.perf_counter() - t_build
            agg_index_stats["node_count"] += stats.node_count
            agg_index_stats["edge_count"] += stats.edge_count
            agg_index_stats["community_count"] += stats.community_count
            agg_index_stats["embedding_count"] += stats.embedding_count

            for ws in scope_samples[: min(self._warmup, len(scope_samples))]:
                retriever.query(ws.query, top_k=max(self._top_ks))

            for sample in scope_samples:
                try:
                    t0 = time.perf_counter()
                    result = retriever.query(sample.query, top_k=max(self._top_ks), trace=True)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    latency_stats.record(latency_ms)
                    rss_samples.append(get_process_rss_mb())
                    vram_samples.append(get_vram_mb())

                    retrieved_ids = result.episode_ids()
                    relevant = set(sample.evidence_episode_ids)

                    # Cross-scope leak check: every retrieved id must be in scope.
                    out_of_scope = [r for r in retrieved_ids if r not in scope_ep_ids]
                    cross_scope_total += 1
                    if out_of_scope:
                        cross_scope_events += 1

                    q_metrics = compute_all_metrics(retrieved_ids, relevant, ks=self._top_ks)
                    diag = compute_diagnostics(
                        result=result,
                        relevant=relevant,
                        scope_episode_ids=scope_ep_ids,
                        top_ks=self._top_ks,
                        candidate_ks=self._candidate_ks,
                    )
                    q_metrics.update(diag)
                    q_metrics["sample_id"] = sample.sample_id
                    q_metrics["category"] = sample.category.value
                    q_metrics["latency_ms"] = round(latency_ms, 2)
                    per_query_metrics.append(q_metrics)

                    raw_results.append(
                        {
                            "sample_id": sample.sample_id,
                            "scope_id": scope_id,
                            "query": sample.query,
                            "ground_truth": list(relevant),
                            "retrieved": retrieved_ids[: max(self._top_ks)],
                            "out_of_scope_retrieved": out_of_scope[:10],
                            "category": sample.category.value,
                            "latency_ms": round(latency_ms, 2),
                            "trace": result.trace.model_dump() if result.trace else None,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — record, keep in denominator
                    failures.append(
                        {"sample_id": sample.sample_id, "error": str(exc), "query": sample.query}
                    )
                    # Failed query scores zero on every retrieval metric but stays
                    # in the denominator so a crash-prone method cannot look good.
                    template = compute_all_metrics([], {"__none__"}, ks=self._top_ks)
                    zero = {k: 0.0 for k in template}
                    zero["sample_id"] = sample.sample_id
                    zero["category"] = sample.category.value
                    zero["failed"] = 1.0
                    per_query_metrics.append(zero)

            retriever.close()

        write_jsonl(run_dir / f"retrieval_results_{seed}.jsonl", raw_results)
        write_jsonl(run_dir / f"per_query_metrics_{seed}.jsonl", per_query_metrics)
        write_jsonl(run_dir / f"failures_{seed}.jsonl", failures)

        agg = aggregate_metrics(per_query_metrics)
        by_cat: dict[str, list[dict]] = {}
        for qm in per_query_metrics:
            by_cat.setdefault(qm.get("category", "unknown"), []).append(qm)
        cat_metrics = {cat: aggregate_metrics(qms) for cat, qms in by_cat.items()}

        return {
            "seed": seed,
            "sample_count": len(samples),
            "scope_count": len(scoped_samples),
            "build_wall_time_s": round(build_time_total, 3),
            "index_stats": {"method": "aggregate", **agg_index_stats},
            "latency": latency_stats.to_dict(),
            "rss_peak_mb": round(max(rss_samples) if rss_samples else 0.0, 1),
            "vram_peak_mb": round(max(vram_samples) if vram_samples else 0.0, 1),
            "failure_count": len(failures),
            "cross_scope_retrieval_rate": (
                cross_scope_events / cross_scope_total if cross_scope_total else 0.0
            ),
            "metrics": agg,
            "metrics_by_category": cat_metrics,
        }

    def _aggregate_seeds(self, seed_results: list[dict]) -> dict:
        if not seed_results:
            return {}
        if len(seed_results) == 1:
            return seed_results[0]
        all_metrics = [r.get("metrics", {}) for r in seed_results]
        merged: dict[str, float] = {}
        for key in all_metrics[0]:
            vals = [m.get(key, 0.0) for m in all_metrics]
            merged[key] = sum(vals) / len(vals)
        result = dict(seed_results[0])
        result["metrics"] = merged
        result["seed_count"] = len(seed_results)
        return result
