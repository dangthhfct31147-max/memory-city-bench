"""Benchmark runner — orchestrates retrieval or e2e across methods, seeds, and datasets."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from memcity.datasets.protocol import QASample
from memcity.evaluation.metrics import aggregate_metrics, compute_all_metrics
from memcity.instrumentation.resources import (
    LatencyStats,
    collect_environment,
    get_process_rss_mb,
    get_vram_mb,
)
from memcity.retrieval.protocol import IndexStats, Retriever
from memcity.utils.helpers import run_id, seed_everything, write_jsonl

EVALUATION_PROTOCOL = "retrieval-episode-id-v1"


def _corpus_hash(corpus: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in corpus:
        digest.update(str(item.get("id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.get("text", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class RetrievalBenchmarkRunner:
    """Runs retrieval benchmarks across methods with warmup, seeds, and artifact writing."""

    def __init__(
        self,
        output_dir: str | Path = "runs",
        warmup: int = 5,
        top_ks: tuple[int, ...] = (1, 3, 5, 10),
        seeds: list[int] | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._warmup = warmup
        self._top_ks = top_ks
        self._seeds = seeds or [42]

    def run(
        self,
        retriever: Retriever,
        corpus: list[dict],
        samples: list[QASample],
        config: Any = None,
        dataset_name: str = "unknown",
        limit: int | None = None,
        resume: bool = True,
    ) -> dict:
        """Run full benchmark for one retriever over all seeds.

        Returns aggregated metrics dict.
        """
        rid = run_id()
        run_dir = self._output_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        env = collect_environment()
        env_path = run_dir / "environment.json"
        env_path.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")

        # Write resolved config
        cfg_path = run_dir / "config.resolved.yaml"
        cfg_path.write_text(
            f"method: {retriever.name}\ndataset: {dataset_name}\ntop_ks: {list(self._top_ks)}\n"
            f"warmup: {self._warmup}\nseeds: {self._seeds}\n",
            encoding="utf-8",
        )

        all_seed_metrics: list[dict] = []

        for seed in self._seeds:
            seed_everything(seed)
            seed_metrics = self._run_seed(
                retriever=retriever,
                corpus=corpus,
                samples=samples[:limit] if limit else samples,
                seed=seed,
                run_dir=run_dir,
                dataset_name=dataset_name,
                config=config,
            )
            all_seed_metrics.append(seed_metrics)

        # Aggregate across seeds
        final_metrics = self._aggregate_seeds(all_seed_metrics)
        final_metrics["run_id"] = rid
        final_metrics["method"] = retriever.name
        final_metrics["dataset"] = dataset_name
        final_metrics["dataset_hash"] = _corpus_hash(corpus)
        final_metrics["top_ks"] = list(self._top_ks)
        final_metrics["evaluation_protocol"] = EVALUATION_PROTOCOL
        final_metrics["seeds"] = self._seeds

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return final_metrics

    def _run_seed(
        self,
        retriever: Retriever,
        corpus: list[dict],
        samples: list[QASample],
        seed: int,
        run_dir: Path,
        dataset_name: str,
        config: Any,
    ) -> dict:
        # ── Build index ───────────────────────────────────────────────────────
        t_build_start = time.perf_counter()
        stats: IndexStats = retriever.build(corpus, config)
        build_time = time.perf_counter() - t_build_start

        # ── Warmup ────────────────────────────────────────────────────────────
        warmup_samples = samples[: min(self._warmup, len(samples))]
        for ws in warmup_samples:
            retriever.query(ws.query, top_k=max(self._top_ks))

        # ── Benchmark queries ─────────────────────────────────────────────────
        latency_stats = LatencyStats()
        per_query_metrics: list[dict] = []
        raw_results: list[dict] = []
        failures: list[dict] = []
        rss_samples: list[float] = []
        vram_samples: list[float] = []

        for sample in samples:
            try:
                t0 = time.perf_counter()
                result = retriever.query(sample.query, top_k=max(self._top_ks), trace=True)
                latency_ms = (time.perf_counter() - t0) * 1000
                latency_stats.record(latency_ms)
                rss_samples.append(get_process_rss_mb())
                vram_samples.append(get_vram_mb())

                retrieved_ids = result.episode_ids()
                relevant = set(sample.evidence_episode_ids)

                q_metrics = compute_all_metrics(retrieved_ids, relevant, ks=self._top_ks)
                q_metrics["sample_id"] = sample.sample_id
                q_metrics["category"] = sample.category.value
                q_metrics["latency_ms"] = round(latency_ms, 2)
                per_query_metrics.append(q_metrics)

                raw_results.append(
                    {
                        "sample_id": sample.sample_id,
                        "query": sample.query,
                        "ground_truth": list(relevant),
                        "retrieved": retrieved_ids[: max(self._top_ks)],
                        "category": sample.category.value,
                        "latency_ms": round(latency_ms, 2),
                        "trace": result.trace.model_dump() if result.trace else None,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "sample_id": sample.sample_id,
                        "error": str(exc),
                        "query": sample.query,
                    }
                )

        # ── Write artifacts ───────────────────────────────────────────────────
        write_jsonl(run_dir / f"retrieval_results_{seed}.jsonl", raw_results)
        write_jsonl(run_dir / f"failures_{seed}.jsonl", failures)

        # ── Compute aggregate metrics ─────────────────────────────────────────
        agg = aggregate_metrics(per_query_metrics)
        latency = latency_stats.to_dict()
        rss_peak = max(rss_samples) if rss_samples else 0.0
        vram_peak = max(vram_samples) if vram_samples else 0.0

        # Per-category breakdowns
        by_cat: dict[str, list[dict]] = {}
        for qm in per_query_metrics:
            cat = qm.get("category", "unknown")
            by_cat.setdefault(cat, []).append(qm)
        cat_metrics = {cat: aggregate_metrics(qms) for cat, qms in by_cat.items()}

        return {
            "seed": seed,
            "sample_count": len(samples),
            "build_wall_time_s": round(build_time, 3),
            "index_stats": stats.model_dump(),
            "latency": latency,
            "rss_peak_mb": round(rss_peak, 1),
            "vram_peak_mb": round(vram_peak, 1),
            "failure_count": len(failures),
            "metrics": agg,
            "metrics_by_category": cat_metrics,
        }

    def _aggregate_seeds(self, seed_results: list[dict]) -> dict:
        if not seed_results:
            return {}
        if len(seed_results) == 1:
            return seed_results[0]

        # Merge metrics across seeds (mean only for now)
        all_metrics = [r.get("metrics", {}) for r in seed_results]
        merged: dict[str, float] = {}
        for key in all_metrics[0]:
            vals = [m.get(key, 0.0) for m in all_metrics]
            merged[key] = sum(vals) / len(vals)

        # Use first seed's build/latency stats as representative
        result = dict(seed_results[0])
        result["metrics"] = merged
        result["seed_count"] = len(seed_results)
        return result
