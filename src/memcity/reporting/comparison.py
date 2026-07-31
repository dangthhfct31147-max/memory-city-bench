"""Strict paired per-query comparison for retrieval benchmark runs."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from memcity.evaluation.metrics import bootstrap_ci, compute_all_metrics


def normalise_metric(metric: str) -> str:
    aliases = {
        "recall_at_": "recall@",
        "precision_at_": "precision@",
        "any_hit_at_": "any_hit@",
        "all_evidence_recall_at_": "all_evidence_recall@",
        "ndcg_at_": "ndcg@",
        "evidence_f1_at_": "evidence_f1@",
    }
    value = metric.strip().lower()
    for prefix, replacement in aliases.items():
        if value.startswith(prefix):
            return replacement + value.removeprefix(prefix)
    return value


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object in {path} at line {line_number}")
            records.append(record)
    return records


def _seed_from_path(path: Path) -> str:
    match = re.search(r"_(\d+)\.jsonl$", path.name)
    return match.group(1) if match else "unknown"


def _load_run(run_dir: Path, metric: str) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise ValueError(f"Missing metrics.json in {run_dir}")
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_metadata = ("dataset", "dataset_hash", "top_ks", "evaluation_protocol", "method")
    missing = [key for key in required_metadata if key not in metadata]
    if missing:
        raise ValueError(f"Run {run_dir.name} lacks compatibility metadata: {', '.join(missing)}")

    results: dict[str, float] = {}
    query_ids: set[str] = set()
    for path in sorted(run_dir.glob("retrieval_results_*.jsonl")):
        seed = _seed_from_path(path)
        for record in _read_jsonl_strict(path):
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"Missing sample_id in {path}")
            key = f"{seed}:{sample_id}"
            if key in query_ids:
                raise ValueError(f"Duplicate query ID {key} in {run_dir}")
            query_ids.add(key)
            ground_truth = set(str(value) for value in record.get("ground_truth", []))
            retrieved = [str(value) for value in record.get("retrieved", [])]
            k_values = tuple(int(k) for k in metadata["top_ks"])
            per_query = compute_all_metrics(retrieved, ground_truth, ks=k_values)
            if metric not in per_query:
                raise ValueError(
                    f"Metric '{metric}' is unavailable; choices include {sorted(per_query)}"
                )
            results[key] = float(per_query[metric])

    failures: dict[str, str] = {}
    failure_paths = sorted(run_dir.glob("failures_*.jsonl"))
    if not failure_paths and (run_dir / "failures.jsonl").exists():
        failure_paths = [run_dir / "failures.jsonl"]
    for path in failure_paths:
        seed = _seed_from_path(path)
        for record in _read_jsonl_strict(path):
            sample_id = str(record.get("sample_id", ""))
            key = f"{seed}:{sample_id}"
            if key in query_ids:
                raise ValueError(f"Query {key} is both successful and failed in {run_dir}")
            query_ids.add(key)
            failures[key] = str(record.get("error", "unknown error"))
    if not query_ids:
        raise ValueError(f"Run {run_dir.name} has no per-query artifacts")
    return {"metadata": metadata, "values": results, "failures": failures, "query_ids": query_ids}


def _two_sided_bootstrap_pvalue(deltas: list[float], n_resamples: int, seed: int) -> float:
    if not deltas:
        return 1.0
    rng = random.Random(seed)
    n = len(deltas)
    non_positive = non_negative = 0
    for _ in range(n_resamples):
        mean = sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        non_positive += mean <= 0
        non_negative += mean >= 0
    return min(1.0, 2 * min(non_positive + 1, non_negative + 1) / (n_resamples + 1))


def compare_runs(
    run_dir_1: Path,
    run_dir_2: Path,
    *,
    metric: str = "recall_at_10",
    bootstrap_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare run 2 against run 1; positive delta means run 2 wins."""
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    metric_name = normalise_metric(metric)
    first = _load_run(run_dir_1, metric_name)
    second = _load_run(run_dir_2, metric_name)

    for field in ("dataset", "dataset_hash", "top_ks", "evaluation_protocol"):
        if first["metadata"][field] != second["metadata"][field]:
            raise ValueError(
                f"Incompatible runs: {field} differs "
                f"({first['metadata'][field]!r} != {second['metadata'][field]!r})"
            )
    if first["query_ids"] != second["query_ids"]:
        missing_1 = sorted(second["query_ids"] - first["query_ids"])
        missing_2 = sorted(first["query_ids"] - second["query_ids"])
        raise ValueError(
            f"Query IDs do not match; missing from run 1={missing_1[:5]}, "
            f"missing from run 2={missing_2[:5]}"
        )

    valid_ids = sorted(set(first["values"]) & set(second["values"]))
    if not valid_ids:
        raise ValueError("No queries succeeded in both runs")
    values_1 = [first["values"][query_id] for query_id in valid_ids]
    values_2 = [second["values"][query_id] for query_id in valid_ids]
    deltas = [value_2 - value_1 for value_1, value_2 in zip(values_1, values_2, strict=True)]
    ci_low, ci_high = bootstrap_ci(
        deltas, n_resamples=bootstrap_resamples, ci=confidence, seed=seed
    )
    tolerance = 1e-12
    error_ids = sorted(first["query_ids"] - set(valid_ids))
    return {
        "run_id_1": run_dir_1.name,
        "run_id_2": run_dir_2.name,
        "method_1": first["metadata"]["method"],
        "method_2": second["metadata"]["method"],
        "dataset": first["metadata"]["dataset"],
        "dataset_hash": first["metadata"]["dataset_hash"],
        "evaluation_protocol": first["metadata"]["evaluation_protocol"],
        "top_ks": first["metadata"]["top_ks"],
        "metric": metric_name,
        "delta_definition": "run_2_minus_run_1",
        "mean_1": sum(values_1) / len(values_1),
        "mean_2": sum(values_2) / len(values_2),
        "mean_delta": sum(deltas) / len(deltas),
        "confidence": confidence,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value_two_sided": _two_sided_bootstrap_pvalue(deltas, bootstrap_resamples, seed),
        "win_count": sum(delta > tolerance for delta in deltas),
        "tie_count": sum(abs(delta) <= tolerance for delta in deltas),
        "loss_count": sum(delta < -tolerance for delta in deltas),
        "valid_query_count": len(valid_ids),
        "error_query_count": len(error_ids),
        "run_1_failure_count": len(first["failures"]),
        "run_2_failure_count": len(second["failures"]),
        "error_query_ids": error_ids,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
    }


def write_comparison(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    scalar_fields = [key for key, value in result.items() if not isinstance(value, (list, dict))]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerow({key: result[key] for key in scalar_fields})
    report = f"""# Paired bootstrap comparison

- Dataset: `{result["dataset"]}`
- Metric: `{result["metric"]}`
- Delta: run 2 minus run 1
- Run 1: `{result["run_id_1"]}` (`{result["method_1"]}`), mean `{result["mean_1"]:.6f}`
- Run 2: `{result["run_id_2"]}` (`{result["method_2"]}`), mean `{result["mean_2"]:.6f}`
- Mean delta: `{result["mean_delta"]:.6f}`
- {result["confidence"]:.1%} CI: `[{result["ci_low"]:.6f}, {result["ci_high"]:.6f}]`
- Two-sided bootstrap p-value: `{result["p_value_two_sided"]:.6f}`
- Win / tie / loss for run 2: `{result["win_count"]} / {result["tie_count"]} / {result["loss_count"]}`
- Paired valid queries: `{result["valid_query_count"]}`
- Queries with an error in either run: `{result["error_query_count"]}`
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
