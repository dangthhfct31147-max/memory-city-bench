"""Rich-based terminal reporting and artifact export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich import box


console = Console()


RETRIEVAL_DISPLAY_COLS = [
    ("recall@1", "R@1", ".3f"),
    ("recall@5", "R@5", ".3f"),
    ("recall@10", "R@10", ".3f"),
    ("mrr", "MRR", ".3f"),
    ("ndcg@10", "nDCG@10", ".3f"),
    ("all_evidence_recall@10", "AllEv@10", ".3f"),
]

LATENCY_DISPLAY_COLS = [
    ("p50_ms", "p50ms", ".1f"),
    ("p95_ms", "p95ms", ".1f"),
]

RESOURCE_DISPLAY_COLS = [
    ("rss_peak_mb", "RAM MB", ".0f"),
    ("vram_peak_mb", "VRAM MB", ".0f"),
]


def print_retrieval_table(
    results: list[dict],
    dataset_name: str = "",
    n_samples: int = 0,
) -> None:
    """Print a compact retrieval results table."""
    title = f"RETRIEVAL RESULTS — {dataset_name or 'unknown'}"
    if n_samples:
        title += f" — n={n_samples}"

    table = Table(title=title, box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("Method", style="bold cyan", no_wrap=True)
    for _, col_name, _ in RETRIEVAL_DISPLAY_COLS + LATENCY_DISPLAY_COLS + RESOURCE_DISPLAY_COLS:
        table.add_column(col_name, justify="right")

    for r in results:
        m = r.get("metrics", {})
        lat = r.get("latency", {})
        row = [r.get("method", r.get("name", "unknown"))]
        for key, _, fmt in RETRIEVAL_DISPLAY_COLS:
            val = m.get(key)
            row.append(f"{val:{fmt}}" if val is not None else "—")
        for key, _, fmt in LATENCY_DISPLAY_COLS:
            val = lat.get(key)
            row.append(f"{val:{fmt}}" if val is not None else "—")
        for key, _, fmt in RESOURCE_DISPLAY_COLS:
            val = r.get(key)
            row.append(f"{val:{fmt}}" if val is not None else "—")
        table.add_row(*row)

    console.print(table)


def print_category_table(
    results: list[dict],
    metric_key: str = "recall@10",
    dataset_name: str = "",
) -> None:
    """Print per-category metric breakdown."""
    all_cats: set[str] = set()
    for r in results:
        all_cats.update(r.get("metrics_by_category", {}).keys())

    if not all_cats:
        return

    cats = sorted(all_cats)
    table = Table(title=f"BY CATEGORY — {metric_key} — {dataset_name}", box=box.SIMPLE_HEAD)
    table.add_column("Method", style="bold cyan", no_wrap=True)
    for cat in cats:
        table.add_column(cat[:12], justify="right")

    for r in results:
        cat_m = r.get("metrics_by_category", {})
        method = r.get("method", r.get("name", "unknown"))
        row = [method]
        for cat in cats:
            val = cat_m.get(cat, {}).get(metric_key)
            row.append(f"{val:.3f}" if val is not None else "—")
        table.add_row(*row)

    console.print(table)


E2E_DISPLAY_COLS = [
    ("exact_match", "EM", ".3f"),
    ("token_f1", "TokF1", ".3f"),
    ("abstention_f1", "AbstF1", ".3f"),
    ("evidence_citation_precision", "CiteP", ".3f"),
    ("evidence_citation_recall", "CiteR", ".3f"),
    ("grounded_answer_rate", "Grnd", ".3f"),
    ("unsupported_answer_rate", "Unsup", ".3f"),
    ("json_schema_success_rate", "JSON", ".3f"),
    ("reader_latency_p50_ms", "p50ms", ".1f"),
    ("reader_latency_p95_ms", "p95ms", ".1f"),
]


def print_e2e_table(
    results: list[dict],
    dataset_name: str = "",
    n_samples: int = 0,
) -> None:
    """Print a compact end-to-end (reader) results table."""
    title = f"END-TO-END RESULTS — {dataset_name or 'unknown'}"
    if n_samples:
        title += f" — n={n_samples}"

    table = Table(title=title, box=box.SIMPLE_HEAD)
    table.add_column("Retriever / Reader", style="bold cyan", no_wrap=True)
    for _, col_name, _ in E2E_DISPLAY_COLS:
        table.add_column(col_name, justify="right")

    for r in results:
        row = [r.get("retriever", "unknown")]
        for key, _, fmt in E2E_DISPLAY_COLS:
            val = r.get(key)
            row.append(f"{val:{fmt}}" if val is not None else "N/A")
        table.add_row(*row)

    console.print(table)


def export_markdown(results: list[dict], dataset_name: str, output_path: Path) -> None:
    lines = [
        f"# Memory City Benchmark Report\n",
        f"**Dataset:** {dataset_name}\n",
        "## Retrieval Results\n",
        "| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 | AllEv@10 | p50ms | p95ms |",
        "|--------|-----|-----|------|-----|---------|----------|-------|-------|",
    ]
    for r in results:
        m = r.get("metrics", {})
        lat = r.get("latency", {})
        method = r.get("method", r.get("name", "unknown"))
        row_vals = [
            method,
            f"{m.get('recall@1', 0):.3f}",
            f"{m.get('recall@5', 0):.3f}",
            f"{m.get('recall@10', 0):.3f}",
            f"{m.get('mrr', 0):.3f}",
            f"{m.get('ndcg@10', 0):.3f}",
            f"{m.get('all_evidence_recall@10', 0):.3f}",
            f"{lat.get('p50_ms', 0):.1f}",
            f"{lat.get('p95_ms', 0):.1f}",
        ]
        lines.append("| " + " | ".join(row_vals) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_csv(results: list[dict], output_path: Path) -> None:
    if not results:
        return
    all_keys: list[str] = ["method"]
    sample_m = results[0].get("metrics", {})
    sample_lat = results[0].get("latency", {})
    all_keys += [f"metric_{k}" for k in sample_m]
    all_keys += [f"latency_{k}" for k in sample_lat]
    all_keys += ["rss_peak_mb", "vram_peak_mb", "failure_count", "build_wall_time_s"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            flat: dict[str, Any] = {"method": r.get("method", r.get("name", ""))}
            for k, v in r.get("metrics", {}).items():
                flat[f"metric_{k}"] = v
            for k, v in r.get("latency", {}).items():
                flat[f"latency_{k}"] = v
            flat["rss_peak_mb"] = r.get("rss_peak_mb", "")
            flat["vram_peak_mb"] = r.get("vram_peak_mb", "")
            flat["failure_count"] = r.get("failure_count", "")
            flat["build_wall_time_s"] = r.get("build_wall_time_s", "")
            writer.writerow(flat)
