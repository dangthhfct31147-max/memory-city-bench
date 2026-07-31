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


def print_evidence_loss_table(
    per_query_metrics: list[dict],
    method: str = "",
    top_n: int = 20,
    candidate_k: int = 50,
    final_k: int = 10,
) -> None:
    """Show the queries where evidence was reachable but did not survive to top-k.

    Loss = candidate_recall@candidate_k − final_recall@final_k. A large positive
    loss means the evidence was in the candidate union but the graph/temporal/
    rerank/dedup stages dropped it out of the returned top-k — exactly the
    failure mode to diagnose. Abstention queries (no evidence) are skipped.
    """
    cand_key = f"candidate_recall@{candidate_k}"
    final_key = f"final_recall@{final_k}"
    rows = []
    for qm in per_query_metrics:
        if cand_key not in qm or final_key not in qm:
            continue
        # Skip queries with no real evidence (recall is trivially 1.0 there).
        if qm.get("category") == "abstention":
            continue
        loss = qm[cand_key] - qm[final_key]
        if loss > 0:
            rows.append((loss, qm))
    rows.sort(key=lambda x: x[0], reverse=True)
    if not rows:
        console.print(
            f"[green]No evidence-loss cases for {method}: "
            f"nothing reachable was dropped before top-{final_k}.[/green]"
        )
        return

    title = f"WORST {min(top_n, len(rows))} EVIDENCE-LOSS QUERIES — {method}"
    table = Table(title=title, box=box.SIMPLE_HEAD)
    table.add_column("Sample", style="bold cyan", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column(f"cand@{candidate_k}", justify="right")
    table.add_column(f"final@{final_k}", justify="right")
    table.add_column("loss", justify="right", style="red")
    table.add_column("rerank", justify="right")
    table.add_column("temporal", justify="right")
    table.add_column("graphloss", justify="right")

    for loss, qm in rows[:top_n]:
        table.add_row(
            str(qm.get("sample_id", ""))[:28],
            str(qm.get("category", ""))[:12],
            f"{qm.get(cand_key, 0):.2f}",
            f"{qm.get(final_key, 0):.2f}",
            f"{loss:.2f}",
            f"{qm.get('evidence_lost_by_reranking', 0):.2f}",
            f"{qm.get('evidence_lost_by_temporal', 0):.2f}",
            f"{qm.get('graph_induced_loss', 0):.2f}",
        )
    console.print(table)


def print_diagnostics_table(results: list[dict], dataset_name: str = "") -> None:
    """Print aggregate pipeline diagnostics per method."""
    cols = [
        ("candidate_recall@50", "cand@50", ".3f"),
        ("final_recall@10", "final@10", ".3f"),
        ("graph_only_gain", "g+gain", ".3f"),
        ("graph_induced_loss", "g-loss", ".3f"),
        ("evidence_lost_by_reranking", "rerank-", ".3f"),
        ("evidence_lost_by_temporal", "temp-", ".3f"),
        ("duplicate_source_rate", "dup", ".3f"),
        ("raw_episode_ratio", "raw%", ".3f"),
        ("readme_ratio", "hub%", ".3f"),
    ]
    present = [c for c in cols if any(c[0] in r.get("metrics", {}) for r in results)]
    if not present:
        return
    table = Table(title=f"PIPELINE DIAGNOSTICS — {dataset_name}", box=box.SIMPLE_HEAD)
    table.add_column("Method", style="bold cyan", no_wrap=True)
    for _, name, _ in present:
        table.add_column(name, justify="right")
    for r in results:
        m = r.get("metrics", {})
        row = [r.get("method", r.get("name", "unknown"))]
        for key, _, fmt in present:
            val = m.get(key)
            row.append(f"{val:{fmt}}" if val is not None else "—")
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
