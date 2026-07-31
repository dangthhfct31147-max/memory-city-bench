"""Main Typer CLI for Memory City Benchmark."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# On legacy Windows consoles (cp1252) Rich cannot encode box-drawing/arrow
# glyphs. Force UTF-8 on stdout/stderr before anything is printed.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(_stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="memcity",
    help="Memory City Benchmark — evaluate long-term AI memory architectures.",
    no_args_is_help=True,
)
console = Console()

# Sub-apps
datasets_app = typer.Typer(help="Dataset management commands.")
benchmark_app = typer.Typer(help="Benchmark commands.")
report_app = typer.Typer(help="Report and comparison commands.")
index_app = typer.Typer(help="Index management commands.")

app.add_typer(datasets_app, name="datasets")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(report_app, name="report")
app.add_typer(index_app, name="index")


# ── doctor ───────────────────────────────────────────────────────────────────


@app.command()
def doctor() -> None:
    """Check environment health: Python, GPU, deps, disk, model endpoints."""

    from memcity.instrumentation.resources import collect_environment

    console.print(Panel("[bold green]Memory City — Doctor[/bold green]", expand=False))

    env = collect_environment()

    table = Table(box=box.SIMPLE_HEAD, show_header=False)
    table.add_column("Item", style="bold")
    table.add_column("Value")

    table.add_row("Python", env.get("python_version", "unknown").split()[0])
    table.add_row("Platform", env.get("platform", "unknown"))
    table.add_row("OS", env.get("os", "unknown"))
    table.add_row("CPU", env.get("cpu", "unknown"))
    table.add_row("CPU cores", str(env.get("cpu_count", "unknown")))
    table.add_row("RAM total", f"{env.get('ram_total_gb', 0):.1f} GB")
    table.add_row("RAM free", f"{env.get('ram_available_gb', 0):.1f} GB")

    gpus = env.get("gpus", [])
    if gpus:
        for g in gpus:
            table.add_row("GPU", g.get("name", "unknown"))
            table.add_row("VRAM total", f"{g.get('vram_total_gb', 0):.1f} GB")
    else:
        table.add_row("GPU", "[yellow]none detected[/yellow]")

    # CUDA availability
    try:
        import torch

        cuda = torch.cuda.is_available()
        table.add_row(
            "CUDA", "[green]available[/green]" if cuda else "[yellow]not available[/yellow]"
        )
    except ImportError:
        table.add_row("CUDA", "[dim]torch not installed[/dim]")

    # Optional packages
    opt_pkgs = ["sentence_transformers", "hnswlib", "pynvml", "spacy"]
    for pkg in opt_pkgs:
        try:
            import importlib

            importlib.import_module(pkg)
            table.add_row(f"[dim]{pkg}[/dim]", "[green]installed[/green]")
        except ImportError:
            table.add_row(f"[dim]{pkg}[/dim]", "[yellow]not installed[/yellow]")

    # Disk free
    try:
        import shutil

        stat = shutil.disk_usage(".")
        free_gb = stat.free / 1024**3
        table.add_row("Disk free", f"{free_gb:.1f} GB")
    except Exception:
        pass

    # Dataset cache
    data_path = Path("data")
    if data_path.exists():
        datasets_found = list(data_path.iterdir())
        table.add_row("Dataset cache", f"{len(datasets_found)} folder(s) in data/")
    else:
        table.add_row("Dataset cache", "[dim]data/ not found[/dim]")

    console.print(table)
    console.print("[bold green]Doctor complete.[/bold green]")


# ── datasets ─────────────────────────────────────────────────────────────────


@datasets_app.command("list")
def datasets_list() -> None:
    """List available datasets and cached data."""
    from memcity.datasets.synthetic import SCALES

    console.print("[bold]Built-in datasets:[/bold]")
    for scale in SCALES:
        console.print(f"  synthetic-{scale}")
    external = {
        "longmemeval-s": Path("data/longmemeval/longmemeval_s_cleaned.json"),
        "locomo": Path("data/locomo/locomo10.json"),
    }
    for name, path in external.items():
        status = (
            f"[green]cached ({path.stat().st_size / 1024**2:.1f} MiB)[/green]"
            if path.exists()
            else "[dim]requires fetch[/dim]"
        )
        console.print(f"  {name:<16} {status}")


@datasets_app.command("generate")
def datasets_generate(
    scale: str = typer.Option(
        "tiny", "--scale", "-s", help="Dataset scale: tiny/small/medium/stress"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    data_dir: str = typer.Option("data", "--data-dir"),
) -> None:
    """Generate the synthetic benchmark dataset."""
    from memcity.datasets.synthetic import SyntheticDataset

    console.print(f"[bold]Generating synthetic-{scale} (seed={seed})…[/bold]")
    ds = SyntheticDataset(scale=scale, seed=seed)
    samples = ds.load(Path(data_dir))
    m = ds.manifest()
    console.print(f"[green]Generated {len(samples)} samples.[/green]")
    console.print(f"Categories: {m.category_counts}")


@datasets_app.command("validate")
def datasets_validate(
    name: str = typer.Argument(..., help="Dataset name (e.g. synthetic-tiny)"),
    data_dir: str = typer.Option("data", "--data-dir"),
    limit: int | None = typer.Option(None, "--limit", help="Validate only the first N samples"),
) -> None:
    """Validate a dataset for required fields."""
    from memcity.datasets.loader import load_dataset

    try:
        ds, samples = load_dataset(name, data_dir, limit=limit)
        errors = ds.validate(samples)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if errors:
        console.print(f"[red]Validation errors ({len(errors)}):[/red]")
        for error in errors[:20]:
            console.print(f"  {error}")
        raise typer.Exit(1)
    manifest = ds.manifest()
    console.print(f"[green]All {len(samples)} samples valid.[/green]")
    console.print(f"Categories: {manifest.category_counts}")


@datasets_app.command("fetch")
def datasets_fetch(
    name: str = typer.Argument(..., help="Dataset to fetch (longmemeval, locomo)"),
    variant: str = typer.Option("s", "--variant", "-v"),
    data_dir: str = typer.Option("data", "--data-dir"),
) -> None:
    """Fetch and atomically cache an official external dataset."""
    from memcity.datasets.loader import fetch_dataset

    console.print(f"[bold]Fetching {name} from its official source...[/bold]")
    try:
        path = fetch_dataset(name, data_dir, variant=variant)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Fetch failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Cached {path} ({path.stat().st_size / 1024**2:.1f} MiB)[/green]")


# ── index ─────────────────────────────────────────────────────────────────────


@index_app.command("build")
def index_build(
    dataset: str = typer.Option("synthetic-tiny", "--dataset", "-d"),
    method: str = typer.Option("memory_city_full", "--method", "-m"),
    config: str | None = typer.Option(None, "--config", "-c"),
    data_dir: str = typer.Option("data", "--data-dir"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Build an index for a dataset and method."""
    from memcity.datasets.loader import load_dataset
    from memcity.retrieval.registry import get_retriever

    _, samples = load_dataset(dataset, data_dir, seed=seed)

    # Build corpus from episodes
    corpus = _samples_to_corpus(samples)
    retriever = get_retriever(method)

    console.print(f"[bold]Building {method} index for {dataset}…[/bold]")
    t0 = time.perf_counter()
    stats = retriever.build(corpus)
    elapsed = time.perf_counter() - t0
    console.print(f"[green]Index built in {elapsed:.2f}s[/green]")
    console.print(
        f"  Nodes: {stats.node_count}  Edges: {stats.edge_count}  "
        f"Embeddings: {stats.embedding_count}"
    )


# ── benchmark ─────────────────────────────────────────────────────────────────


@benchmark_app.command("retrieval")
def benchmark_retrieval(
    dataset: str = typer.Option("synthetic-tiny", "--dataset", "-d"),
    methods: str = typer.Option("bm25,vector,hybrid_rrf,memory_city_full", "--methods"),
    top_k: str = typer.Option("1,3,5,10", "--top-k"),
    seeds: str = typer.Option("42", "--seeds"),
    warmup: int = typer.Option(5, "--warmup"),
    limit: int | None = typer.Option(None, "--limit"),
    output_dir: str = typer.Option("runs", "--output"),
    data_dir: str = typer.Option("data", "--data-dir"),
) -> None:
    """Run retrieval benchmark across methods."""
    from memcity.datasets.loader import load_dataset
    from memcity.evaluation.runner import RetrievalBenchmarkRunner
    from memcity.reporting.tables import (
        print_category_table,
        print_diagnostics_table,
        print_evidence_loss_table,
        print_retrieval_table,
    )
    from memcity.retrieval.registry import get_retriever_factory

    seed_val = int(seeds.split(",")[0])
    ds, samples = load_dataset(dataset, data_dir, seed=seed_val, limit=limit)
    source_hash = ds.manifest().source_hash

    top_ks = tuple(int(k) for k in top_k.split(","))
    seed_list = [int(s) for s in seeds.split(",")]
    method_list = [m.strip() for m in methods.split(",")]

    runner = RetrievalBenchmarkRunner(
        output_dir=output_dir,
        warmup=warmup,
        top_ks=top_ks,
        seeds=seed_list,
    )

    all_results: list[dict] = []
    for method in method_list:
        try:
            factory = get_retriever_factory(method)
            console.print(f"\n[bold cyan]Running {method}…[/bold cyan]")
            result = runner.run(
                retriever_factory=factory,
                samples=samples,
                dataset_name=dataset,
                source_hash=source_hash,
            )
            result["method"] = method
            all_results.append(result)
            xsr = result.get("cross_scope_retrieval_rate", 0.0)
            if xsr:
                console.print(
                    f"[red]cross_scope_retrieval_rate={xsr:.3f} for {method} "
                    f"(should be 0 — scope leak!)[/red]"
                )
        except Exception as exc:
            console.print(f"[red]Error in {method}: {exc}[/red]")
            import traceback

            traceback.print_exc()

    if all_results:
        print_retrieval_table(all_results, dataset_name=dataset, n_samples=len(samples))
        print_category_table(all_results, dataset_name=dataset)
        print_diagnostics_table(all_results, dataset_name=dataset)

        # Worst-20 evidence-loss table for the richest method available.
        loss_method = next(
            (m for m in ("memory_city_full", "hybrid_graph", "hybrid_temporal") if m in method_list),
            None,
        )
        if loss_method:
            loss_result = next(r for r in all_results if r["method"] == loss_method)
            rid = loss_result.get("run_id")
            seed0 = seed_list[0]
            pq_path = Path(output_dir) / rid / f"per_query_metrics_{seed0}.jsonl"
            if pq_path.exists():
                per_query = [
                    json.loads(line) for line in pq_path.read_text(encoding="utf-8").splitlines() if line
                ]
                print_evidence_loss_table(per_query, method=loss_method)


@benchmark_app.command("ablation")
def benchmark_ablation(
    dataset: str = typer.Option("synthetic-tiny", "--dataset", "-d"),
    base: str = typer.Option("memory_city_full", "--base"),
    seeds: str = typer.Option("42", "--seeds"),
    remove: str = typer.Option("", "--remove", help="Comma-separated components to remove"),
    data_dir: str = typer.Option("data", "--data-dir"),
    output_dir: str = typer.Option("runs", "--output"),
) -> None:
    """Run a real ablation ladder for Memory City.

    Each variant toggles exactly one component so the delta between adjacent
    rows attributes gain/loss to that component. ``--remove`` optionally drops
    named variants from the ladder.
    """
    from memcity.retrieval.registry import ablation_methods as _ladder

    console.print("[bold]Ablation study[/bold] — one component per step.")
    ladder = _ladder()
    removed = {r.strip() for r in remove.split(",") if r.strip()}
    if removed:
        ladder = [m for m in ladder if m not in removed]
        console.print(f"[dim]Removed from ladder: {sorted(removed)}[/dim]")
    if base and base not in ladder:
        ladder.append(base)

    benchmark_retrieval(
        dataset=dataset,
        methods=",".join(ladder),
        top_k="1,3,5,10",
        seeds=seeds,
        warmup=3,
        limit=None,
        output_dir=output_dir,
        data_dir=data_dir,
    )


@benchmark_app.command("e2e")
def benchmark_e2e(
    dataset: str = typer.Option("synthetic-tiny", "--dataset", "-d"),
    retrievers: str = typer.Option("bm25,memory_city_full,oracle", "--retrievers"),
    reader_provider: str = typer.Option(
        "oracle", "--reader-provider", help="oracle | openai-compatible"
    ),
    reader_base_url: str = typer.Option("http://127.0.0.1:8080/v1", "--reader-base-url"),
    reader_model: str = typer.Option("qwen3-0.6b", "--reader-model"),
    context_token_budget: int = typer.Option(4096, "--context-token-budget"),
    top_k: int = typer.Option(5, "--top-k"),
    limit: int | None = typer.Option(None, "--limit"),
    seed: int = typer.Option(42, "--seed"),
    data_dir: str = typer.Option("data", "--data-dir"),
    output_dir: str = typer.Option("runs", "--output"),
) -> None:
    """Run end-to-end (retriever → reader LLM) benchmark with scope isolation.

    Retrieval is scope-isolated: one index is built per corpus scope (one per
    LoCoMo conversation, one per LongMemEval haystack) — identical to the
    retrieval benchmark. The oracle also looks up evidence from the correct
    scope so cross-scope contamination is zero by construction.
    """
    from memcity.datasets.corpus import build_scope_corpus, group_samples_by_scope
    from memcity.datasets.loader import load_dataset
    from memcity.evaluation.e2e_metrics import compute_e2e_metrics
    from memcity.instrumentation.resources import collect_environment, get_vram_mb
    from memcity.readers.reader import OpenAICompatibleReader, OracleReader
    from memcity.retrieval.registry import get_retriever_factory
    from memcity.utils.helpers import run_id as make_run_id
    from memcity.utils.helpers import write_jsonl

    _, samples = load_dataset(dataset, data_dir, seed=seed, limit=limit)

    retriever_list = [r.strip() for r in retrievers.split(",")]

    if reader_provider in ("oracle", "none"):
        reader = OracleReader()
        use_oracle = True
        console.print(
            "[yellow]Using oracle reader (no LLM). "
            "Pass --reader-provider openai-compatible for a real reader.[/yellow]"
        )
    elif reader_provider == "openai-compatible":
        reader = OpenAICompatibleReader(
            base_url=reader_base_url,
            model=reader_model,
            context_token_budget=context_token_budget,
            cache_path=Path(data_dir) / "reader_cache.jsonl",
        )
        use_oracle = False
    else:
        console.print(f"[red]Unknown reader provider: {reader_provider}[/red]")
        raise typer.Exit(1)

    # Pre-build scope data structures once (shared across retriever variants).
    scope_corpora = build_scope_corpus(samples)
    scoped_samples = group_samples_by_scope(samples)
    # Flat lookup for oracle evidence (scoped: episode id → corpus item per scope).
    scope_corpus_maps: dict[str, dict[str, dict]] = {
        scope_id: {c["id"]: c for c in corpus}
        for scope_id, corpus in scope_corpora.items()
    }
    # Map each sample_id to its scope for fast lookup during evaluation.
    sample_to_scope: dict[str, str] = {
        sample.sample_id: scope_id
        for scope_id, scope_samples_list in scoped_samples.items()
        for sample in scope_samples_list
    }

    all_tables: list[dict] = []
    for method in retriever_list:
        console.print(f"\n[bold cyan]E2E: {method} -> {reader_provider}...[/bold cyan]")

        retriever_factory = get_retriever_factory(method)
        reader_results: list[dict] = []
        vram_samples_list: list[float] = []

        try:
            # Build one scope-isolated index per scope (mirror of retrieval runner).
            scope_retrievers: dict[str, object] = {}
            for scope_id, corpus in scope_corpora.items():
                r = retriever_factory()
                r.build(corpus)
                scope_retrievers[scope_id] = r

            for sample in samples:
                scope_id = sample_to_scope.get(sample.sample_id, "")
                scope_retriever = scope_retrievers.get(scope_id)
                scope_corpus_map = scope_corpus_maps.get(scope_id, {})

                if method == "oracle":
                    evidence = [
                        scope_corpus_map[evidence_id]
                        for evidence_id in sample.evidence_episode_ids
                        if evidence_id in scope_corpus_map
                    ]
                elif scope_retriever is not None:
                    result = scope_retriever.query(sample.query, top_k=top_k)
                    evidence = [
                        scope_corpus_map[item.id]
                        for item in result.items
                        if item.id in scope_corpus_map
                    ]
                else:
                    evidence = []

                if use_oracle:
                    ev_ids = sample.evidence_episode_ids
                    oracle_ev = [scope_corpus_map[e] for e in ev_ids if e in scope_corpus_map]
                    rr = reader.answer(
                        sample_id=sample.sample_id,
                        query=sample.query,
                        evidence=oracle_ev or evidence,
                        retriever_name=method,
                        ground_truth_answer=sample.answer,
                        evidence_ids=ev_ids,
                    )
                else:
                    rr = reader.answer(
                        sample_id=sample.sample_id,
                        query=sample.query,
                        evidence=evidence,
                        retriever_name=method,
                    )
                reader_results.append(rr.model_dump())
                vram_samples_list.append(get_vram_mb())

            for r in scope_retrievers.values():
                r.close()

        except Exception as exc:
            console.print(f"[red]Skipping {method}: {exc}[/red]")
            import traceback
            traceback.print_exc()
            continue

        metrics = compute_e2e_metrics(reader_results, samples)
        metrics["retriever"] = method
        metrics["dataset"] = dataset
        metrics["reader_provider"] = reader_provider
        metrics["reader_model"] = reader_model if not use_oracle else "oracle"
        metrics["context_token_budget"] = context_token_budget
        metrics["top_k"] = top_k
        metrics["peak_vram_mb"] = max(vram_samples_list) if vram_samples_list else 0.0
        all_tables.append(metrics)

        rid = make_run_id()
        run_path = Path(output_dir) / f"e2e_{method}_{rid}"
        run_path.mkdir(parents=True, exist_ok=True)
        write_jsonl(run_path / "reader_results.jsonl", reader_results)
        (run_path / "environment.json").write_text(
            json.dumps(collect_environment(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_path / "config.resolved.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "retriever": method,
                    "reader_provider": reader_provider,
                    "reader_base_url": reader_base_url,
                    "reader_model": reader_model,
                    "context_token_budget": context_token_budget,
                    "top_k": top_k,
                    "limit": limit,
                    "seed": seed,
                    "scope_count": len(scope_corpora),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if all_tables:
        from memcity.reporting.tables import print_e2e_table

        print_e2e_table(all_tables, dataset_name=dataset, n_samples=len(samples))


# ── report ────────────────────────────────────────────────────────────────────


@report_app.command("show")
def report_show(run_id: str = typer.Argument(...)) -> None:
    """Show metrics for a run."""
    run_path = Path("runs") / run_id / "metrics.json"
    if not run_path.exists():
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)
    data = json.loads(run_path.read_text(encoding="utf-8"))
    console.print_json(json.dumps(data, indent=2))


@report_app.command("leaderboard")
def report_leaderboard(
    dataset: str = typer.Option("synthetic-tiny", "--dataset", "-d"),
    runs_dir: str = typer.Option("runs", "--runs-dir"),
) -> None:
    """Show leaderboard across all runs for a dataset."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        console.print("[yellow]No runs directory found.[/yellow]")
        return

    results = []
    for run_dir in sorted(runs_path.iterdir()):
        m_path = run_dir / "metrics.json"
        if m_path.exists():
            data = json.loads(m_path.read_text(encoding="utf-8"))
            if data.get("dataset") == dataset:
                results.append(data)

    if not results:
        console.print(f"[yellow]No runs found for dataset: {dataset}[/yellow]")
        return

    from memcity.reporting.tables import print_retrieval_table

    print_retrieval_table(results, dataset_name=dataset)


@report_app.command("compare")
def report_compare(
    run_id_1: str = typer.Argument(..., help="Baseline run ID"),
    run_id_2: str = typer.Argument(..., help="Candidate run ID"),
    metric: str = typer.Option("recall_at_10", "--metric"),
    bootstrap_resamples: int = typer.Option(10_000, "--bootstrap-resamples", min=1),
    confidence: float = typer.Option(0.95, "--confidence", min=0.001, max=0.999),
    seed: int = typer.Option(42, "--seed"),
    runs_dir: str = typer.Option("runs", "--runs-dir"),
    output_dir: str | None = typer.Option(None, "--output"),
) -> None:
    """Compare run 2 against run 1 with a paired per-query bootstrap."""
    from memcity.reporting.comparison import compare_runs, write_comparison

    root = Path(runs_dir)
    first = root / run_id_1
    second = root / run_id_2
    if not first.is_dir() or not second.is_dir():
        console.print(f"[red]Run not found: {first if not first.is_dir() else second}[/red]")
        raise typer.Exit(1)
    destination = (
        Path(output_dir)
        if output_dir
        else root / "comparisons" / f"{run_id_1}_vs_{run_id_2}_{metric}"
    )
    try:
        result = compare_runs(
            first,
            second,
            metric=metric,
            bootstrap_resamples=bootstrap_resamples,
            confidence=confidence,
            seed=seed,
        )
        write_comparison(result, destination)
    except ValueError as exc:
        console.print(f"[red]Comparison failed: {exc}[/red]")
        raise typer.Exit(1) from exc

    table = Table(title="Paired bootstrap comparison", box=box.SIMPLE_HEAD)
    table.add_column("Metric")
    table.add_column(result["method_1"], justify="right")
    table.add_column(result["method_2"], justify="right")
    table.add_column("Delta (2 - 1)", justify="right")
    table.add_column(f"{confidence:.0%} CI", justify="right")
    table.add_row(
        result["metric"],
        f"{result['mean_1']:.4f}",
        f"{result['mean_2']:.4f}",
        f"{result['mean_delta']:+.4f}",
        f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]",
    )
    console.print(table)
    console.print(
        f"Win/tie/loss: {result['win_count']}/{result['tie_count']}/{result['loss_count']}  "
        f"valid={result['valid_query_count']} errors={result['error_query_count']}"
    )
    console.print(f"[green]Wrote comparison artifacts to {destination}[/green]")


@report_app.command("export")
def report_export(
    run_id: str = typer.Argument(...),
    format: str = typer.Option("markdown,json", "--format"),
    runs_dir: str = typer.Option("runs", "--runs-dir"),
) -> None:
    """Export run results to markdown/CSV/JSON."""
    run_path = Path(runs_dir) / run_id
    if not run_path.exists():
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)
    m_path = run_path / "metrics.json"
    data = json.loads(m_path.read_text(encoding="utf-8"))
    fmts = [f.strip() for f in format.split(",")]
    if "json" in fmts:
        out = run_path / "export.json"
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]Exported JSON: {out}[/green]")
    if "markdown" in fmts:
        from memcity.reporting.tables import export_markdown

        out = run_path / "report.md"
        export_markdown([data], dataset_name=data.get("dataset", ""), output_path=out)
        console.print(f"[green]Exported Markdown: {out}[/green]")
    if "csv" in fmts:
        from memcity.reporting.tables import export_csv

        out = run_path / "comparison.csv"
        export_csv([data], output_path=out)
        console.print(f"[green]Exported CSV: {out}[/green]")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _samples_to_corpus(samples: list) -> list[dict]:
    """Convert QASamples to a flat corpus of episode dicts for retrievers."""
    seen: set[str] = set()
    corpus: list[dict] = []
    for sample in samples:
        for ep in sample.history:
            if ep.episode_id not in seen:
                seen.add(ep.episode_id)
                corpus.append(
                    {
                        "id": ep.episode_id,
                        "node_type": "episode",
                        "text": f"{ep.user_text} {ep.assistant_text}".strip(),
                        "user_text": ep.user_text,
                        "assistant_text": ep.assistant_text,
                        "session_id": ep.session_id,
                        "timestamp": ep.timestamp,
                        "source_episode_ids": [ep.episode_id],
                    }
                )
    return corpus
