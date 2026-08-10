"""Typer CLI — commands for agents and humans.

Socbench — 'The unexamined dataset is not worth training on.'
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

os.environ.setdefault("TY_DISABLE_SHOW_LOCALS", "1")

# Force UTF-8 output so Unicode (→, ★, ✓) renders on Windows consoles.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = typer.Typer(
    name="socbench",
    help="Socbench — Scientific dataset intelligence. 'The unexamined dataset is not worth training on.'",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()
banner_console = Console(stderr=True)

SOCBENCH_BANNER = r"""
  ___  ___   ___ ___  ___ _  _  ___ _  _
 / __|/ _ \ / __| _ )| __| \| |/ __| || |
 \__ \ (_) | (__| _ \| _|| .` | (__| __ |
 |___/\___/ \___|___/|___|_|\_|\___|_||_|
""".strip("\n")


class OutputFormat(str, Enum):
    table = "table"
    json = "json"


class LeaderboardSort(str, Enum):
    combined = "combined"
    auto = "auto"
    quality = "quality"
    diversity = "diversity"
    utility = "utility"
    training = "training"


@app.callback()
def cli_root(
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Hide the startup banner; useful for captured terminal output.",
        envvar="SOCBENCH_NO_BANNER",
    ),
):
    """Scientific dataset intelligence from discovery through training evidence."""
    if not no_banner:
        banner_console.print(SOCBENCH_BANNER, style="bold cyan", highlight=False)


async def _prepare_database() -> None:
    """Create tables and hydrate an empty database from canonical evidence."""
    from socbench.bootstrap import bootstrap_from_canonical
    from socbench.db import init_db

    await init_db()
    await bootstrap_from_canonical()


async def _dispose_database() -> None:
    from socbench.db import engine

    await engine.dispose()


def _score_text(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}"


def _render_dataset_evidence(evidence, *, decision_only: bool = False) -> None:
    from socbench.categories import CATEGORIES
    from socbench.dataset_intelligence import SCORE_DIMENSIONS, assess_readiness

    category = CATEGORIES.get(evidence.category)
    category_label = category.label if category else evidence.category or "Unclassified"
    console.print(
        Panel(
            f"[bold]{evidence.dataset_id}[/bold]\n"
            f"Category: {category_label}\n"
            f"Evidence source: {evidence.source} | Last scored: {evidence.last_scored or '-'}",
            title="Socbench Dataset Evidence",
        )
    )

    if not decision_only:
        table = Table(title="Stored scoring dimensions (0-100)")
        table.add_column("Dimension", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Evidence", max_width=54)
        for dimension in SCORE_DIMENSIONS:
            item = evidence.dimensions[dimension]
            details = ", ".join(
                f"{key}={value}" for key, value in list(item.details.items())[:3]
            )
            table.add_row(
                dimension.replace("_", " ").title(),
                _score_text(item.score),
                details or ("missing" if item.score is None else "stored"),
            )
        table.add_row("Automated score", _score_text(evidence.auto_score), "Mean of seven dimensions")
        console.print(table)

        risk = Table(title="Risk and training evidence")
        risk.add_column("Signal", style="bold")
        risk.add_column("Value", justify="right")
        risk.add_row("Max benchmark overlap", _score_text(evidence.contamination_rate))
        risk.add_row(
            "Estimated repetition",
            "-" if evidence.repetition_pct is None else f"{evidence.repetition_pct:.1f}%",
        )
        risk.add_row("License", evidence.license or "missing")
        risk.add_row("Latest proxy-training outcome", evidence.training_outcome or "not run")
        risk.add_row("Proxy-training score", _score_text(evidence.training_score))
        console.print(risk)

    readiness = assess_readiness(evidence)
    color = {
        "PROCEED": "green",
        "REVIEW": "yellow",
        "DO NOT PROCEED": "red",
    }[readiness.decision]
    reasons = "\n".join(f"- {reason}" for reason in readiness.reasons)
    console.print(
        Panel(
            f"[{color}][bold]{readiness.decision}[/bold][/{color}]\n\n"
            f"{reasons}\n\nNext: {readiness.next_step}",
            title="Training-readiness decision",
        )
    )


async def _load_dataset_evidence(
    dataset_id: str,
    *,
    sample_size: int,
    refresh: bool,
    cached_only: bool,
    token: str | None,
):
    from socbench.dataset_intelligence import get_or_score_dataset

    await _prepare_database()
    try:
        return await get_or_score_dataset(
            dataset_id,
            sample_size=sample_size,
            refresh=refresh,
            cached_only=cached_only,
            token=token,
        )
    finally:
        await _dispose_database()


def _run_dataset_command(
    dataset_id: str,
    *,
    sample_size: int,
    refresh: bool,
    cached_only: bool,
    token: str | None,
    output_format: OutputFormat,
    decision_only: bool = False,
) -> None:
    from socbench.dataset_intelligence import DatasetIntelligenceError

    try:
        evidence = asyncio.run(
            _load_dataset_evidence(
                dataset_id,
                sample_size=sample_size,
                refresh=refresh,
                cached_only=cached_only,
                token=token,
            )
        )
    except DatasetIntelligenceError as exc:
        banner_console.print(f"[red]Dataset assessment failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    if output_format is OutputFormat.json:
        typer.echo(json.dumps(evidence.to_dict(), indent=2, sort_keys=True))
    else:
        _render_dataset_evidence(evidence, decision_only=decision_only)


@app.command()
def discover(
    search: str = typer.Option("", help="Search query"),
    limit: int = typer.Option(50, help="Max datasets to find"),
    days: Optional[int] = typer.Option(None, help="Only datasets from last N days"),
):
    """Discover new/trending datasets from HuggingFace."""
    from socbench.discovery.scanner import scan_datasets

    async def _run():
        results = await scan_datasets(search=search, limit=limit, days=days)
        table = Table(title=f"Discovered {len(results)} datasets")
        table.add_column("HF ID", style="cyan")
        table.add_column("Downloads", justify="right")
        table.add_column("Likes", justify="right")
        table.add_column("Trending", justify="right")
        table.add_column("Category", style="dim")
        for ds in results:
            from socbench.categories import CATEGORIES, classify_dataset
            cat = classify_dataset(ds.tags or [], ds.description or "", dataset_id=ds.hf_id)
            cat_label = CATEGORIES.get(cat, CATEGORIES["pretraining-web"]).label
            table.add_row(
                ds.hf_id,
                f"{ds.downloads:,}" if ds.downloads else "0",
                f"{ds.likes}" if ds.likes else "0",
                f"{ds.trending_score:.1f}" if ds.trending_score else "0",
                cat_label,
            )
        console.print(table)

    asyncio.run(_run())


@app.command()
def assess(
    dataset_id: str = typer.Argument(..., help="HuggingFace dataset ID"),
    sample_size: int = typer.Option(10_000, min=100, max=100_000, help="Number of samples to analyze on a cache miss"),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore complete cached evidence and score again"),
    cached_only: bool = typer.Option(False, "--cached-only", help="Never use the network; fail if complete evidence is absent"),
    token: Optional[str] = typer.Option(None, "--token", envvar="HF_TOKEN", help="Optional token for gated datasets"),
    output_format: OutputFormat = typer.Option(OutputFormat.table, "--format", help="Output format"),
):
    """Show a cache-first multi-dimension assessment and readiness decision."""
    _run_dataset_command(
        dataset_id,
        sample_size=sample_size,
        refresh=refresh,
        cached_only=cached_only,
        token=token,
        output_format=output_format,
    )


@app.command("score")
def score_command(
    dataset_id: str = typer.Argument(..., help="HuggingFace dataset ID"),
    sample_size: int = typer.Option(10_000, min=100, max=100_000, help="Number of samples to analyze on a cache miss"),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore complete cached evidence and score again"),
    cached_only: bool = typer.Option(False, "--cached-only", help="Never use the network; fail if complete evidence is absent"),
    token: Optional[str] = typer.Option(None, "--token", envvar="HF_TOKEN", help="Optional token for gated datasets"),
    output_format: OutputFormat = typer.Option(OutputFormat.table, "--format", help="Output format"),
):
    """Score a dataset on cache miss; otherwise return exact-ID stored evidence."""
    _run_dataset_command(
        dataset_id,
        sample_size=sample_size,
        refresh=refresh,
        cached_only=cached_only,
        token=token,
        output_format=output_format,
    )


@app.command()
def readiness(
    dataset_id: str = typer.Argument(..., help="HuggingFace dataset ID"),
    sample_size: int = typer.Option(10_000, min=100, max=100_000, help="Number of samples to analyze on a cache miss"),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore complete cached evidence and score again"),
    cached_only: bool = typer.Option(False, "--cached-only", help="Never use the network; fail if complete evidence is absent"),
    token: Optional[str] = typer.Option(None, "--token", envvar="HF_TOKEN", help="Optional token for gated datasets"),
    output_format: OutputFormat = typer.Option(OutputFormat.table, "--format", help="Output format"),
):
    """Return a PROCEED, REVIEW, or DO NOT PROCEED training decision."""
    _run_dataset_command(
        dataset_id,
        sample_size=sample_size,
        refresh=refresh,
        cached_only=cached_only,
        token=token,
        output_format=output_format,
        decision_only=True,
    )


@app.command()
def compare(
    dataset_ids: list[str] = typer.Argument(..., help="Two to twenty HuggingFace dataset IDs"),
    sample_size: int = typer.Option(10_000, min=100, max=100_000, help="Samples per dataset on a cache miss"),
    refresh: bool = typer.Option(False, "--refresh", help="Score every requested dataset again"),
    cached_only: bool = typer.Option(False, "--cached-only", help="Never use the network; fail on missing evidence"),
    token: Optional[str] = typer.Option(None, "--token", envvar="HF_TOKEN", help="Optional token for gated datasets"),
    output_format: OutputFormat = typer.Option(OutputFormat.table, "--format", help="Output format"),
):
    """Compare dataset scores and readiness side by side."""
    from socbench.dataset_intelligence import DatasetIntelligenceError, assess_readiness, rank_comparison

    unique_ids = list(dict.fromkeys(dataset_ids))
    if not 2 <= len(unique_ids) <= 20:
        raise typer.BadParameter("compare requires between 2 and 20 unique dataset IDs")

    async def _run():
        await _prepare_database()
        try:
            rows = []
            for dataset_id in unique_ids:
                from socbench.dataset_intelligence import get_or_score_dataset

                rows.append(
                    await get_or_score_dataset(
                        dataset_id,
                        sample_size=sample_size,
                        refresh=refresh,
                        cached_only=cached_only,
                        token=token,
                    )
                )
            return rank_comparison(rows)
        finally:
            await _dispose_database()

    try:
        evidence_rows = asyncio.run(_run())
    except DatasetIntelligenceError as exc:
        banner_console.print(f"[red]Dataset comparison failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    if output_format is OutputFormat.json:
        typer.echo(
            json.dumps(
                {
                    "ranking": "readiness decision, then automated score",
                    "datasets": [row.to_dict() for row in evidence_rows],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    table = Table(title="Socbench dataset comparison (scores are 0-100)")
    table.add_column("Dataset", style="cyan", max_width=38)
    table.add_column("Decision", style="bold")
    table.add_column("Auto", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Diversity", justify="right")
    table.add_column("Utility", justify="right")
    table.add_column("PII", justify="right")
    table.add_column("Overlap", justify="right")
    table.add_column("Source")
    for row in evidence_rows:
        table.add_row(
            row.dataset_id,
            assess_readiness(row).decision,
            _score_text(row.auto_score),
            _score_text(row.dimensions["quality"].score),
            _score_text(row.dimensions["diversity"].score),
            _score_text(row.dimensions["utility"].score),
            _score_text(row.dimensions["pii_safety"].score),
            _score_text(row.contamination_rate),
            row.source,
        )
    console.print(table)
    console.print("Comparison order: readiness decision, then automated score. Missing evidence is never ranked PROCEED.")


@app.command("cache-status")
def cache_status(
    dataset_id: str = typer.Argument(..., help="Exact HuggingFace dataset ID"),
    output_format: OutputFormat = typer.Option(OutputFormat.table, "--format", help="Output format"),
):
    """Inspect cached evidence without scoring or making a network request."""
    from socbench.dataset_intelligence import get_cached_dataset

    async def _run():
        await _prepare_database()
        try:
            return await get_cached_dataset(dataset_id)
        finally:
            await _dispose_database()

    evidence = asyncio.run(_run())
    if evidence is None:
        if output_format is OutputFormat.json:
            typer.echo(json.dumps({"dataset_id": dataset_id, "found": False}, indent=2, sort_keys=True))
        else:
            console.print(f"[yellow]No cached evidence for {dataset_id}.[/yellow]")
        raise typer.Exit(code=1)
    if output_format is OutputFormat.json:
        typer.echo(json.dumps({"found": True, **evidence.to_dict()}, indent=2, sort_keys=True))
    else:
        _render_dataset_evidence(evidence, decision_only=True)


@app.command()
def audit(
    dataset_id: str = typer.Argument(..., help="HuggingFace dataset ID"),
    output_dir: Path = typer.Option(Path("audit_outputs"), help="Directory for the cleaned JSONL and summary"),
    max_rows: int = typer.Option(100_000, min=1, help="Maximum source rows to inspect"),
    output_size: Optional[int] = typer.Option(None, min=1, help="Optional cap for balanced accepted rows"),
    text_key: Optional[str] = typer.Option(None, help="Text column to audit; auto-detected by default"),
    eval_bank_dir: Optional[Path] = typer.Option(None, help="Local evaluation-bank directory; defaults to built-in benchmarks"),
    min_tokens: int = typer.Option(32, min=1, help="Minimum token count per accepted row"),
    max_tokens: int = typer.Option(8192, min=1, help="Maximum token count per accepted row"),
):
    """Run the full seven-stage cleaning and decontamination audit."""
    from socbench.audit import audit_dataset

    async def _run():
        console.print(f"[bold]Auditing {dataset_id}...[/bold]")
        result = await audit_dataset(
            dataset_id,
            output_dir=output_dir,
            max_rows=max_rows,
            output_size=output_size,
            text_key=text_key,
            eval_bank_dir=eval_bank_dir,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        table = Table(title=f"Audit complete: {dataset_id}")
        table.add_column("Outcome", style="bold")
        table.add_column("Rows", justify="right")
        for label, count in [
            ("Accepted before balancing", result.accepted),
            ("Final balanced output", result.final),
            ("Rejected: license", result.rejected_license),
            ("Rejected: language", result.rejected_language),
            ("Rejected: syntax", result.rejected_syntax),
            ("Rejected: eval contamination", result.rejected_eval_contamination),
            ("Rejected: exact duplicates", result.rejected_exact_dup),
            ("Rejected: near duplicates", result.rejected_near_dup),
            ("Rejected: token length", result.rejected_token_length),
        ]:
            table.add_row(label, str(count))
        console.print(table)
        console.print(f"[green]Balanced data:[/green] {result.output_path}")
        console.print(f"[green]Machine-readable summary:[/green] {result.summary_path}")

    asyncio.run(_run())


@app.command()
def provenance(
    dataset_id: str = typer.Argument(..., help="HuggingFace dataset ID"),
):
    """Show known provenance for a dataset."""
    from socbench.provenance import get_provenance

    entries = get_provenance(dataset_id)
    if not entries:
        console.print(f"[dim]No provenance records for {dataset_id}[/dim]")
        return

    table = Table(title=f"Provenance: {dataset_id}")
    table.add_column("Model", style="cyan")
    table.add_column("Paper", max_width=50)
    table.add_column("URL", max_width=40)
    table.add_column("Verified", justify="right")
    for e in entries:
        verified = "[green]✓[/green]" if e.verified else "[dim]?[/dim]"
        table.add_row(e.model_name or "?", e.paper_title or "?", e.paper_url or "?", verified)
    console.print(table)


@app.command()
def recommendations(
    dataset_id: str = typer.Argument(..., help="HuggingFace dataset ID"),
    sample_size: int = typer.Option(10_000, min=100, max=100_000, help="Number of samples to analyze on a cache miss"),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore complete cached evidence and score again"),
    cached_only: bool = typer.Option(False, "--cached-only", help="Never use the network; fail if complete evidence is absent"),
    token: Optional[str] = typer.Option(None, "--token", envvar="HF_TOKEN", help="Optional token for gated datasets"),
):
    """Generate cache-first 'Best for:' recommendations for a dataset."""
    from socbench.dataset_intelligence import DatasetIntelligenceError
    from socbench.recommendations import format_recommendations_markdown, generate_recommendations

    try:
        evidence = asyncio.run(
            _load_dataset_evidence(
                dataset_id,
                sample_size=sample_size,
                refresh=refresh,
                cached_only=cached_only,
                token=token,
            )
        )
    except DatasetIntelligenceError as exc:
        banner_console.print(f"[red]Recommendation failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    scores = {
        name: item.score
        for name, item in evidence.dimensions.items()
        if item.score is not None
    }
    if evidence.repetition_pct is not None:
        scores["dedup_rate"] = evidence.repetition_pct / 100
    pii_rate = evidence.dimensions["pii_safety"].details.get("pii_rate")
    if isinstance(pii_rate, (int, float)):
        scores["pii_rate"] = float(pii_rate)
    recs = generate_recommendations(dataset_id, scores, {}, [])
    console.print(format_recommendations_markdown(recs))


@app.command()
def classify(
    dataset_id: str = typer.Argument(..., help="HuggingFace dataset ID"),
):
    """Classify a dataset into its hierarchical category."""
    import httpx

    from socbench.categories import CATEGORIES, classify_dataset

    async def _run():
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"https://huggingface.co/api/datasets/{dataset_id}")
            if resp.status_code == 200:
                data = resp.json()
                tags = data.get("tags", [])
                desc = data.get("description", "")
                cat = classify_dataset(tags, desc, dataset_id=dataset_id)
                cat_info = CATEGORIES.get(cat, CATEGORIES["pretraining-web"])
                console.print(f"[bold]{dataset_id}[/bold] → [green]{cat_info.label}[/green] ({cat})")
                console.print(f"  Metrics: {', '.join(cat_info.metrics)}")
            else:
                console.print(f"[red]Could not fetch {dataset_id}[/red]")

    asyncio.run(_run())


@app.command()
def qualify(
    dataset_id: str = typer.Argument(..., help="HuggingFace dataset ID"),
    downloads: int = typer.Option(0, help="Download count"),
    likes: int = typer.Option(0, help="Like count"),
):
    """Check if a dataset qualifies for benchmarking."""
    from socbench.discovery.scanner import qualify_dataset

    async def _run():
        result = await qualify_dataset(dataset_id, downloads=downloads, likes=likes)
        if result.qualified:
            console.print(f"[green]QUALIFIED[/green] — rows={result.row_count}, bytes={result.byte_size}")
        else:
            console.print(f"[red]REJECTED[/red] — {result.reason}")

    asyncio.run(_run())


@app.command()
def leaderboard(
    top: int = typer.Option(20, min=1, max=500, help="Show top N"),
    category: Optional[str] = typer.Option(None, help="Only include this exact category key"),
    sort_by: LeaderboardSort = typer.Option(LeaderboardSort.combined, "--sort", help="Score used for ordering"),
    output_format: OutputFormat = typer.Option(OutputFormat.table, "--format", help="Output format"),
):
    """Show stored leaderboard dimensions without making network requests."""
    from sqlalchemy import select

    from socbench.db import async_session_factory
    from socbench.models import DatasetRow, LeaderboardRow

    async def _run():
        await _prepare_database()
        try:
            sort_columns = {
                LeaderboardSort.combined: LeaderboardRow.combined_score,
                LeaderboardSort.auto: LeaderboardRow.auto_score,
                LeaderboardSort.quality: LeaderboardRow.quality,
                LeaderboardSort.diversity: LeaderboardRow.diversity,
                LeaderboardSort.utility: LeaderboardRow.utility,
                LeaderboardSort.training: LeaderboardRow.training_score,
            }
            async with async_session_factory() as session:
                stmt = select(LeaderboardRow, DatasetRow).join(
                    DatasetRow, DatasetRow.id == LeaderboardRow.dataset_id
                )
                if category:
                    stmt = stmt.where(LeaderboardRow.category == category)
                stmt = stmt.order_by(
                    sort_columns[sort_by].desc().nullslast(),
                    DatasetRow.hf_id,
                ).limit(top)
                return (await session.execute(stmt)).all()
        finally:
            await _dispose_database()

    entries = asyncio.run(_run())
    payload = [
        {
            "rank": entry.rank,
            "dataset_id": dataset.hf_id,
            "category": entry.category,
            "auto_score": entry.auto_score,
            "quality": entry.quality,
            "diversity": entry.diversity,
            "utility": entry.utility,
            "pii_safety": entry.pii_safety,
            "contamination_rate": entry.contamination_score,
            "training_score": entry.training_score,
            "combined_score": entry.combined_score,
        }
        for entry, dataset in entries
    ]
    if output_format is OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    table = Table(title=f"Socbench Leaderboard (Top {top}, sorted by {sort_by.value})")
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Dataset", style="cyan", max_width=42)
    table.add_column("Auto", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Diversity", justify="right")
    table.add_column("Utility", justify="right")
    table.add_column("Training", justify="right")
    table.add_column("Combined", justify="right")
    for row in payload:
        table.add_row(
            str(row["rank"] or "-"),
            row["dataset_id"],
            _score_text(row["auto_score"]),
            _score_text(row["quality"]),
            _score_text(row["diversity"]),
            _score_text(row["utility"]),
            _score_text(row["training_score"]),
            _score_text(row["combined_score"]),
        )
    console.print(table)


@app.command("export-proofs")
def export_proofs(
    output_dir: Path = typer.Option(Path("../eval-proof"), help="Proof export directory"),
    limit: Optional[int] = typer.Option(None, help="Limit number of datasets"),
):
    """Export dataset proof JSON files from the local database."""
    from socbench.eval_proof import export_eval_proofs

    async def _run():
        manifest = await export_eval_proofs(output_dir=output_dir, limit=limit)
        console.print(f"[green]Exported {manifest['count']} proof files[/green]")
        console.print(f"Manifest: {Path(output_dir) / 'manifest.json'}")
        from socbench.db import engine

        await engine.dispose()

    asyncio.run(_run())


@app.command("benchmark-audit")
def benchmark_audit(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Canonical benchmark JSON"),
    output: Optional[Path] = typer.Option(None, help="Write the full machine-readable audit JSON"),
    local_only: bool = typer.Option(False, "--local-only", help="Disable a configured remote semantic judge"),
):
    """Audit conversational benchmark consistency, complexity, and policy coverage."""
    import json

    from socbench.evals.semantic_audit import audit_benchmark, load_benchmark_input

    async def _run():
        request = load_benchmark_input(input_path, use_api_judge=not local_only)
        result = await audit_benchmark(request)

        scores = result["scores"]
        table = Table(title=f"Benchmark audit: {result['benchmark_name']}")
        table.add_column("Dimension", style="bold")
        table.add_column("Score", justify="right")
        table.add_row("Description to expected behavior", f"{scores['description_expected_alignment']:.1f}/100")
        table.add_row("Policy to expected behavior", f"{scores['policy_expected_alignment']:.1f}/100")
        table.add_row("Policy violations per task", f"{scores['policy_violations_per_task']:.3f}")
        table.add_row("Policy violation coverage", f"{scores['policy_violation_coverage']:.1f}%")
        table.add_row("Semantic quality", f"{scores['semantic_quality']:.1f}/100")
        console.print(table)
        console.print(f"Engine: [cyan]{result['engine']}[/cyan]")
        for diagnostic in result["diagnostics"]:
            console.print(f"[yellow]-[/yellow] {diagnostic}")
        for warning in result["warnings"]:
            console.print(f"[yellow]Warning:[/yellow] {warning}")

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            console.print(f"[green]Audit JSON:[/green] {output}")

    try:
        asyncio.run(_run())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Benchmark audit failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="API host"),
    port: int = typer.Option(8000, help="API port"),
):
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run("socbench.api.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
