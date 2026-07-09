"""Typer CLI — commands for agents and humans.

Socbench — 'The unexamined dataset is not worth training on.'
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

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
)
console = Console()


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
            from socbench.categories import classify_dataset, CATEGORIES
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
    sample_size: int = typer.Option(10_000, help="Number of samples to analyze"),
):
    """Run full multi-dimension assessment on a dataset."""
    from socbench.runner import run_socbench_scoring

    async def _run():
        console.print(f"[bold]Assessing {dataset_id}...[/bold]")
        result = await run_socbench_scoring(dataset_id, sample_size=sample_size)

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            return

        cat = result["category"]
        cat_label = result["category_label"]
        console.print(Panel(f"[bold]{dataset_id}[/bold]\nCategory: {cat_label} ({cat})", title="Socbench Assessment"))

        # Core dimensions
        t = Table(title="Core Dimensions")
        t.add_column("Dimension", style="bold")
        t.add_column("Score", justify="right")
        for dim in ["quality", "diversity", "utility"]:
            s = result.get(dim, {})
            score_val = s.get("score", 0)
            color = "green" if score_val >= 0.7 else "yellow" if score_val >= 0.4 else "red"
            t.add_row(dim.title(), f"[{color}]{score_val:.3f}[/{color}]")
        console.print(t)

        # Supporting
        t2 = Table(title="Supporting Dimensions")
        t2.add_column("Dimension", style="bold")
        t2.add_column("Score", justify="right")
        t2.add_column("Details", max_width=40)
        for dim in ["documentation", "popularity", "freshness", "pii_safety"]:
            s = result.get(dim, {})
            score_val = s.get("score", 0)
            details = s.get("details", {})
            detail_str = ", ".join(f"{k}={v}" for k, v in list(details.items())[:3])
            t2.add_row(dim.title(), f"{score_val:.3f}", detail_str)
        console.print(t2)

        # Contamination
        t3 = Table(title="Contamination Check")
        t3.add_column("Benchmark", style="cyan")
        t3.add_column("Overlap Rate", justify="right")
        for c in result.get("contamination_checks", []):
            val = c.get("score", 1.0)
            color = "red" if val < 0.9 else "yellow" if val < 0.95 else "green"
            t3.add_row(c.get("name", "?"), f"[{color}]{val:.3f}[/{color}]")
        console.print(t3)

        # Provenance
        prov = result.get("provenance", [])
        if prov:
            t4 = Table(title="Provenance — Known Models & Papers")
            t4.add_column("Model", style="cyan")
            t4.add_column("Paper", max_width=40)
            t4.add_column("Verified", justify="right")
            for p in prov:
                verified = "[green]✓[/green]" if p.get("verified") else "[dim]?[/dim]"
                t4.add_row(p.get("model_name", "?"), p.get("paper_title", "?"), verified)
            console.print(t4)

        # Coverage
        cov = result.get("coverage", {})
        if cov:
            domains = cov.get("domain_distribution", {})
            if domains:
                t5 = Table(title="Domain Coverage")
                t5.add_column("Domain", style="bold")
                t5.add_column("Share", justify="right")
                for dk, dv in list(domains.items())[:8]:
                    t5.add_row(dk, f"{dv*100:.1f}%")
                console.print(t5)

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
):
    """Generate 'Best for:' recommendations for a dataset."""
    from socbench.runner import run_socbench_scoring
    from socbench.recommendations import generate_recommendations, format_recommendations_markdown

    async def _run():
        result = await run_socbench_scoring(dataset_id)
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            return

        # Build score dict
        scores = {}
        for dim in ["quality", "diversity", "utility"]:
            s = result.get(dim, {})
            scores[dim] = s.get("score", 0)

        tags = result.get("metadata", {}).get("tags", [])
        coverage = result.get("coverage", {})

        recs = generate_recommendations(dataset_id, scores, coverage, tags)
        console.print(format_recommendations_markdown(recs))

    asyncio.run(_run())


@app.command()
def classify(
    dataset_id: str = typer.Argument(..., help="HuggingFace dataset ID"),
):
    """Classify a dataset into its hierarchical category."""
    import httpx
    from socbench.categories import classify_dataset, CATEGORIES

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
    top: int = typer.Option(20, help="Show top N"),
):
    """Show the dataset leaderboard."""
    from socbench.db import async_session_factory
    from socbench.models import DatasetRow, LeaderboardRow
    from sqlalchemy import select

    async def _run():
        async with async_session_factory() as session:
            stmt = (
                select(LeaderboardRow)
                .order_by(LeaderboardRow.combined_score.desc().nullslast())
                .limit(top)
            )
            result = await session.execute(stmt)
            entries = result.scalars().all()

            table = Table(title=f"Socbench Leaderboard (Top {top})")
            table.add_column("Rank", justify="right", style="bold")
            table.add_column("Dataset", style="cyan")
            table.add_column("Quality", justify="right")
            table.add_column("Diversity", justify="right")
            table.add_column("Utility", justify="right")
            for entry in entries:
                ds_stmt = select(DatasetRow).where(DatasetRow.id == entry.dataset_id)
                ds = (await session.execute(ds_stmt)).scalar_one_or_none()
                name = ds.hf_id if ds else "?"
                quality = entry.auto_score or 0
                table.add_row(
                    str(entry.rank or "-"),
                    name,
                    f"{quality:.3f}",
                    f"{quality:.3f}",
                    f"{quality:.3f}",
                )
            console.print(table)

    asyncio.run(_run())


@app.command()
def notebook(
    dataset_id: str = typer.Argument(..., help="HuggingFace dataset ID"),
    output_dir: str = typer.Option("./kaggle_output", help="Output directory"),
):
    """Generate a Kaggle training notebook for a dataset."""
    from socbench.kaggle.notebook import save_notebook

    result = save_notebook(output_dir, dataset_id)
    console.print(f"[green]Notebook saved:[/green] {result['path']}")
    console.print(f"Kernel slug: {result['slug']}")


@app.command()
def queue_stats():
    """Show training job queue statistics."""
    from socbench.kaggle.queue import JobQueue

    q = JobQueue()
    stats = q.get_stats()
    console.print(f"Queue: {json.dumps(stats, indent=2)}")


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