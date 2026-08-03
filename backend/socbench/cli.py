"""Typer CLI — commands for agents and humans.

Socbench — 'The unexamined dataset is not worth training on.'
"""

from __future__ import annotations

import asyncio
import os
import sys
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
)
kaggle_app = typer.Typer(help="Kaggle training bundle helpers.")
app.add_typer(kaggle_app, name="kaggle")
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
    from socbench.recommendations import format_recommendations_markdown, generate_recommendations
    from socbench.runner import run_socbench_scoring

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
    top: int = typer.Option(20, help="Show top N"),
):
    """Show the dataset leaderboard."""
    from sqlalchemy import select

    from socbench.db import async_session_factory
    from socbench.models import DatasetRow, LeaderboardRow

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


@kaggle_app.command("accounts")
def kaggle_accounts(
    profiles_dir: Optional[str] = typer.Option(None, help="Kaggle profiles directory"),
):
    """List configured Kaggle accounts without printing credentials."""
    from socbench.kaggle.accounts import DEFAULT_PROFILES_DIR, MultiAccountManager

    manager = MultiAccountManager(profiles_dir or DEFAULT_PROFILES_DIR)
    table = Table(title=f"Kaggle accounts ({len(manager.accounts)})")
    table.add_column("Name", style="cyan")
    table.add_column("Username")
    table.add_column("Max kernels", justify="right")
    table.add_column("Config dir", style="dim")
    for account in manager.accounts:
        table.add_row(
            account.name,
            account.username,
            str(account.max_kernels),
            account.config_dir,
        )
    console.print(table)
    console.print(f"[bold]Nominal slots:[/bold] {manager.get_total_slots()}")


@kaggle_app.command("bundle")
def kaggle_bundle(
    dataset_id: str = typer.Argument(..., help="Hugging Face dataset ID"),
    prepared_data_dir: Path = typer.Argument(..., help="Directory containing train.bin"),
    output_root: Path = typer.Option(Path("kaggle_train"), help="Bundle output directory"),
    account: Optional[str] = typer.Option(None, help="Kaggle account name or username"),
    tokens: int = typer.Option(1_000_000_000, help="Token budget for generated training script"),
    binary_filename: str = typer.Option("train.bin", help="Prepared binary filename"),
):
    """Create a local Kaggle dataset+kernel bundle for a prepared training run."""
    from socbench.kaggle.accounts import DEFAULT_PROFILES_DIR, MultiAccountManager
    from socbench.kaggle.pipeline import create_training_bundle

    manager = MultiAccountManager(DEFAULT_PROFILES_DIR)
    selected = None
    if account:
        selected = next((a for a in manager.accounts if account in (a.name, a.username)), None)
        if selected is None:
            raise typer.BadParameter(f"Unknown Kaggle account: {account}")
    else:
        selected = manager.get_available_account()
        if selected is None:
            raise typer.BadParameter("No Kaggle accounts available")

    bundle = create_training_bundle(
        dataset_id=dataset_id,
        prepared_data_dir=prepared_data_dir,
        output_root=output_root,
        kaggle_owner=selected.username,
        tokens=tokens,
        binary_filename=binary_filename,
    )

    console.print(Panel(f"[bold]{dataset_id}[/bold]\n{bundle.bundle_dir}", title="Kaggle Training Bundle"))
    console.print(f"Dataset ref: [cyan]{bundle.kaggle_dataset_ref}[/cyan]")
    console.print(f"Kernel slug: [cyan]{bundle.kernel_slug}[/cyan]")
    console.print(f"Account: [cyan]{selected.name}[/cyan] ({selected.username})")
    console.print("\n[bold]Review files:[/bold]")
    console.print(f"  Data:   {bundle.data_dir}")
    console.print(f"  Kernel: {bundle.kernel_dir}")
    console.print("\n[bold]Push commands:[/bold]")
    console.print(f'$env:KAGGLE_CONFIG_DIR = "{selected.config_dir}"')
    console.print(f'kaggle datasets create -p "{bundle.data_dir}" --dir-mode zip')
    console.print(f'kaggle kernels push -p "{bundle.kernel_dir}"')
    console.print("\nIf the dataset already exists, use:")
    console.print(f'kaggle datasets version -p "{bundle.data_dir}" -m "Update prepared Socbench data" --dir-mode zip')


@kaggle_app.command("launch")
def kaggle_launch(
    dataset_id: str = typer.Argument(..., help="Hugging Face dataset ID"),
    prepared_data_dir: Path = typer.Argument(..., help="Directory containing train.bin"),
    output_root: Path = typer.Option(Path("kaggle_train"), help="Bundle output directory"),
    account: Optional[str] = typer.Option(None, help="Kaggle account name or username"),
    tokens: int = typer.Option(1_000_000_000, help="Token budget for generated training script"),
    binary_filename: str = typer.Option("train.bin", help="Prepared binary filename"),
):
    """Create/version the Kaggle dataset and push the GPU training kernel."""
    from socbench.kaggle.accounts import DEFAULT_PROFILES_DIR, MultiAccountManager
    from socbench.kaggle.pipeline import create_training_bundle, push_training_bundle

    manager = MultiAccountManager(DEFAULT_PROFILES_DIR)
    selected = None
    if account:
        selected = next((a for a in manager.accounts if account in (a.name, a.username)), None)
        if selected is None:
            raise typer.BadParameter(f"Unknown Kaggle account: {account}")
    else:
        selected = manager.get_available_account()
        if selected is None:
            raise typer.BadParameter("No Kaggle accounts available")

    bundle = create_training_bundle(
        dataset_id=dataset_id,
        prepared_data_dir=prepared_data_dir,
        output_root=output_root,
        kaggle_owner=selected.username,
        tokens=tokens,
        binary_filename=binary_filename,
    )
    try:
        result = push_training_bundle(
            bundle=bundle,
            kaggle_config_dir=selected.config_dir,
            kaggle_api_token=selected.access_token,
            kaggle_credentials_file=selected.credentials_file,
        )
    except RuntimeError as exc:
        console.print(f"[red]Kaggle launch failed:[/red] {exc}")
        raise typer.Exit(code=1) from None

    console.print(Panel(f"[bold]{dataset_id}[/bold]\n{bundle.bundle_dir}", title="Kaggle Training Launched"))
    console.print(f"Account: [cyan]{selected.name}[/cyan] ({selected.username})")
    console.print(f"Dataset ref: [cyan]{bundle.kaggle_dataset_ref}[/cyan] ({result.dataset_action})")
    console.print(f"Kernel: [cyan]{bundle.kaggle_owner}/{bundle.kernel_slug}[/cyan]")
    console.print(f"Manifest: [cyan]{result.manifest_path}[/cyan]")


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
