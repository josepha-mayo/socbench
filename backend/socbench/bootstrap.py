"""Canonical catalog export and database bootstrap for clean deployments."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, func, select

from socbench.db import async_session_factory, engine
from socbench.models import (
    ContaminationRow,
    DatasetRow,
    LeaderboardRow,
    ScoreRow,
    TrainingRunRow,
)
from socbench.training.import_results import (
    TrainingArtifact,
    _combined_score,
    _loss_stability,
    discover_result_files,
    load_training_artifact,
)

CATALOG_SCHEMA_VERSION = 1


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _default_catalog_path() -> Path:
    configured = os.getenv("SOCBENCH_CATALOG_PATH")
    if configured:
        return Path(configured)
    candidates = [
        Path.cwd() / "catalog" / "catalog.json",
        Path(__file__).resolve().parents[2] / "catalog" / "catalog.json",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def _default_training_root() -> Path:
    configured = os.getenv("SOCBENCH_TRAINING_RESULTS_ROOT")
    if configured:
        return Path(configured)
    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    return next(
        (root for root in candidates if (root / "training_results" / "validated").exists()),
        candidates[0],
    )


async def export_catalog(output_path: Path) -> int:
    """Export scored catalog state without runtime IDs or training results."""
    records: list[dict] = []
    async with async_session_factory() as session:
        datasets = (
            await session.execute(select(DatasetRow).order_by(DatasetRow.hf_id))
        ).scalars().all()
        for dataset in datasets:
            leaderboard = (
                await session.execute(
                    select(LeaderboardRow).where(LeaderboardRow.dataset_id == dataset.id)
                )
            ).scalar_one_or_none()
            scores = (
                await session.execute(
                    select(ScoreRow)
                    .where(ScoreRow.dataset_id == dataset.id)
                    .order_by(ScoreRow.scorer_name)
                )
            ).scalars().all()
            contamination = (
                await session.execute(
                    select(ContaminationRow)
                    .where(ContaminationRow.dataset_id == dataset.id)
                    .order_by(ContaminationRow.benchmark_name)
                )
            ).scalars().all()
            records.append(
                {
                    "dataset": {
                        "hf_id": dataset.hf_id,
                        "name": dataset.name,
                        "description": dataset.description,
                        "license": dataset.license,
                        "languages": dataset.languages or [],
                        "tags": dataset.tags or [],
                        "source_url": dataset.source_url,
                        "row_count": dataset.row_count,
                        "byte_size": dataset.byte_size,
                        "downloads": dataset.downloads,
                        "likes": dataset.likes,
                        "trending_score": dataset.trending_score,
                        "created_at": dataset.created_at,
                        "discovered_at": _iso(dataset.discovered_at),
                        "last_scored": _iso(dataset.last_scored),
                    },
                    "leaderboard": None
                    if leaderboard is None
                    else {
                        "category": leaderboard.category,
                        "auto_score": leaderboard.auto_score,
                        "quality": leaderboard.quality,
                        "diversity": leaderboard.diversity,
                        "utility": leaderboard.utility,
                        "documentation": leaderboard.documentation,
                        "popularity": leaderboard.popularity,
                        "freshness": leaderboard.freshness,
                        "pii_safety": leaderboard.pii_safety,
                        "contamination_score": leaderboard.contamination_score,
                        "repetition_pct": leaderboard.repetition_pct,
                    },
                    "scores": [
                        {
                            "scorer_name": row.scorer_name,
                            "score": row.score,
                            "details": row.details or {},
                            "warnings": row.warnings or [],
                            "sample_size": row.sample_size,
                            "scored_at": _iso(row.scored_at),
                        }
                        for row in scores
                    ],
                    "contamination": [
                        {
                            "benchmark_name": row.benchmark_name,
                            "overlap_rate": row.overlap_rate,
                            "overlap_count": row.overlap_count,
                            "total_eval": row.total_eval,
                            "method": row.method,
                            "checked_at": _iso(row.checked_at),
                        }
                        for row in contamination
                    ],
                }
            )

    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "dataset_count": len(records),
        "datasets": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return len(records)


async def _load_catalog(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported catalog schema in {path}")
    records = payload.get("datasets")
    if not isinstance(records, list) or payload.get("dataset_count") != len(records):
        raise RuntimeError(f"Invalid catalog dataset count in {path}")

    async with async_session_factory() as session:
        for record in records:
            data = record["dataset"]
            dataset = DatasetRow(
                **{key: value for key, value in data.items() if key not in {"discovered_at", "last_scored"}},
                discovered_at=_datetime(data.get("discovered_at")),
                last_scored=_datetime(data.get("last_scored")),
            )
            session.add(dataset)
            await session.flush()

            leaderboard = record.get("leaderboard")
            if leaderboard:
                auto_score = leaderboard.get("auto_score")
                session.add(
                    LeaderboardRow(
                        dataset_id=dataset.id,
                        **leaderboard,
                        training_score=None,
                        combined_score=auto_score,
                    )
                )
            for score in record.get("scores", []):
                session.add(
                    ScoreRow(
                        dataset_id=dataset.id,
                        **{key: value for key, value in score.items() if key != "scored_at"},
                        scored_at=_datetime(score.get("scored_at")),
                    )
                )
            for row in record.get("contamination", []):
                session.add(
                    ContaminationRow(
                        dataset_id=dataset.id,
                        **{key: value for key, value in row.items() if key != "checked_at"},
                        checked_at=_datetime(row.get("checked_at")),
                    )
                )
        await session.commit()
    return len(records)


def _latest_artifacts(root: Path) -> dict[str, TrainingArtifact]:
    selected: dict[str, TrainingArtifact] = {}
    incomplete: list[str] = []
    for path in discover_result_files(root):
        artifact, reason = load_training_artifact(path, root)
        if artifact is None:
            incomplete.append(f"{path}: {reason}")
            continue
        current = selected.get(artifact.dataset_id)
        if current is None or (artifact.campaign_version, -artifact.best_val_loss) > (
            current.campaign_version,
            -current.best_val_loss,
        ):
            selected[artifact.dataset_id] = artifact
    if incomplete:
        raise RuntimeError("Invalid canonical training artifacts:\n" + "\n".join(incomplete))
    return selected


async def _load_training_results(root: Path) -> int:
    artifacts = _latest_artifacts(root)
    async with async_session_factory() as session:
        for artifact in artifacts.values():
            dataset = (
                await session.execute(select(DatasetRow).where(DatasetRow.hf_id == artifact.dataset_id))
            ).scalar_one_or_none()
            if dataset is None:
                dataset = DatasetRow(
                    hf_id=artifact.dataset_id,
                    name=artifact.dataset_id,
                    description="Validated training result; automated dataset scoring pending.",
                    tags=["training-result-only"],
                )
                session.add(dataset)
                await session.flush()

            await session.execute(delete(TrainingRunRow).where(TrainingRunRow.dataset_id == dataset.id))
            session.add(
                TrainingRunRow(
                    dataset_id=dataset.id,
                    model_config=artifact.model_config,
                    tokens_seen=artifact.n_tokens,
                    final_val_loss=artifact.final_val_loss,
                    loss_curve=artifact.loss_curve,
                    convergence_steps=artifact.convergence_steps,
                    loss_stability=_loss_stability(artifact.loss_curve),
                    eval_scores=artifact.eval_scores,
                    trained_at=_datetime(artifact.trained_at),
                )
            )

            leaderboard = (
                await session.execute(
                    select(LeaderboardRow).where(LeaderboardRow.dataset_id == dataset.id)
                )
            ).scalar_one_or_none()
            score = artifact.eval_scores["training_score"]
            if leaderboard is None:
                leaderboard = LeaderboardRow(dataset_id=dataset.id, category="posttraining-sft")
                session.add(leaderboard)
                await session.flush()
            values = {
                "auto_score": leaderboard.auto_score,
                "quality": leaderboard.quality,
                "diversity": leaderboard.diversity,
                "utility": leaderboard.utility,
                "documentation": leaderboard.documentation,
                "popularity": leaderboard.popularity,
                "freshness": leaderboard.freshness,
                "pii_safety": leaderboard.pii_safety,
            }
            leaderboard.training_score = score
            leaderboard.combined_score = _combined_score(values, score)

        ranked = (
            await session.execute(
                select(LeaderboardRow).order_by(LeaderboardRow.combined_score.desc().nullslast())
            )
        ).scalars().all()
        for rank, row in enumerate(ranked, start=1):
            row.rank = rank
        await session.commit()
    return len(artifacts)


async def bootstrap_from_canonical(strict: bool = False) -> dict[str, int | bool]:
    """Populate a new database from versioned, provider-neutral artifacts."""
    catalog_path = _default_catalog_path()
    training_root = _default_training_root()
    async with async_session_factory() as session:
        existing = await session.scalar(select(func.count()).select_from(DatasetRow))

    if existing:
        return {"bootstrapped": False, "datasets": int(existing), "training_runs": 0}
    if not catalog_path.exists():
        if strict:
            raise RuntimeError(f"Canonical catalog is missing: {catalog_path}")
        return {"bootstrapped": False, "datasets": 0, "training_runs": 0}

    datasets = await _load_catalog(catalog_path)
    training_runs = await _load_training_results(training_root)
    return {"bootstrapped": True, "datasets": datasets, "training_runs": training_runs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="export the current scored catalog")
    export_parser.add_argument("--output", type=Path, default=_default_catalog_path())
    args = parser.parse_args(argv)
    if args.command == "export":
        async def run_export() -> int:
            try:
                return await export_catalog(args.output)
            finally:
                await engine.dispose()

        count = asyncio.run(run_export())
        print(f"exported {count} datasets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
