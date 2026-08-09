"""Export dataset proof JSON files from the Socbench database."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from socbench.db import async_session_factory
from socbench.models import ContaminationRow, DatasetRow, LeaderboardRow, ScoreRow, TrainingRunRow


def proof_path_for_hf_id(output_dir: str | Path, hf_id: str) -> Path:
    """Return the canonical proof path for a Hugging Face dataset ID."""
    parts = [part for part in hf_id.split("/") if part]
    if len(parts) == 1:
        parts = ["_", parts[0]]
    return Path(output_dir).joinpath(*parts, "dataset.json")


def build_dataset_proof(
    dataset: DatasetRow,
    leaderboard: LeaderboardRow | None,
    scores: list[ScoreRow],
    contamination: list[ContaminationRow],
    training_runs: list[TrainingRunRow],
    generated_at: str | None = None,
) -> dict:
    """Build a serializable proof object for one dataset."""
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    training_run = training_runs[-1] if training_runs else None

    return {
        "schema_version": 1,
        "generated_at": generated_at,
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
            "discovered_at": dataset.discovered_at.isoformat() if dataset.discovered_at else None,
            "last_scored": dataset.last_scored.isoformat() if dataset.last_scored else None,
        },
        "leaderboard": None
        if leaderboard is None
        else {
            "rank": leaderboard.rank,
            "category": leaderboard.category,
            "auto_score": leaderboard.auto_score,
            "combined_score": leaderboard.combined_score,
            "quality": leaderboard.quality,
            "diversity": leaderboard.diversity,
            "utility": leaderboard.utility,
            "documentation": leaderboard.documentation,
            "popularity": leaderboard.popularity,
            "freshness": leaderboard.freshness,
            "pii_safety": leaderboard.pii_safety,
            "training_score": leaderboard.training_score,
            "contamination_score": leaderboard.contamination_score,
            "repetition_pct": leaderboard.repetition_pct,
            "updated_at": leaderboard.updated_at.isoformat() if leaderboard.updated_at else None,
        },
        "scores": [
            {
                "scorer_name": score.scorer_name,
                "score": score.score,
                "details": score.details or {},
                "warnings": score.warnings or [],
                "sample_size": score.sample_size,
                "scored_at": score.scored_at.isoformat() if score.scored_at else None,
            }
            for score in sorted(scores, key=lambda item: item.scorer_name)
        ],
        "contamination": [
            {
                "benchmark_name": row.benchmark_name,
                "overlap_rate": row.overlap_rate,
                "overlap_count": row.overlap_count,
                "total_eval": row.total_eval,
                "method": row.method,
                "checked_at": row.checked_at.isoformat() if row.checked_at else None,
            }
            for row in sorted(contamination, key=lambda item: item.benchmark_name)
        ],
        "training": None
        if training_run is None
        else {
            "tokens_seen": training_run.tokens_seen,
            "final_val_loss": training_run.final_val_loss,
            "loss_curve": training_run.loss_curve or [],
            "convergence_steps": training_run.convergence_steps,
            "loss_stability": training_run.loss_stability,
            "eval_scores": training_run.eval_scores or {},
            "gpu_hours": training_run.gpu_hours,
            "trained_at": training_run.trained_at.isoformat() if training_run.trained_at else None,
            "model_config": training_run.model_config or {},
            "initial_val_loss": (training_run.eval_scores or {}).get("initial_val_loss"),
            "best_val_loss": (training_run.eval_scores or {}).get("best_val_loss"),
            "relative_improvement": (training_run.eval_scores or {}).get("relative_improvement"),
            "run_outcome": (training_run.eval_scores or {}).get("run_outcome"),
            "outcome_reason": (training_run.eval_scores or {}).get("outcome_reason"),
        },
    }


async def export_eval_proofs(output_dir: str | Path, limit: int | None = None) -> dict:
    """Export proof JSON files for datasets in the current database."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    exported: list[str] = []

    async with async_session_factory() as session:
        stmt = select(DatasetRow).order_by(DatasetRow.hf_id)
        if limit is not None:
            stmt = stmt.limit(limit)
        datasets = (await session.execute(stmt)).scalars().all()

        for dataset in datasets:
            leaderboard = (
                await session.execute(
                    select(LeaderboardRow).where(LeaderboardRow.dataset_id == dataset.id)
                )
            ).scalar_one_or_none()
            scores = (
                await session.execute(select(ScoreRow).where(ScoreRow.dataset_id == dataset.id))
            ).scalars().all()
            contamination = (
                await session.execute(select(ContaminationRow).where(ContaminationRow.dataset_id == dataset.id))
            ).scalars().all()
            training_runs = (
                await session.execute(select(TrainingRunRow).where(TrainingRunRow.dataset_id == dataset.id))
            ).scalars().all()

            proof = build_dataset_proof(
                dataset=dataset,
                leaderboard=leaderboard,
                scores=list(scores),
                contamination=list(contamination),
                training_runs=list(training_runs),
                generated_at=generated_at,
            )
            path = proof_path_for_hf_id(output, dataset.hf_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            exported.append(str(path))

    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "count": len(exported),
        "files": exported,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
