"""FastAPI routes — datasets, multi-dimension scoring, provenance."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from socbench.db import async_session_factory
from socbench.models import (
    ContaminationRow,
    DatasetRow,
    LeaderboardRow,
    ScoreRow,
    TrainingRunRow,
)
from socbench.runner import run_socbench_scoring

router = APIRouter()


@router.get("/datasets")
async def list_datasets(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    category: Optional[str] = Query(None),
    sort: str = Query("downloads"),
    order: str = Query("desc"),
):
    async with async_session_factory() as session:
        valid_sorts = {"downloads", "likes", "trending_score", "row_count"}
        col = getattr(DatasetRow, sort if sort in valid_sorts else "downloads")
        stmt = select(DatasetRow).order_by(col.desc() if order == "desc" else col.asc())
        stmt = stmt.offset(offset).limit(limit)
        result = await session.execute(stmt)
        datasets = result.scalars().all()
        return [
            {
                "hf_id": ds.hf_id,
                "name": ds.name,
                "license": ds.license,
                "languages": ds.languages,
                "row_count": ds.row_count,
                "downloads": ds.downloads,
                "likes": ds.likes,
                "tags": ds.tags,
            }
            for ds in datasets
        ]


@router.get("/datasets/{hf_id:path}")
async def get_dataset(hf_id: str):
    async with async_session_factory() as session:
        stmt = select(DatasetRow).where(DatasetRow.hf_id == hf_id)
        result = await session.execute(stmt)
        ds = result.scalar_one_or_none()
        if not ds:
            raise HTTPException(404, f"Dataset {hf_id} not found")

        scores_stmt = select(ScoreRow).where(ScoreRow.dataset_id == ds.id)
        scores = (await session.execute(scores_stmt)).scalars().all()

        cont_stmt = select(ContaminationRow).where(ContaminationRow.dataset_id == ds.id)
        contamination = (await session.execute(cont_stmt)).scalars().all()

        train_stmt = select(TrainingRunRow).where(TrainingRunRow.dataset_id == ds.id)
        training = (await session.execute(train_stmt)).scalar_one_or_none()

        return {
            "hf_id": ds.hf_id,
            "name": ds.name,
            "description": ds.description,
            "license": ds.license,
            "languages": ds.languages,
            "tags": ds.tags,
            "row_count": ds.row_count,
            "downloads": ds.downloads,
            "likes": ds.likes,
            "scores": [
                {"name": s.scorer_name, "score": s.score, "details": s.details, "warnings": s.warnings}
                for s in scores
            ],
            "contamination": [
                {"benchmark": c.benchmark_name, "overlap_rate": c.overlap_rate,
                 "details": {"overlap_count": c.overlap_count, "total_eval": c.total_eval}}
                for c in contamination
            ],
            "training": {
                "final_val_loss": training.final_val_loss,
                "loss_curve": training.loss_curve,
                "convergence_steps": training.convergence_steps,
                "eval_scores": training.eval_scores,
            } if training else None,
        }


@router.post("/datasets/{hf_id:path}/score")
async def score_dataset(hf_id: str, sample_size: int = Query(10_000, ge=100, le=100_000)):
    return await run_socbench_scoring(hf_id, sample_size=sample_size)


@router.get("/leaderboard")
async def get_leaderboard(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    category: Optional[str] = Query(None),
):
    async with async_session_factory() as session:
        stmt = (
            select(LeaderboardRow)
            .order_by(LeaderboardRow.combined_score.desc().nullslast())
            .offset(offset)
            .limit(limit)
        )
        result = await session.execute(stmt)
        entries = result.scalars().all()

        leaderboard = []
        for entry in entries:
            ds_stmt = select(DatasetRow).where(DatasetRow.id == entry.dataset_id)
            ds = (await session.execute(ds_stmt)).scalar_one_or_none()
            if ds:
                leaderboard.append({
                    "rank": entry.rank,
                    "hf_id": ds.hf_id,
                    "name": ds.name,
                    "tags": ds.tags,
                    "quality": entry.auto_score,
                    "diversity": entry.auto_score,
                    "utility": entry.auto_score,
                    "documentation": entry.auto_score,
                    "popularity": ds.downloads,
                    "freshness": entry.auto_score,
                    "contamination": entry.contamination_score,
                    "downloads": ds.downloads,
                    "likes": ds.likes,
                })
        return leaderboard


@router.get("/discover")
async def discover_datasets(
    search: str = Query(""),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    days: Optional[int] = Query(None),
):
    """Run HF discovery scan."""
    from socbench.discovery.scanner import scan_datasets

    datasets = await scan_datasets(
        search=search,
        limit=limit,
        days=days,
    )

    qualified = []
    for ds in datasets:
        if ds.private or ds.gated:
            continue
        if ds.downloads < 1000 or ds.likes < 10:
            continue
        qualified.append({
            "hf_id": ds.hf_id,
            "name": ds.name,
            "tags": ds.tags,
            "downloads": ds.downloads,
            "likes": ds.likes,
            "trending_score": ds.trending_score,
            "languages": ds.languages,
            "created_at": ds.created_at,
            "last_modified": ds.last_modified,
            "qualified": True,
        })
    return qualified


@router.get("/stats")
async def get_stats():
    async with async_session_factory() as session:
        ds_count = (await session.execute(select(func.count(DatasetRow.id)))).scalar() or 0
        score_count = (await session.execute(select(func.count(ScoreRow.id)))).scalar() or 0
        return {"total_datasets": ds_count, "total_scores": score_count}