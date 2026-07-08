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

        if ds:
            lb_stmt = select(LeaderboardRow).where(LeaderboardRow.dataset_id == ds.id)
            lb = (await session.execute(lb_stmt)).scalar_one_or_none()

            train_stmt = select(TrainingRunRow).where(TrainingRunRow.dataset_id == ds.id)
            training = (await session.execute(train_stmt)).scalar_one_or_none()

            from socbench.categories import CATEGORIES
            from socbench.provenance import get_provenance

            cat_key = lb.category if lb else "pretraining-web"
            cat_label = CATEGORIES[cat_key].label if cat_key in CATEGORIES else cat_key

            def block(val):
                return {"score": val, "details": {}} if val is not None else None

            return {
                "hf_id": ds.hf_id,
                "name": ds.name,
                "description": ds.description,
                "license": ds.license,
                "tags": ds.tags,
                "category": cat_key,
                "category_label": cat_label,
                "quality": block(lb.quality if lb else None),
                "diversity": block(lb.diversity if lb else None),
                "utility": block(lb.utility if lb else None),
                "documentation": block(lb.documentation if lb else None),
                "popularity": block(lb.popularity if lb else None),
                "freshness": block(lb.freshness if lb else None),
                "pii_safety": block(lb.pii_safety if lb else None),
                "coverage": {},
                "contamination_rate": lb.contamination_score if lb else 0.0,
                "provenance": [
                    {
                        "model_name": p.model_name,
                        "paper_title": p.paper_title,
                        "paper_url": p.paper_url,
                        "verified": p.verified,
                    }
                    for p in get_provenance(hf_id)
                ],
                "category_metrics": [],
                "metadata": {
                    "downloads": ds.downloads or 0,
                    "likes": ds.likes or 0,
                    "license": ds.license,
                },
                "training": {
                    "final_val_loss": training.final_val_loss,
                    "loss_curve": training.loss_curve,
                    "convergence_steps": training.convergence_steps,
                    "eval_scores": training.eval_scores,
                } if training else None,
            }

    # Not in DB yet — perform a live on-demand examination.
    return await run_socbench_scoring(hf_id)


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
        stmt = select(LeaderboardRow)
        if category:
            stmt = stmt.where(LeaderboardRow.category.like(f"%{category}%"))
        stmt = stmt.order_by(LeaderboardRow.combined_score.desc().nullslast())
        stmt = stmt.offset(offset).limit(limit)
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
                    "category": entry.category if hasattr(entry, "category") else None,
                    "quality": entry.quality,
                    "diversity": entry.diversity,
                    "utility": entry.utility,
                    "documentation": entry.documentation,
                    "popularity": entry.popularity,
                    "freshness": entry.freshness,
                    "pii_safety": entry.pii_safety,
                    "contamination": entry.contamination_score,
                    "combined_score": entry.combined_score,
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
    from socbench.categories import classify_dataset, CATEGORIES

    datasets = await scan_datasets(
        search=search,
        limit=limit,
        days=days,
    )

    qualified = []
    for ds in datasets:
        cat_key = classify_dataset(ds.tags)
        is_qualified = not (ds.private or ds.gated) and ds.downloads >= 1000 and ds.likes >= 10
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
            "category": cat_key,
            "category_label": CATEGORIES[cat_key].label if cat_key in CATEGORIES else cat_key,
            "qualified": is_qualified,
        })
    return qualified


@router.get("/stats")
async def get_stats():
    async with async_session_factory() as session:
        ds_count = (await session.execute(select(func.count(DatasetRow.id)))).scalar() or 0
        score_count = (await session.execute(select(func.count(ScoreRow.id)))).scalar() or 0
        return {"total_datasets": ds_count, "total_scores": score_count}