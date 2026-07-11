"""FastAPI routes — datasets, multi-dimension scoring, provenance."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
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
            from socbench.recommendations import generate_recommendations

            cat_key = lb.category if lb else "pretraining-web"
            cat_label = CATEGORIES[cat_key].label if cat_key in CATEGORIES else cat_key

            def block(val):
                return {"score": val, "details": {}} if val is not None else None

            auto_scores = {
                "quality": lb.quality if lb else 0.0,
                "diversity": lb.diversity if lb else 0.0,
                "utility": lb.utility if lb else 0.0,
                "documentation": lb.documentation if lb else 0.0,
                "popularity": lb.popularity if lb else 0.0,
                "freshness": lb.freshness if lb else 0.0,
                "pii_safety": lb.pii_safety if lb else 0.0,
                "contamination": lb.contamination_score if lb else 0.0,
            }
            recs = generate_recommendations(
                ds.hf_id,
                auto_scores,
                {},
                ds.tags or [],
            )

            def _ser(items):
                return [
                    {
                        "category_key": r.category_key,
                        "label": r.label,
                        "rating": r.rating,
                        "confidence": r.confidence,
                        "reasoning": r.reasoning,
                    }
                    for r in items
                ]

            recommendations = {
                "best_for": _ser(recs.best_for),
                "good_for": _ser(recs.good_for),
                "not_for": _ser(recs.not_for),
            }

            return {
                "hf_id": ds.hf_id,
                "name": ds.name,
                "description": ds.description,
                "license": ds.license,
                "tags": ds.tags,
                "source_url": ds.source_url,
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
                "recommendations": recommendations,
                "metadata": {
                    "downloads": ds.downloads or 0,
                    "likes": ds.likes or 0,
                    "license": ds.license,
                },
                "training": {
                    "final_val_loss": training.final_val_loss,
                    "loss_curve": training.loss_curve,
                    "convergence_steps": training.convergence_steps,
                    "tokens_seen": training.tokens_seen,
                    "model_config": training.model_config,
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
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    category: Optional[str] = Query(None),
    sort: str = Query("quality"),
):
    async with async_session_factory() as session:
        valid_sorts = {"quality", "diversity", "utility", "documentation", "popularity", "freshness", "combined_score", "downloads"}
        sort_col = getattr(LeaderboardRow, sort if sort in valid_sorts else "quality")
        stmt = select(LeaderboardRow)
        if category:
            stmt = stmt.where(LeaderboardRow.category.like(f"%{category}%"))
        stmt = stmt.order_by(sort_col.desc().nullslast())
        stmt = stmt.offset(offset).limit(limit)
        result = await session.execute(stmt)
        entries = result.scalars().all()

        def s100(val):
            """Convert 0-1 score to 0-100, rounded to 1 decimal."""
            if val is None:
                return None
            return round(val * 100, 1)

        leaderboard = []
        for entry in entries:
            ds_stmt = select(DatasetRow).where(DatasetRow.id == entry.dataset_id)
            ds = (await session.execute(ds_stmt)).scalar_one_or_none()
            if ds:
                cont = entry.contamination_score or 0.0
                leaderboard.append({
                    "rank": entry.rank,
                    "hf_id": ds.hf_id,
                    "name": ds.name,
                    "tags": ds.tags,
                    "category": entry.category if hasattr(entry, "category") else None,
                    "quality": s100(entry.quality),
                    "diversity": s100(entry.diversity),
                    "utility": s100(entry.utility),
                    "documentation": s100(entry.documentation),
                    "popularity": s100(entry.popularity),
                    "freshness": s100(entry.freshness),
                    "pii_safety": s100(entry.pii_safety),
                    "contamination": s100(cont),
                    "contaminated": cont > 0.01,
                    "repetition_pct": entry.repetition_pct,
                    "combined_score": s100(entry.combined_score),
                    "downloads": ds.downloads,
                    "likes": ds.likes,
                    "created_at": ds.created_at,
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
        cat_key = classify_dataset(ds.tags, dataset_id=ds.hf_id)
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


# ---------------------------------------------------------------------------
# Evaluation requests
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    hf_id: str
    visibility: str = "public"  # "public" or "private"
    requester: str = ""
    notes: str = ""


@router.post("/request-evaluation")
async def request_evaluation(req: EvalRequest):
    """Submit a dataset evaluation request (public or private)."""
    if req.visibility not in ("public", "private"):
        raise HTTPException(status_code=400, detail="visibility must be 'public' or 'private'")
    # In a full deployment this would persist to a queue table and notify.
    # For now we return an acknowledgement so the frontend can confirm.
    return {
        "status": "received",
        "hf_id": req.hf_id,
        "visibility": req.visibility,
        "message": f"Evaluation request for {req.hf_id} ({req.visibility}) received. "
        f"You will be notified when results are available.",
    }