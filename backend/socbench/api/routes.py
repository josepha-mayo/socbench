"""FastAPI routes — datasets, multi-dimension scoring, provenance."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, StringConstraints
from sqlalchemy import func, select

from socbench.db import async_session_factory
from socbench.evals.semantic_audit import BenchmarkAuditInput
from socbench.models import (
    DatasetRow,
    LeaderboardRow,
    ScoreRow,
    TrainingRunRow,
)
from socbench.runner import run_socbench_scoring

router = APIRouter()

_PENDING_SCAN_CACHE_TTL_SECONDS = 300
_PENDING_SCAN_TIMEOUT_SECONDS = 6
_pending_scan_cache: dict[str, object] = {
    "expires_at": 0.0,
    "trending": [],
    "most_used": [],
}


def _training_payload(training: TrainingRunRow | None) -> dict | None:
    if training is None:
        return None
    evaluation = training.eval_scores or {}
    model_config = training.model_config or {}
    final_loss = training.final_val_loss
    perplexity = None
    if final_loss is not None and final_loss < 20:
        perplexity = round(math.exp(final_loss), 2)
    return {
        "final_val_loss": final_loss,
        "initial_val_loss": evaluation.get("initial_val_loss"),
        "best_val_loss": evaluation.get("best_val_loss"),
        "relative_improvement": evaluation.get("relative_improvement"),
        "best_relative_improvement": evaluation.get("best_relative_improvement"),
        "run_outcome": evaluation.get("run_outcome", "insufficient_evidence"),
        "outcome_reason": evaluation.get("outcome_reason"),
        "scoring_method": evaluation.get("scoring_method"),
        "training_score": evaluation.get("training_score"),
        "perplexity": perplexity,
        "loss_curve": training.loss_curve or [],
        "convergence_steps": training.convergence_steps,
        "completed_steps": model_config.get("completed_steps"),
        "tokens_seen": training.tokens_seen,
        "model_config": model_config,
        "eval_scores": evaluation,
        "trained_at": training.trained_at.isoformat() if training.trained_at else None,
    }


async def _get_training_pending_candidates():
    """Fetch HF pending-training candidates without making the page wait on slow scans."""
    now = time.monotonic()
    if now < float(_pending_scan_cache["expires_at"]):
        return _pending_scan_cache["trending"], _pending_scan_cache["most_used"]

    from socbench.discovery.scanner import scan_datasets

    async def _scan(sort: str):
        return await asyncio.wait_for(
            scan_datasets(sort=sort, limit=10, days=None),
            timeout=_PENDING_SCAN_TIMEOUT_SECONDS,
        )

    trending_result, most_used_result = await asyncio.gather(
        _scan("trendingScore"),
        _scan("downloads"),
        return_exceptions=True,
    )

    if not isinstance(trending_result, Exception):
        _pending_scan_cache["trending"] = trending_result
    if not isinstance(most_used_result, Exception):
        _pending_scan_cache["most_used"] = most_used_result

    _pending_scan_cache["expires_at"] = now + _PENDING_SCAN_CACHE_TTL_SECONDS
    return _pending_scan_cache["trending"], _pending_scan_cache["most_used"]


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
        stmt = select(DatasetRow)
        if category:
            stmt = stmt.join(LeaderboardRow).where(LeaderboardRow.category == category)
        stmt = stmt.order_by(col.desc() if order == "desc" else col.asc())
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

            train_stmt = (
                select(TrainingRunRow)
                .where(TrainingRunRow.dataset_id == ds.id)
                .order_by(TrainingRunRow.trained_at.desc(), TrainingRunRow.id.desc())
                .limit(1)
            )
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
                    "created_at": ds.created_at,
                },
                "training": _training_payload(training),
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
        valid_sorts = {"quality", "diversity", "utility", "documentation", "popularity", "freshness", "combined_score", "downloads", "training_score"}
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
                    "training_score": s100(entry.training_score),
                    "downloads": ds.downloads,
                    "likes": ds.likes,
                    "created_at": ds.created_at,
                })
        return leaderboard


@router.get("/training-leaderboard")
async def get_training_leaderboard(
    limit: int = Query(100, ge=1, le=500),
):
    """Training-impact leaderboard — a higher-level evaluation.

    Returns a single flat list:
      - Trained datasets (GPT-2 124M runs) ranked by training_score
      - Top 10 trending HF datasets (marked "pending")
      - Top 10 most downloaded HF datasets (marked "pending")
    """
    from socbench.categories import CATEGORIES, classify_dataset

    def s100(val):
        if val is None:
            return None
        return round(val * 100, 1)

    entries: list[dict] = []
    seen: set[str] = set()

    async with async_session_factory() as session:
        latest_training_id = (
            select(TrainingRunRow.id)
            .where(TrainingRunRow.dataset_id == DatasetRow.id)
            .order_by(TrainingRunRow.trained_at.desc(), TrainingRunRow.id.desc())
            .limit(1)
            .correlate(DatasetRow)
            .scalar_subquery()
        )
        stmt = (
            select(LeaderboardRow, DatasetRow, TrainingRunRow)
            .join(DatasetRow, LeaderboardRow.dataset_id == DatasetRow.id)
            .outerjoin(TrainingRunRow, TrainingRunRow.id == latest_training_id)
            .where(LeaderboardRow.training_score.isnot(None))
            .order_by(LeaderboardRow.training_score.desc(), TrainingRunRow.trained_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows = result.all()

        for i, (lb, ds, tr) in enumerate(rows, start=1):
            training = _training_payload(tr)
            cat_key = lb.category or "pretraining-web"
            entries.append({
                "training_rank": i,
                "hf_id": ds.hf_id,
                "name": ds.name,
                "category": cat_key,
                "category_label": CATEGORIES[cat_key].label if cat_key in CATEGORIES else cat_key,
                **(training or {}),
                "training_score": s100(lb.training_score),
                "combined_score": s100(lb.combined_score),
                "quality": s100(lb.quality),
                "downloads": ds.downloads,
                "likes": ds.likes,
                "created_at": ds.created_at,
                "status": "trained",
                "source": "trained",
            })
            seen.add(ds.hf_id)

    # Pull live HF pending candidates when fast, otherwise keep the trained
    # leaderboard responsive and fall back to the last successful scan.
    trending_raw, most_used_raw = await _get_training_pending_candidates()

    def _add_pending(rows, source):
        for ds in rows:
            if ds.hf_id in seen:
                continue
            cat_key = classify_dataset(ds.tags, dataset_id=ds.hf_id)
            entries.append({
                "training_rank": None,
                "hf_id": ds.hf_id,
                "name": ds.name,
                "category": cat_key,
                "category_label": CATEGORIES[cat_key].label if cat_key in CATEGORIES else cat_key,
                "training_score": None,
                "combined_score": None,
                "quality": None,
                "final_val_loss": None,
                "initial_val_loss": None,
                "best_val_loss": None,
                "relative_improvement": None,
                "best_relative_improvement": None,
                "run_outcome": None,
                "outcome_reason": None,
                "scoring_method": None,
                "perplexity": None,
                "tokens_seen": None,
                "convergence_steps": None,
                "loss_curve": None,
                "model_config": None,
                "eval_scores": None,
                "trained_at": None,
                "downloads": ds.downloads,
                "likes": ds.likes,
                "created_at": ds.created_at,
                "status": "pending",
                "source": source,
            })
            seen.add(ds.hf_id)

    _add_pending(trending_raw, "trending")
    _add_pending(most_used_raw, "most_used")

    return entries


@router.get("/discover")
async def discover_datasets(
    search: str = Query(""),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    days: Optional[int] = Query(None),
):
    """Run HF discovery scan."""
    from socbench.categories import CATEGORIES, classify_dataset
    from socbench.discovery.scanner import scan_datasets

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
        train_count = (await session.execute(select(func.count(TrainingRunRow.id)))).scalar() or 0
        lb_count = (await session.execute(select(func.count(LeaderboardRow.id)))).scalar() or 0
        return {
            "total_datasets": ds_count,
            "total_scores": score_count,
            "total_training_runs": train_count,
            "total_leaderboard_entries": lb_count,
        }


@router.get("/evals")
async def get_evals(category: Optional[str] = Query(None)):
    """Get eval benchmark analysis — contamination & saturation for top 20 evals."""
    from socbench.evals.analysis import analyze_evals, get_eval_summary
    results = await analyze_evals()
    if category:
        results = [r for r in results if r.category == category]
    summary = get_eval_summary()
    return {
        "summary": summary,
        "evals": [
            {
                "key": r.key,
                "name": r.name,
                "category": r.category,
                "open_source": r.open_source,
                "hf_id": r.hf_id,
                "description": r.description,
                "paper": r.paper,
                "num_examples": r.num_examples,
                "sota_pass_rate": r.sota_pass_rate,
                "ceiling": r.ceiling,
                "saturation_index": r.saturation_index,
                "contamination_risk": r.contamination_risk,
                "discriminative_power": r.discriminative_power,
                "annotation_quality": r.annotation_quality,
                "coverage_breadth": r.coverage_breadth,
                "effective_quality": r.effective_quality,
                "headroom": r.headroom,
                "status": r.status,
                "data_publicly_available": r.data_publicly_available,
                "deprecated": r.deprecated,
                "successor": r.successor,
                "year": r.year,
                "actively_maintained": r.actively_maintained,
                "contamination_incidents": r.contamination_incidents,
                "overlap_with_training": r.overlap_with_training,
                "overlap_benchmarks": r.overlap_benchmarks,
            }
            for r in results
        ],
    }


@router.post("/evals/benchmark-audit")
async def run_benchmark_audit(request: BenchmarkAuditInput):
    """Run a bounded, reference-free semantic quality audit on a benchmark."""
    from socbench.evals.semantic_audit import audit_benchmark

    try:
        return await audit_benchmark(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/trending")
async def get_trending(
    limit: int = Query(30, ge=1, le=200),
    days: Optional[int] = Query(7),
):
    """Fetch real trending datasets from HuggingFace — dynamic, not hardcoded."""
    from socbench.categories import CATEGORIES, classify_dataset
    from socbench.discovery.scanner import scan_datasets

    datasets = await scan_datasets(
        sort="trendingScore",
        limit=limit,
        days=days,
    )

    results = []
    for ds in datasets:
        cat_key = classify_dataset(ds.tags, dataset_id=ds.hf_id)
        is_qualified = not (ds.private or ds.gated) and ds.downloads >= 1000 and ds.likes >= 10
        results.append({
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
    return results


@router.get("/categories")
async def get_categories():
    """Return the full hierarchical category tree for frontend use."""
    from socbench.categories import CATEGORIES
    tree = {}
    for key, cat in CATEGORIES.items():
        node = {
            "key": cat.key,
            "label": cat.label,
            "description": cat.description,
            "parent": cat.parent,
            "metrics": cat.metrics,
            "children": [],
        }
        tree[key] = node
    # Build parent->children mapping
    for key, node in tree.items():
        parent = node["parent"]
        if parent and parent in tree:
            tree[parent]["children"].append(key)
    return tree


# ---------------------------------------------------------------------------
# Evaluation requests
# ---------------------------------------------------------------------------

DatasetId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512, pattern=r"^\S+$"),
]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=256)]
NotesText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=4_000)]


class EvalRequest(BaseModel):
    hf_id: DatasetId
    visibility: Literal["public", "private"] = "public"
    requester_email: ShortText = ""
    requester_name: ShortText = ""
    notes: NotesText = ""


@router.post("/request-evaluation")
async def request_evaluation(req: EvalRequest):
    """Submit a dataset evaluation request (public or private).
    Persists to database. Email is handled client-side via mailto: link."""
    from socbench.models import EvalRequestRow

    async with async_session_factory() as session:
        row = EvalRequestRow(
            hf_id=req.hf_id,
            visibility=req.visibility,
            requester_email=req.requester_email or None,
            requester_name=req.requester_name or None,
            notes=req.notes or None,
            status="pending",
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        request_id = row.id

    return {
        "status": "received",
        "request_id": request_id,
        "hf_id": req.hf_id,
        "visibility": req.visibility,
        "message": f"Evaluation request for {req.hf_id} ({req.visibility}) received. "
        f"Request ID: {request_id}. Saved to queue.",
    }
