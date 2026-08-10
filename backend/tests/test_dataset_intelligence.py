"""Offline tests for cache-first scoring and training-readiness decisions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from socbench.dataset_intelligence import (
    SCORE_DIMENSIONS,
    CacheMissError,
    DatasetEvidence,
    DimensionEvidence,
    assess_readiness,
    get_or_score_dataset,
    rank_comparison,
)
from socbench.models import (
    Base,
    ContaminationRow,
    DatasetRow,
    LeaderboardRow,
    ScoreRow,
    TrainingRunRow,
)


async def _database(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'intelligence.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _seed_dataset(
    session_factory,
    *,
    dataset_id: str = "owner/dataset",
    score: float = 0.8,
    complete: bool = True,
    training_outcome: str | None = None,
) -> None:
    scored_at = datetime(2026, 8, 1, tzinfo=timezone.utc) if complete else None
    async with session_factory() as session:
        dataset = DatasetRow(
            hf_id=dataset_id,
            name=dataset_id,
            license="apache-2.0",
            last_scored=scored_at,
        )
        session.add(dataset)
        await session.flush()
        leaderboard = LeaderboardRow(
            dataset_id=dataset.id,
            category="pretraining-web",
            auto_score=score if complete else None,
            contamination_score=0.0 if complete else None,
            repetition_pct=2.0 if complete else None,
            combined_score=score if complete else None,
        )
        for dimension in SCORE_DIMENSIONS:
            setattr(leaderboard, dimension, score if complete else None)
        session.add(leaderboard)
        if complete:
            for dimension in SCORE_DIMENSIONS:
                session.add(
                    ScoreRow(
                        dataset_id=dataset.id,
                        scorer_name=dimension,
                        score=score,
                        details={"fixture": True},
                        warnings=[],
                        sample_size=500,
                        scored_at=scored_at,
                    )
                )
            session.add(
                ContaminationRow(
                    dataset_id=dataset.id,
                    benchmark_name="fixture-eval",
                    overlap_rate=0.0,
                    overlap_count=0,
                    total_eval=100,
                    method="ngram_13",
                    checked_at=scored_at,
                )
            )
        if training_outcome:
            session.add(
                TrainingRunRow(
                    dataset_id=dataset.id,
                    final_val_loss=3.0,
                    eval_scores={"run_outcome": training_outcome},
                    trained_at=scored_at or datetime(2026, 8, 1, tzinfo=timezone.utc),
                )
            )
        await session.commit()


def _live_result(dataset_id: str, score: float = 0.9) -> dict:
    result = {
        "hf_id": dataset_id,
        "name": dataset_id,
        "description": "Fixture dataset used only by an isolated offline test.",
        "license": "apache-2.0",
        "tags": ["language:en", "task_categories:text-generation"],
        "category": "pretraining-web",
        "contamination_rate": 0.002,
        "contamination_checks": [
            {
                "name": "contamination_fixture",
                "details": {
                    "benchmark": "fixture-eval",
                    "overlap_rate": 0.002,
                    "overlap_count": 2,
                    "total_eval_ngrams": 1000,
                    "method": "ngram_13",
                },
            }
        ],
        "repetition_pct": 3.5,
        "samples_fetched": 500,
        "metadata": {
            "downloads": 1000,
            "likes": 25,
            "created_at": "2026-01-01T00:00:00Z",
        },
    }
    for dimension in SCORE_DIMENSIONS:
        result[dimension] = {
            "score": score,
            "details": {"fixture": True},
            "warnings": [],
        }
    return result


def _evidence(
    *,
    score: float = 0.8,
    complete: bool = True,
    contamination: float = 0.0,
    repetition: float = 2.0,
    training_outcome: str | None = None,
) -> DatasetEvidence:
    dimensions = {
        name: DimensionEvidence(score=score, details={"fixture": True})
        for name in SCORE_DIMENSIONS
    }
    dimensions["pii_safety"] = DimensionEvidence(score=0.99, details={"fixture": True})
    return DatasetEvidence(
        dataset_id="owner/dataset",
        name="owner/dataset",
        category="pretraining-web",
        license="apache-2.0",
        dimensions=dimensions,
        auto_score=score,
        contamination_rate=contamination,
        contamination_checks=({"benchmark": "fixture-eval", "overlap_rate": contamination},),
        repetition_pct=repetition,
        training_score=0.1 if training_outcome else None,
        training_outcome=training_outcome,
        training_reason=None,
        last_scored=datetime(2026, 8, 1, tzinfo=timezone.utc),
        complete=complete,
        missing_evidence=() if complete else ("quality",),
        source="cache",
    )


@pytest.mark.asyncio
async def test_complete_exact_id_cache_prevents_network_scoring(tmp_path):
    engine, session_factory = await _database(tmp_path)
    await _seed_dataset(session_factory)
    calls = 0

    async def forbidden_scorer(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("network scorer must not be called for complete cached evidence")

    try:
        evidence = await get_or_score_dataset(
            "owner/dataset",
            scorer=forbidden_scorer,
            session_factory=session_factory,
        )
        assert evidence.source == "cache"
        assert evidence.complete is True
        assert calls == 0

        with pytest.raises(CacheMissError, match="No cached assessment"):
            await get_or_score_dataset(
                "OWNER/DATASET",
                cached_only=True,
                scorer=forbidden_scorer,
                session_factory=session_factory,
            )
        assert calls == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_cache_miss_scores_once_persists_and_then_reuses(tmp_path):
    engine, session_factory = await _database(tmp_path)
    await _seed_dataset(session_factory, complete=False)
    calls: list[tuple[str, int, str | None]] = []

    async def fake_scorer(dataset_id: str, *, sample_size: int, token: str | None):
        calls.append((dataset_id, sample_size, token))
        return _live_result(dataset_id)

    try:
        live = await get_or_score_dataset(
            "owner/dataset",
            sample_size=500,
            token="fixture-token",
            scorer=fake_scorer,
            session_factory=session_factory,
        )
        assert live.source == "live"
        assert live.complete is True
        assert live.auto_score == pytest.approx(0.9)
        assert calls == [("owner/dataset", 500, "fixture-token")]

        cached = await get_or_score_dataset(
            "owner/dataset",
            cached_only=True,
            scorer=fake_scorer,
            session_factory=session_factory,
        )
        assert cached.source == "cache"
        assert calls == [("owner/dataset", 500, "fixture-token")]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_refresh_forces_scoring_even_when_cache_is_complete(tmp_path):
    engine, session_factory = await _database(tmp_path)
    await _seed_dataset(session_factory, score=0.7)
    calls = 0

    async def fake_scorer(dataset_id: str, *, sample_size: int, token: str | None):
        nonlocal calls
        calls += 1
        return _live_result(dataset_id, score=0.95)

    try:
        evidence = await get_or_score_dataset(
            "owner/dataset",
            refresh=True,
            scorer=fake_scorer,
            session_factory=session_factory,
        )
        assert evidence.source == "live"
        assert evidence.auto_score == pytest.approx(0.95)
        assert calls == 1
    finally:
        await engine.dispose()


def test_readiness_never_marks_incomplete_or_failed_training_evidence_ready():
    assert assess_readiness(_evidence()).decision == "PROCEED"
    assert assess_readiness(_evidence(complete=False)).decision == "REVIEW"
    assert assess_readiness(_evidence(training_outcome="diverged")).decision == "DO NOT PROCEED"
    assert assess_readiness(_evidence(contamination=0.08)).decision == "DO NOT PROCEED"
    assert assess_readiness(_evidence(repetition=60.0)).decision == "DO NOT PROCEED"


def test_comparison_orders_decision_then_score():
    review = _evidence(score=0.6)
    proceed_low = _evidence(score=0.75)
    proceed_high = _evidence(score=0.9)
    rows = rank_comparison([review, proceed_low, proceed_high])

    assert [row.auto_score for row in rows] == [0.9, 0.75, 0.6]
