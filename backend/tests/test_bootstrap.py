from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from socbench import bootstrap
from socbench.models import Base, DatasetRow, LeaderboardRow, TrainingRunRow


@pytest.mark.asyncio
async def test_canonical_bootstrap_loads_catalog_and_training_result(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'clean.db'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(bootstrap, "async_session_factory", session_factory)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    now = datetime.now(timezone.utc).isoformat()
    catalog = {
        "schema_version": 1,
        "dataset_count": 1,
        "datasets": [
            {
                "dataset": {
                    "hf_id": "org/data",
                    "name": "org/data",
                    "description": "fixture",
                    "license": "mit",
                    "languages": ["en"],
                    "tags": ["text"],
                    "source_url": None,
                    "row_count": 100,
                    "byte_size": 200,
                    "downloads": 300,
                    "likes": 4,
                    "trending_score": 5.0,
                    "created_at": now,
                    "discovered_at": now,
                    "last_scored": now,
                },
                "leaderboard": {
                    "category": "pretraining-web",
                    "auto_score": 0.8,
                    "quality": 0.8,
                    "diversity": 0.8,
                    "utility": 0.8,
                    "documentation": 0.8,
                    "popularity": 0.8,
                    "freshness": 0.8,
                    "pii_safety": 0.8,
                    "contamination_score": 0.0,
                    "repetition_pct": 0.0,
                },
                "scores": [],
                "contamination": [],
            }
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    result_path = tmp_path / "training_results" / "validated" / "data_v1" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "dataset_id": "org/data",
                "n_tokens": 1000,
                "max_iters": 10,
                "final_val_loss": 8.0,
                "best_val_loss": 8.0,
                "loss_curve": [
                    {"step": 0, "train_loss": None, "val_loss": 10.0},
                    {"step": 9, "train_loss": None, "val_loss": 8.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert await bootstrap._load_catalog(catalog_path) == 1
    assert await bootstrap._load_training_results(tmp_path) == 1

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(DatasetRow)) == 1
        run = (await session.execute(select(TrainingRunRow))).scalar_one()
        leaderboard = (await session.execute(select(LeaderboardRow))).scalar_one()
        assert run.eval_scores["run_outcome"] == "improved"
        assert run.convergence_steps == 9
        assert leaderboard.training_score == pytest.approx(0.2)
        assert leaderboard.combined_score == pytest.approx(0.74)

    await engine.dispose()
