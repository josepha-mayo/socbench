"""Tests for eval-proof export helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from socbench.eval_proof import build_dataset_proof, proof_path_for_hf_id


def test_proof_path_for_hf_id_uses_org_and_dataset():
    path = proof_path_for_hf_id("eval-proof", "HuggingFaceFW/fineweb")
    assert path.as_posix() == "eval-proof/HuggingFaceFW/fineweb/dataset.json"


def test_proof_path_for_hf_id_handles_single_segment_ids():
    path = proof_path_for_hf_id("eval-proof", "wikitext")
    assert path.as_posix() == "eval-proof/_/wikitext/dataset.json"


def test_build_dataset_proof_includes_scores_and_training():
    now = datetime(2026, 8, 3, tzinfo=timezone.utc)
    dataset = SimpleNamespace(
        hf_id="org/data",
        name="data",
        description="demo",
        license="mit",
        languages=["en"],
        tags=["text"],
        source_url="https://huggingface.co/datasets/org/data",
        row_count=10,
        byte_size=20,
        downloads=30,
        likes=4,
        trending_score=5.0,
        created_at="2026-01-01T00:00:00Z",
        discovered_at=now,
        last_scored=now,
    )
    leaderboard = SimpleNamespace(
        rank=1,
        category="pretraining-web",
        auto_score=0.8,
        combined_score=0.82,
        quality=0.7,
        diversity=0.6,
        utility=0.5,
        documentation=0.4,
        popularity=0.3,
        freshness=0.2,
        pii_safety=0.9,
        training_score=1.0,
        contamination_score=0.95,
        repetition_pct=0.01,
        updated_at=now,
    )
    score = SimpleNamespace(
        scorer_name="quality",
        score=0.7,
        details={"ok": True},
        warnings=[],
        sample_size=10,
        scored_at=now,
    )
    contamination = SimpleNamespace(
        benchmark_name="MMLU",
        overlap_rate=0.01,
        overlap_count=1,
        total_eval=100,
        method="ngram_13",
        checked_at=now,
    )
    training = SimpleNamespace(
        tokens_seen=123,
        final_val_loss=3.14,
        loss_curve=[4.0, 3.14],
        convergence_steps=10,
        loss_stability=0.1,
        eval_scores={"ppl": 23.1},
        gpu_hours=1.5,
        trained_at=now,
        model_config={"source_artifact": "kaggle_socbench/results/run/loss_curve.json"},
    )

    proof = build_dataset_proof(
        dataset=dataset,
        leaderboard=leaderboard,
        scores=[score],
        contamination=[contamination],
        training_runs=[training],
        generated_at="2026-08-03T00:00:00+00:00",
    )

    assert proof["schema_version"] == 1
    assert proof["dataset"]["hf_id"] == "org/data"
    assert proof["leaderboard"]["combined_score"] == 0.82
    assert proof["scores"][0]["scorer_name"] == "quality"
    assert proof["contamination"][0]["benchmark_name"] == "MMLU"
    assert proof["training"]["model_config"]["source_artifact"].endswith("loss_curve.json")
