"""Focused API route contract tests."""

import inspect
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from socbench.api.routes import (
    EvalRequest,
    _training_payload,
    get_dataset,
    get_training_leaderboard,
    list_datasets,
)
from socbench.models import TrainingRunRow


def test_eval_request_normalizes_and_bounds_public_input():
    request = EvalRequest(
        hf_id="  Salesforce/wikitext  ",
        requester_email="  maintainer@example.com  ",
        notes="  please evaluate  ",
    )

    assert request.hf_id == "Salesforce/wikitext"
    assert request.requester_email == "maintainer@example.com"
    assert request.notes == "please evaluate"

    with pytest.raises(ValidationError):
        EvalRequest(hf_id="invalid dataset id")
    with pytest.raises(ValidationError):
        EvalRequest(hf_id="owner/dataset", visibility="secret")
    with pytest.raises(ValidationError):
        EvalRequest(hf_id="owner/dataset", notes="x" * 4_001)


def test_dataset_routes_select_latest_training_run_and_apply_category():
    detail_source = inspect.getsource(get_dataset)
    training_source = inspect.getsource(get_training_leaderboard)
    list_source = inspect.getsource(list_datasets)

    assert "TrainingRunRow.trained_at.desc()" in detail_source
    assert ".limit(1)" in detail_source
    assert "latest_training_id" in training_source
    assert "TrainingRunRow.id == latest_training_id" in training_source
    assert "LeaderboardRow.category == category" in list_source


def test_training_payload_exposes_outcome_and_avoids_unbounded_perplexity():
    run = TrainingRunRow(
        final_val_loss=66.5,
        loss_curve=[10.9, 66.5],
        convergence_steps=0,
        tokens_seen=1_000_000_000,
        model_config={"completed_steps": 1906, "distributed_world_size": 2},
        eval_scores={
            "initial_val_loss": 10.9,
            "best_val_loss": 10.9,
            "relative_improvement": -5.1,
            "run_outcome": "diverged",
            "outcome_reason": "validation loss increased",
            "training_score": 0.0,
        },
        trained_at=datetime(2026, 8, 9, tzinfo=timezone.utc),
    )

    payload = _training_payload(run)

    assert payload is not None
    assert payload["run_outcome"] == "diverged"
    assert payload["initial_val_loss"] == 10.9
    assert payload["completed_steps"] == 1906
    assert payload["perplexity"] is None
