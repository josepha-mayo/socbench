"""Focused API route contract tests."""

import inspect

import pytest
from pydantic import ValidationError

from socbench.api.routes import EvalRequest, get_dataset, get_training_leaderboard, list_datasets


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
