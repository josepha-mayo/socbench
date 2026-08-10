"""CLI surface tests for dataset scoring and auditing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

import socbench.cli as cli_module
from socbench.audit.pipeline import AuditResult
from socbench.cli import app
from socbench.dataset_intelligence import (
    SCORE_DIMENSIONS,
    DatasetEvidence,
    DimensionEvidence,
)

runner = CliRunner()


def _evidence(
    dataset_id: str = "owner/dataset",
    *,
    score: float = 0.8,
    source: str = "cache",
) -> DatasetEvidence:
    dimensions = {
        name: DimensionEvidence(score=score, details={"fixture": True})
        for name in SCORE_DIMENSIONS
    }
    dimensions["pii_safety"] = DimensionEvidence(score=0.99, details={"pii_rate": 0.0})
    return DatasetEvidence(
        dataset_id=dataset_id,
        name=dataset_id,
        category="pretraining-web",
        license="apache-2.0",
        dimensions=dimensions,
        auto_score=score,
        contamination_rate=0.0,
        contamination_checks=({"benchmark": "fixture", "overlap_rate": 0.0},),
        repetition_pct=2.0,
        training_score=None,
        training_outcome=None,
        training_reason=None,
        last_scored=datetime(2026, 8, 1, tzinfo=timezone.utc),
        complete=True,
        missing_evidence=(),
        source=source,
    )


def test_audit_command_runs_full_pipeline(monkeypatch, tmp_path: Path):
    async def fake_audit(dataset_id: str, **kwargs):
        assert dataset_id == "owner/dataset"
        assert kwargs["max_rows"] == 250
        assert kwargs["output_size"] == 100
        assert kwargs["output_dir"] == tmp_path
        return AuditResult(
            dataset_id=dataset_id,
            accepted=120,
            rejected_license=0,
            rejected_language=2,
            rejected_syntax=3,
            rejected_eval_contamination=4,
            rejected_exact_dup=5,
            rejected_near_dup=6,
            rejected_token_length=7,
            rejected_other=0,
            final=100,
            output_path=tmp_path / "audit_balanced.jsonl",
            summary_path=tmp_path / "audit_summary.json",
        )

    monkeypatch.setattr("socbench.audit.audit_dataset", fake_audit)

    result = runner.invoke(
        app,
        [
            "audit",
            "owner/dataset",
            "--output-dir",
            str(tmp_path),
            "--max-rows",
            "250",
            "--output-size",
            "100",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Audit complete: owner/dataset" in result.output
    assert "Rejected: eval contamination" in result.output
    assert "audit_balanced.jsonl" in result.output


def test_audit_command_is_listed_in_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "audit" in result.output
    assert "seven-stage" in result.output


def test_readiness_json_is_machine_readable_and_forwards_cache_controls(monkeypatch):
    async def fake_load(dataset_id: str, **kwargs):
        assert dataset_id == "owner/dataset"
        assert kwargs == {
            "sample_size": 750,
            "refresh": False,
            "cached_only": True,
            "token": None,
        }
        return _evidence()

    monkeypatch.setattr(cli_module, "_load_dataset_evidence", fake_load)
    result = runner.invoke(
        app,
        [
            "--no-banner",
            "readiness",
            "owner/dataset",
            "--sample-size",
            "750",
            "--cached-only",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["dataset_id"] == "owner/dataset"
    assert payload["source"] == "cache"
    assert payload["readiness"]["decision"] == "PROCEED"
    assert "___  ___" not in result.stdout


def test_ascii_banner_precedes_human_command_output(monkeypatch):
    async def fake_load(dataset_id: str, **kwargs):
        return _evidence(dataset_id)

    monkeypatch.setattr(cli_module, "_load_dataset_evidence", fake_load)
    result = runner.invoke(app, ["readiness", "owner/dataset"])

    assert result.exit_code == 0, result.output
    assert "___  ___" in result.output
    assert result.output.index("___  ___") < result.output.index("Training-readiness decision")


def test_compare_orders_cached_datasets_without_network(monkeypatch):
    requested: list[str] = []

    async def no_op():
        return None

    async def fake_get(dataset_id: str, **kwargs):
        requested.append(dataset_id)
        assert kwargs["cached_only"] is True
        return _evidence(dataset_id, score=0.9 if dataset_id.endswith("better") else 0.75)

    monkeypatch.setattr(cli_module, "_prepare_database", no_op)
    monkeypatch.setattr(cli_module, "_dispose_database", no_op)
    monkeypatch.setattr("socbench.dataset_intelligence.get_or_score_dataset", fake_get)

    result = runner.invoke(
        app,
        [
            "--no-banner",
            "compare",
            "owner/good",
            "owner/better",
            "--cached-only",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert requested == ["owner/good", "owner/better"]
    assert [item["dataset_id"] for item in payload["datasets"]] == [
        "owner/better",
        "owner/good",
    ]


def test_compare_rejects_fewer_than_two_unique_datasets():
    result = runner.invoke(
        app,
        ["--no-banner", "compare", "owner/one", "owner/one", "--cached-only"],
    )

    assert result.exit_code == 2
    assert "between 2 and 20 unique dataset IDs" in result.output
