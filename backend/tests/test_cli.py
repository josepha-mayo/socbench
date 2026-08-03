"""CLI surface tests for dataset scoring and auditing."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from socbench.audit.pipeline import AuditResult
from socbench.cli import app

runner = CliRunner()


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
    assert "kaggle" not in result.output.lower()
