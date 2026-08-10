"""Semantic benchmark-quality audit tests."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from socbench.api.routes import run_benchmark_audit
from socbench.cli import app
from socbench.evals import semantic_audit
from socbench.evals.semantic_audit import BenchmarkAuditInput, audit_benchmark

POLICY_ITEMS = [
    "The agent must not cancel an order after shipment.",
    "Refunds must return to the original payment method.",
]


def _request(*, swapped: bool = False, use_api_judge: bool = False) -> BenchmarkAuditInput:
    expected = [
        "The agent must refuse to cancel the shipped order.",
        "Return the refund to the original payment card.",
    ]
    if swapped:
        expected.reverse()
    return BenchmarkAuditInput(
        name="Retail benchmark",
        policy_items=POLICY_ITEMS,
        coverage_threshold=1,
        use_api_judge=use_api_judge,
        tasks=[
            {
                "id": "cancel-shipped",
                "description": "The user asks to cancel a shipped order.",
                "expected_behavior": expected[0],
            },
            {
                "id": "refund-card",
                "description": "The user requests a card refund.",
                "expected_behavior": expected[1],
            },
        ],
    )


def test_local_semantics_detect_alignment_and_policy_coverage(monkeypatch):
    monkeypatch.delenv("SOCBENCH_SEMANTIC_REMOTE_ENABLED", raising=False)

    aligned = asyncio.run(audit_benchmark(_request()))
    swapped = asyncio.run(audit_benchmark(_request(swapped=True)))

    assert aligned["engine"] == "local_semantic_v1"
    assert aligned["scores"]["description_expected_alignment"] > 70
    assert aligned["scores"]["policy_expected_alignment"] > 75
    assert aligned["tasks"][0]["violated_policy_items"] == [0]
    assert aligned["scores"]["policy_violation_coverage"] == 50.0
    assert aligned["scores"]["semantic_quality"] > swapped["scores"]["semantic_quality"] + 40


def test_policy_text_is_split_and_input_is_bounded():
    request = BenchmarkAuditInput(
        policy="1. Verify identity before account changes.\n2. Never disclose a password.",
        tasks=[{"description": "Change my account", "expected_behavior": "Verify identity first"}],
        use_api_judge=False,
    )
    result = asyncio.run(audit_benchmark(request))

    assert result["policy_item_count"] == 2
    with pytest.raises(ValidationError):
        BenchmarkAuditInput(policy="Policy", tasks=[])
    with pytest.raises(ValidationError):
        BenchmarkAuditInput(
            policy_items=[f"Policy {index}" for index in range(201)],
            tasks=[{"description": "Task", "expected_behavior": "Expected"}],
        )


def test_optional_api_judge_is_server_configured_and_never_leaks_key(monkeypatch):
    monkeypatch.setenv("SOCBENCH_SEMANTIC_REMOTE_ENABLED", "true")
    monkeypatch.setenv("SOCBENCH_SEMANTIC_API_URL", "https://judge.example/v1/chat/completions")
    monkeypatch.setenv("SOCBENCH_SEMANTIC_MODEL", "judge-model")
    monkeypatch.setenv("SOCBENCH_SEMANTIC_API_KEY", "super-secret-key")

    async def fake_judge(client, config, task, policy_items):
        assert config.api_key == "super-secret-key"
        return {
            "description_expected_alignment": 90,
            "policy_expected_alignment": 95,
            "violated_policy_items": [0],
            "diagnostics": ["API diagnostic"],
        }

    monkeypatch.setattr(semantic_audit, "_request_api_judgment", fake_judge)
    result = asyncio.run(audit_benchmark(_request(use_api_judge=True)))

    assert result["engine"] == "api_enhanced_semantic_v1"
    assert result["api_judge_enhanced_tasks"] == 2
    assert "super-secret-key" not in json.dumps(result)


def test_invalid_api_judge_limits_fall_back_to_local_scoring(monkeypatch):
    monkeypatch.setenv("SOCBENCH_SEMANTIC_REMOTE_ENABLED", "true")
    monkeypatch.setenv("SOCBENCH_SEMANTIC_API_URL", "https://judge.example/v1/chat/completions")
    monkeypatch.setenv("SOCBENCH_SEMANTIC_MODEL", "judge-model")
    monkeypatch.setenv("SOCBENCH_SEMANTIC_API_KEY", "super-secret-key")
    monkeypatch.setenv("SOCBENCH_SEMANTIC_TIMEOUT_SECONDS", "not-a-number")

    result = asyncio.run(audit_benchmark(_request(use_api_judge=True)))

    assert result["engine"] == "local_semantic_v1"
    assert result["warnings"] == ["Remote semantic judge limits are invalid; local scoring was used."]
    assert "super-secret-key" not in json.dumps(result)


def test_api_route_uses_the_same_audit_contract():
    result = asyncio.run(run_benchmark_audit(_request()))

    assert result["benchmark_name"] == "Retail benchmark"
    assert result["methodology"]["reference"].endswith("arXiv:2608.06329")


def test_benchmark_audit_cli_writes_machine_readable_result(tmp_path):
    source = tmp_path / "benchmark.json"
    output = tmp_path / "audit.json"
    source.write_text(_request().model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["benchmark-audit", str(source), "--local-only", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "Semantic quality" in result.output
    assert json.loads(output.read_text(encoding="utf-8"))["engine"] == "local_semantic_v1"
