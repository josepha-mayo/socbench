"""Tests for audit eval-bank decontamination."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socbench.audit.decontamination import EvalBankIndex, get_word_ngrams, normalize_text


def test_normalize_text_collapses_case_and_whitespace():
    assert normalize_text("  Hello\n   WORLD  ") == "hello world"


def test_get_word_ngrams_uses_short_text_as_single_gram():
    assert get_word_ngrams("one two", n=13) == {"one two"}


def test_eval_bank_index_loads_jsonl_and_detects_overlap(tmp_path: Path):
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    prompt = " ".join(f"token{i}" for i in range(20))
    (eval_dir / "bench.jsonl").write_text(
        json.dumps({"question": prompt}) + "\n",
        encoding="utf-8",
    )

    index = EvalBankIndex(eval_dir=eval_dir, threshold=0.8)

    assert index.num_entries == 1
    assert index.is_contaminated(prompt)
    assert index.overlap_ratio(prompt) == 1.0
    assert not index.is_contaminated("totally unrelated text with different words")


@pytest.mark.asyncio
async def test_eval_bank_index_build_from_hf_can_be_monkeypatched(monkeypatch):
    async def fake_fetch_eval_samples(dataset_id: str, limit: int = 500, token=None):
        return [" ".join(f"sample{i}" for i in range(20))]

    monkeypatch.setattr(
        "socbench.contamination.checker._fetch_eval_samples",
        fake_fetch_eval_samples,
    )
    monkeypatch.setattr(
        "socbench.contamination.checker.BENCHMARKS",
        {"demo": "demo/bench"},
    )

    index = EvalBankIndex()
    await index._build_from_hf(limit=10)

    assert index.num_entries == 1
    assert index.is_contaminated(" ".join(f"sample{i}" for i in range(20)))
