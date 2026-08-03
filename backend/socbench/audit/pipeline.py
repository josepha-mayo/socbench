"""6-stage dataset audit pipeline.

Stages:
  1. License allow-list filtering
  2. Language detection (fasttext / HF tags)
  3. Tree-sitter syntax validation for code (optional)
  4. MinHash LSH decontamination against eval banks (optional)
  5. Exact + near-duplicate detection
  6. Token-length filter
  7. Per-language water-filling rebalancing

Output:
  ``<output_dir>/audit_balanced.jsonl``
  ``<output_dir>/audit_summary.json``
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from socbench.audit.decontamination import EvalBankIndex
from socbench.audit.dedup import DedupState
from socbench.audit.license import is_license_allowed
from socbench.audit.rebalance import RebalanceResult, water_filling_rebalance
from socbench.audit.syntax import validate_code_syntax
from socbench.audit.tokens import make_token_filter
from socbench.runner import fetch_metadata, fetch_samples

LOGGER = logging.getLogger("socbench.audit")

TEXT_KEYS = ("text", "content", "document", "instruction", "prompt", "input",
             "question", "answer", "response", "completion", "code", "source",
             "sentence", "paragraph", "abstract", "body", "article", "txt")


def _extract_text(row: dict[str, Any], text_key: str | None = None) -> str:
    """Best-effort text extraction from a dataset row."""
    if not isinstance(row, dict):
        return ""
    if text_key is not None:
        val = row.get(text_key)
        return val.strip() if isinstance(val, str) else ""
    for key in TEXT_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _detect_language(text: str) -> str | None:
    """FastText language detection; returns None if unavailable."""
    try:
        from fasttext_langdetect import detect

        result = detect(text.replace("\n", " ")[:500], low_memory=False)
        if isinstance(result, dict):
            return result.get("lang")
    except Exception:
        pass
    return None


@dataclass
class AuditConfig:
    """Configuration for ``audit_dataset``."""

    output_dir: Path
    max_rows: int = 100_000
    output_size: int | None = None
    text_key: str | None = None
    min_tokens: int = 32
    max_tokens: int = 8192
    char_prefilter: int = 60_000
    near_dup_threshold: float = 0.8
    minhash_perm: int = 128
    eval_bank_dir: Path | str | None = None
    target_ratios: dict[str, float] | None = None
    code_target_ratios: dict[str, float] | None = field(default_factory=dict)
    seed: int = 42
    source_default_ok: bool = True
    skip_tree_sitter: bool = False
    skip_eval_decontamination: bool = False

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.eval_bank_dir is not None and isinstance(self.eval_bank_dir, str):
            self.eval_bank_dir = Path(self.eval_bank_dir)


@dataclass
class AuditResult:
    """Summary returned by ``audit_dataset``."""

    dataset_id: str
    accepted: int
    rejected_license: int
    rejected_language: int
    rejected_syntax: int
    rejected_eval_contamination: int
    rejected_exact_dup: int
    rejected_near_dup: int
    rejected_token_length: int
    rejected_other: int
    final: int
    language_counts: dict[str, int] = field(default_factory=dict)
    language_percentages: dict[str, float] = field(default_factory=dict)
    shortfalls: dict[str, int] = field(default_factory=dict)
    output_path: Path | None = None
    summary_path: Path | None = None
    eval_bank_entries: int = 0
    license: str | None = None
    elapsed_seconds: float = 0.0


async def _stream_with_datasets(dataset_id: str, max_rows: int, text_key: str | None) -> list[dict[str, Any]]:
    """Stream up to ``max_rows`` rows from a HF dataset using ``datasets``."""

    def _load():
        from datasets import get_dataset_config_names, load_dataset

        configs = []
        try:
            cfg_names = get_dataset_config_names(dataset_id, trust_remote_code=False)
            if cfg_names:
                for c in ("en", "english", "default", "plain_text"):
                    if c in cfg_names and c not in configs:
                        configs.append(c)
                for c in cfg_names:
                    if c not in configs:
                        configs.append(c)
        except Exception:
            pass

        rows: list[dict[str, Any]] = []
        candidates = configs if configs else [None]
        for config in candidates:
            try:
                if config is None:
                    ds = load_dataset(dataset_id, streaming=True, split="train", trust_remote_code=False)
                else:
                    ds = load_dataset(dataset_id, config, streaming=True, split="train", trust_remote_code=False)
                for i, ex in enumerate(ds):
                    if i >= max_rows:
                        break
                    rows.append(ex)
                if rows:
                    return rows
            except Exception as exc:
                LOGGER.debug("datasets streaming failed for config %s: %s", config, exc)
                continue
        return rows

    return await asyncio.to_thread(_load)


async def _fetch_rows(
    dataset_id: str,
    max_rows: int,
    text_key: str | None,
) -> list[dict[str, Any]]:
    """Fetch rows, trying the HF viewer first and then ``datasets`` streaming."""
    try:
        rows = await fetch_samples(dataset_id, text_key=text_key or "text", sample_size=max_rows)
        if len(rows) >= max_rows * 0.5:
            return rows
    except Exception as exc:
        LOGGER.debug("HF viewer fetch failed: %s", exc)

    return await _stream_with_datasets(dataset_id, max_rows, text_key)


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def _run_audit(
    dataset_id: str,
    config: AuditConfig,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> AuditResult:
    """Core of the audit pipeline."""
    start = datetime.now(timezone.utc)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "audit_balanced.jsonl"
    summary_path = output_dir / "audit_summary.json"

    result = AuditResult(dataset_id=dataset_id, accepted=0, rejected_license=0,
                         rejected_language=0, rejected_syntax=0,
                         rejected_eval_contamination=0, rejected_exact_dup=0,
                         rejected_near_dup=0, rejected_token_length=0,
                         rejected_other=0, final=0, license=metadata.get("license"))

    # Stage 1: license
    license_text = metadata.get("license")
    if not is_license_allowed(license_text, source_default_ok=config.source_default_ok):
        _write_jsonl([], output_path)
        result.rejected_license = len(rows)
        result.output_path = output_path
        result.summary_path = summary_path
        _write_summary(result, summary_path)
        return result

    # Stages 3 & 4 prep
    eval_index: EvalBankIndex | None = None
    if not config.skip_eval_decontamination:
        if config.eval_bank_dir:
            eval_index = EvalBankIndex(eval_dir=config.eval_bank_dir,
                                       threshold=config.near_dup_threshold,
                                       num_perm=config.minhash_perm)
        else:
            eval_index = EvalBankIndex(threshold=config.near_dup_threshold,
                                       num_perm=config.minhash_perm,
                                       use_hf_benchmarks=True)
            await eval_index._build_from_hf(limit=500)
        result.eval_bank_entries = eval_index.num_entries

    dedup = DedupState(threshold=config.near_dup_threshold, num_perm=config.minhash_perm)
    token_filter = make_token_filter(
        min_tokens=config.min_tokens,
        max_tokens=config.max_tokens,
        char_prefilter=config.char_prefilter,
    )

    accepted: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        text = _extract_text(row, config.text_key)
        if not text:
            result.rejected_other += 1
            continue

        # Stage 2: language detection
        language = _detect_language(text)
        if language is None:
            # Fallback to HF language tags if fasttext is missing.
            tags = metadata.get("tags", [])
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("language:"):
                    language = tag.split(":", 1)[1]
                    break
            if language is None:
                result.rejected_language += 1
                continue

        # Stage 3: syntax validation (optional)
        if not config.skip_tree_sitter and not validate_code_syntax(text, language):
            result.rejected_syntax += 1
            continue

        # Stage 4: eval-bank decontamination
        if eval_index is not None and eval_index.is_contaminated(text):
            result.rejected_eval_contamination += 1
            continue

        # Stage 5: exact + near dedup
        record_id = f"{dataset_id}::{i}"
        dup = dedup.is_duplicate(text, record_id)
        if dup:
            if dup == "exact":
                result.rejected_exact_dup += 1
            else:
                result.rejected_near_dup += 1
            continue

        # Stage 6: token length filter
        tf = token_filter(text)
        if not tf.ok:
            result.rejected_token_length += 1
            continue

        accepted.append({
            **row,
            "__audit_language": language,
            "__audit_tokens": tf.token_count,
        })

    result.accepted = len(accepted)

    # Stage 7: per-language water-filling rebalancing
    target_ratios = config.target_ratios
    if target_ratios is None and config.code_target_ratios:
        # Simple heuristic: if the dataset is dominated by code-like rows, use code targets.
        code_like_count = sum(1 for r in accepted if "__audit_tokens" in r)
        if code_like_count >= len(accepted) * 0.5:
            target_ratios = config.code_target_ratios

    rebalance: RebalanceResult = water_filling_rebalance(
        accepted,
        target_ratios=target_ratios,
        output_size=config.output_size,
        seed=config.seed,
    )

    # Write outputs
    _write_jsonl(rebalance.records, output_path)

    result.final = len(rebalance.records)
    result.language_counts = rebalance.language_counts
    result.language_percentages = rebalance.language_percentages
    result.shortfalls = rebalance.shortfalls
    result.output_path = output_path
    result.summary_path = summary_path
    result.elapsed_seconds = (datetime.now(timezone.utc) - start).total_seconds()

    _write_summary(result, summary_path)
    return result


def _write_summary(result: AuditResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, default=str)


async def audit_dataset(
    dataset_id: str,
    output_dir: Path | str = "audit_outputs",
    max_rows: int = 100_000,
    output_size: int | None = None,
    text_key: str | None = None,
    min_tokens: int = 32,
    max_tokens: int = 8192,
    char_prefilter: int = 60_000,
    near_dup_threshold: float = 0.8,
    minhash_perm: int = 128,
    eval_bank_dir: Path | str | None = None,
    target_ratios: dict[str, float] | None = None,
    code_target_ratios: dict[str, float] | None = None,
    seed: int = 42,
    source_default_ok: bool = True,
    skip_tree_sitter: bool = False,
    skip_eval_decontamination: bool = False,
) -> AuditResult:
    """Run the full 6-stage audit on a HuggingFace dataset.

    Returns an ``AuditResult`` summary and writes ``audit_balanced.jsonl`` plus
    ``audit_summary.json`` under ``output_dir``.
    """
    config = AuditConfig(
        output_dir=output_dir,
        max_rows=max_rows,
        output_size=output_size,
        text_key=text_key,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        char_prefilter=char_prefilter,
        near_dup_threshold=near_dup_threshold,
        minhash_perm=minhash_perm,
        eval_bank_dir=eval_bank_dir,
        target_ratios=target_ratios,
        code_target_ratios=code_target_ratios,
        seed=seed,
        source_default_ok=source_default_ok,
        skip_tree_sitter=skip_tree_sitter,
        skip_eval_decontamination=skip_eval_decontamination,
    )

    metadata = await fetch_metadata(dataset_id)
    rows = await _fetch_rows(dataset_id, config.max_rows, config.text_key)

    if not rows:
        return AuditResult(
            dataset_id=dataset_id,
            accepted=0, rejected_license=0, rejected_language=0, rejected_syntax=0,
            rejected_eval_contamination=0, rejected_exact_dup=0, rejected_near_dup=0,
            rejected_token_length=0, rejected_other=0, final=0,
            license=metadata.get("license"),
        )

    return await _run_audit(dataset_id, config, metadata, rows)
