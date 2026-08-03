"""Evaluation-bank decontamination for the audit pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

NGRAM_SIZE = 13
TEXT_KEYS = {
    "answer",
    "article",
    "body",
    "completion",
    "content",
    "document",
    "input",
    "instruction",
    "output",
    "problem",
    "prompt",
    "question",
    "response",
    "text",
}


def normalize_text(text: str) -> str:
    """Normalize text for contamination matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def get_word_ngrams(text: str, n: int = NGRAM_SIZE) -> set[str]:
    """Return normalized word n-grams, with short text as a single gram."""
    words = normalize_text(text).split()
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def extract_text_values(value: Any) -> Iterable[str]:
    """Yield likely text fields from nested JSON-like values."""
    if isinstance(value, str):
        if value.strip():
            yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from extract_text_values(item)
        return
    if isinstance(value, dict):
        matched = False
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in TEXT_KEYS:
                matched = True
                yield from extract_text_values(item)
        if not matched:
            for item in value.values():
                yield from extract_text_values(item)


@dataclass
class EvalBankIndex:
    """N-gram index for filtering rows that overlap evaluation banks."""

    threshold: float = 0.8
    num_perm: int = 128
    eval_dir: Path | None = None
    use_hf_benchmarks: bool = True
    ngram_size: int = NGRAM_SIZE

    def __post_init__(self):
        self.num_entries = 0
        self._ngrams: set[str] = set()
        self._texts: set[str] = set()
        if self.eval_dir is not None:
            self.eval_dir = Path(self.eval_dir)
            self._build_from_dir(self.eval_dir)

    def _add_text(self, text: str) -> None:
        normalized = normalize_text(text)
        if not normalized:
            return
        self._texts.add(normalized)
        self._ngrams.update(get_word_ngrams(normalized, self.ngram_size))
        self.num_entries += 1

    def _load_json_file(self, path: Path) -> None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for text in extract_text_values(value):
            self._add_text(text)

    def _load_jsonl_file(self, path: Path) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                self._add_text(line)
                continue
            for text in extract_text_values(value):
                self._add_text(text)

    def _load_text_file(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return
        self._add_text(text)

    def _build_from_dir(self, eval_dir: Path) -> None:
        """Load local eval-bank files from a directory."""
        if not eval_dir.exists():
            return
        for path in sorted(item for item in eval_dir.rglob("*") if item.is_file()):
            suffix = path.suffix.lower()
            if suffix == ".json":
                self._load_json_file(path)
            elif suffix == ".jsonl":
                self._load_jsonl_file(path)
            elif suffix in {".txt", ".md"}:
                self._load_text_file(path)

    async def _build_from_hf(self, limit: int = 500):
        """Load common HF benchmark samples into the index."""
        if not self.use_hf_benchmarks:
            return
        from socbench.contamination.checker import BENCHMARKS, _fetch_eval_samples

        per_benchmark = max(1, limit // max(len(BENCHMARKS), 1))
        for benchmark_id in BENCHMARKS.values():
            texts = await _fetch_eval_samples(benchmark_id, limit=per_benchmark)
            for text in texts:
                self._add_text(text)

    def overlap_ratio(self, text: str) -> float:
        """Return candidate n-gram overlap against the eval bank."""
        normalized = normalize_text(text)
        if not normalized:
            return 0.0
        if normalized in self._texts:
            return 1.0
        grams = get_word_ngrams(normalized, self.ngram_size)
        if not grams or not self._ngrams:
            return 0.0
        return len(grams & self._ngrams) / len(grams)

    def is_contaminated(self, text: str) -> bool:
        """Return True if text substantially overlaps the eval bank."""
        return self.overlap_ratio(text) >= self.threshold
