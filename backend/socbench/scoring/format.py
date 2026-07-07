"""Format consistency scorer — schema conformance, encoding, code parseability.

Checks from FineWeb (Penedo et al., NeurIPS 2024) and RedPajama-V2.
"""

from __future__ import annotations

import ast
import json
from collections import Counter

from socbench.scoring.base import ScoreResult


async def format_scorer(samples: list[dict], text_key: str = "text") -> ScoreResult:
    """Check format consistency, encoding, and code parseability."""
    if not samples:
        return ScoreResult(name="format", score=0.0, warnings=["No samples"])

    total = len(samples)

    # --- Schema conformance: check expected fields exist ---
    key_counts: Counter = Counter()
    for s in samples:
        key_counts.update(s.keys())
    most_common_keys = [k for k, _ in key_counts.most_common(5)]
    schema_scores = []
    for key in most_common_keys:
        present = sum(1 for s in samples if key in s and s[key] is not None)
        schema_scores.append(present / total)
    schema_conformance = sum(schema_scores) / len(schema_scores) if schema_scores else 0.0

    # --- Encoding consistency: check for mojibake, null bytes ---
    encoding_issues = 0
    for s in samples:
        text = s.get(text_key, "")
        if isinstance(text, str):
            if "\x00" in text:
                encoding_issues += 1
            try:
                text.encode("utf-8").decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                encoding_issues += 1
    encoding_consistency = 1.0 - (encoding_issues / total) if total > 0 else 0.0

    # --- Null/empty rate ---
    empty_count = sum(
        1 for s in samples if not s.get(text_key, "").strip()
    )
    null_rate = empty_count / total if total > 0 else 0.0

    # --- Code parseability (for code datasets) ---
    code_parseable = None
    try:
        parseable = 0
        code_samples = 0
        for s in samples:
            text = s.get(text_key, "")
            if text and ("def " in text or "class " in text or "import " in text):
                code_samples += 1
                try:
                    ast.parse(text)
                    parseable += 1
                except SyntaxError:
                    pass
        if code_samples > 0:
            code_parseable = parseable / code_samples
    except Exception:
        pass

    # --- Structure consistency: uniform field structure ---
    field_sets = [frozenset(s.keys()) for s in samples]
    most_common_fields = Counter(field_sets).most_common(1)[0] if field_sets else (frozenset(), 0)
    structure_consistency = (
        most_common_fields[1] / total if total > 0 else 0.0
    )

    # Composite score
    scores = [schema_conformance, encoding_consistency, 1.0 - null_rate, structure_consistency]
    if code_parseable is not None:
        scores.append(code_parseable)

    score = sum(scores) / len(scores) if scores else 0.0

    warnings = []
    if null_rate > 0.1:
        warnings.append(f"High empty rate: {null_rate:.1%}")
    if encoding_issues > 0:
        warnings.append(f"Encoding issues in {encoding_issues} samples")

    return ScoreResult(
        name="format",
        score=score,
        details={
            "schema_conformance": round(schema_conformance, 4),
            "encoding_consistency": round(encoding_consistency, 4),
            "null_rate": round(null_rate, 4),
            "structure_consistency": round(structure_consistency, 4),
            "code_parseability": round(code_parseable, 4) if code_parseable is not None else None,
            "total_samples": total,
        },
        warnings=warnings,
    )
