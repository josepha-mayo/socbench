"""Quality scorer — Gopher rules + FineWeb filters + diversity metrics.

Research sources:
  - Gopher (Rae et al., 2021): repetition thresholds
  - FineWeb (Penedo et al., NeurIPS 2024): three custom filters
  - RedPajama-V2 (Weber et al., NeurIPS 2024): 46 quality signals
  - DCLM (Li et al., NeurIPS 2024): model-based filtering
"""

from __future__ import annotations

import re
from collections import Counter

from socbench.scoring.base import ScoreResult

# Gopher quality thresholds (Rae et al., 2021, Table A1)
GOPHER_LINE_DUP_THRESHOLD = 0.30
GOPHER_PARA_DUP_THRESHOLD = 0.30
GOPHER_LINE_CHAR_THRESHOLD = 0.20
GOPHER_PARA_CHAR_THRESHOLD = 0.20
GOPHER_2GRAM_THRESHOLD = 0.20
GOPHER_3GRAM_THRESHOLD = 0.18
GOPHER_4GRAM_THRESHOLD = 0.16

# FineWeb thresholds (Penedo et al., NeurIPS 2024)
FINEWEB_PUNCTUATION_THRESHOLD = 0.12
FINEWEB_DUPLICATED_LINE_CHARS_THRESHOLD = 0.10
FINEWEB_SHORT_LINES_THRESHOLD = 0.67


def _gopher_repetition_check(text: str) -> dict:
    """Apply Gopher repetition filters to a single document."""
    lines = text.split("\n")
    paragraphs = text.split("\n\n")

    issues = []

    # Duplicate line fraction
    if lines:
        line_counts = Counter(lines)
        dup_lines = sum(c - 1 for c in line_counts.values() if c > 1)
        dup_line_frac = dup_lines / len(lines)
        if dup_line_frac > GOPHER_LINE_DUP_THRESHOLD:
            issues.append(f"dup_line_frac={dup_line_frac:.3f}")

    # Duplicate paragraph fraction
    if paragraphs:
        para_counts = Counter(paragraphs)
        dup_paras = sum(c - 1 for c in para_counts.values() if c > 1)
        dup_para_frac = dup_paras / len(paragraphs)
        if dup_para_frac > GOPHER_PARA_DUP_THRESHOLD:
            issues.append(f"dup_para_frac={dup_para_frac:.3f}")

    # N-gram character fractions
    words = text.split()
    total_chars = len(text)
    if total_chars > 0 and len(words) >= 4:
        for n, threshold in [
            (2, GOPHER_2GRAM_THRESHOLD),
            (3, GOPHER_3GRAM_THRESHOLD),
            (4, GOPHER_4GRAM_THRESHOLD),
        ]:
            ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            if ngrams:
                ng_counts = Counter(ngrams)
                top_count = ng_counts.most_common(1)[0][1]
                # Approximate character fraction
                top_chars = top_count * (n * 5)  # rough avg word length
                frac = top_chars / total_chars
                if frac > threshold:
                    issues.append(f"top_{n}gram_frac={frac:.3f}")

    return {"issues": issues, "failed": len(issues) > 0}


def _fineweb_filters(text: str) -> dict:
    """Apply FineWeb custom filters (Penedo et al., NeurIPS 2024)."""
    lines = text.split("\n")
    if not lines:
        return {"failed": False, "issues": []}

    issues = []

    # Fraction of lines ending with punctuation
    punct_count = sum(
        1 for l in lines if l.strip() and l.strip()[-1] in ".!?:;,-"
    )
    punct_frac = punct_count / len(lines) if lines else 0
    if punct_frac <= FINEWEB_PUNCTUATION_THRESHOLD:
        issues.append(f"punct_frac={punct_frac:.3f}")

    # Fraction of characters in duplicated lines
    line_set = set()
    dup_chars = 0
    for line in lines:
        if line in line_set:
            dup_chars += len(line)
        line_set.add(line)
    dup_char_frac = dup_chars / max(len(text), 1)
    if dup_char_frac >= FINEWEB_DUPLICATED_LINE_CHARS_THRESHOLD:
        issues.append(f"dup_char_frac={dup_char_frac:.3f}")

    # Fraction of lines shorter than 30 characters
    short_count = sum(1 for l in lines if len(l.strip()) < 30)
    short_frac = short_count / len(lines) if lines else 0
    if short_frac >= FINEWEB_SHORT_LINES_THRESHOLD:
        issues.append(f"short_line_frac={short_frac:.3f}")

    return {"failed": len(issues) > 0, "issues": issues}


def _gopher_quality_check(text: str) -> dict:
    """Apply Gopher quality filters."""
    words = text.split()
    issues = []

    # Document length
    if len(words) < 50:
        issues.append("too_short")
    elif len(words) > 100_000:
        issues.append("too_long")

    # Average word length
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3 or avg_word_len > 10:
            issues.append(f"avg_word_len={avg_word_len:.1f}")

    # Symbol-to-word ratio
    symbols = sum(1 for c in text if c in "#*...")
    if words:
        sym_ratio = symbols / len(words)
        if sym_ratio > 0.1:
            issues.append(f"sym_ratio={sym_ratio:.3f}")

    # Stop words (simple check)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "have", "has", "had"}
    if words:
        stop_count = sum(1 for w in words if w.lower() in stop_words)
        if stop_count < 2:
            issues.append("few_stop_words")

    # Lines ending with ellipsis
    lines = text.split("\n")
    if lines:
        ellipsis_count = sum(1 for l in lines if l.strip().endswith("..."))
        ellipsis_frac = ellipsis_count / len(lines)
        if ellipsis_frac > 0.30:
            issues.append(f"ellipsis_frac={ellipsis_frac:.3f}")

    return {"failed": len(issues) > 0, "issues": issues}


async def quality_scorer(
    samples: list[dict],
    text_key: str = "text",
    category: str | None = None,
) -> ScoreResult:
    """Run Gopher quality + FineWeb filters + diversity metrics.

    Length-based rules (Gopher document length, FineWeb short-line) are
    appropriate for long-form pretraining text but unfair to instruction /
    QA / evaluation datasets, which legitimately contain short samples.
    Those penalties are skipped for non-pretraining categories.
    """
    texts = [s.get(text_key, "") for s in samples if s.get(text_key)]
    if not texts:
        return ScoreResult(name="quality", score=0.0, warnings=["No text content"])

    is_pretraining = bool(category) and category.startswith("pretraining")

    total = len(texts)
    gopher_failures = 0
    fineweb_failures = 0
    quality_failures = 0
    all_issues: list[str] = []

    for text in texts:
        gopher = _gopher_repetition_check(text)
        fineweb = _fineweb_filters(text)
        quality = _gopher_quality_check(text)

        # Skip web-text length penalties for non-pretraining data.
        if not is_pretraining:
            quality["issues"] = [
                i for i in quality["issues"] if i not in ("too_short", "too_long")
            ]
            fineweb["issues"] = [
                i for i in fineweb["issues"]
                if not i.startswith("short_line_frac")
            ]
            quality["failed"] = len(quality["issues"]) > 0
            fineweb["failed"] = len(fineweb["issues"]) > 0

        if gopher["failed"]:
            gopher_failures += 1
        if fineweb["failed"]:
            fineweb_failures += 1
        if quality["failed"]:
            quality_failures += 1

        all_issues.extend(gopher["issues"][:2])
        all_issues.extend(fineweb["issues"][:2])

    gopher_pass_rate = 1.0 - (gopher_failures / total) if total > 0 else 0.0
    fineweb_pass_rate = 1.0 - (fineweb_failures / total) if total > 0 else 0.0
    quality_pass_rate = 1.0 - (quality_failures / total) if total > 0 else 0.0

    # Diversity: unique word ratio
    all_words = []
    for t in texts:
        all_words.extend(t.lower().split())
    unique_words = len(set(all_words))
    diversity = unique_words / max(len(all_words), 1)

    # Score: weighted combination
    score = (
        0.35 * gopher_pass_rate
        + 0.30 * fineweb_pass_rate
        + 0.20 * quality_pass_rate
        + 0.15 * min(diversity * 2, 1.0)  # Scale diversity
    )

    warnings = []
    if gopher_failures / total > 0.2:
        warnings.append(f"High Gopher failure rate: {gopher_failures / total:.1%}")
    if fineweb_failures / total > 0.3:
        warnings.append(f"High FineWeb failure rate: {fineweb_failures / total:.1%}")

    # Top issues
    issue_counts = Counter(all_issues)
    top_issues = [f"{k}({v})" for k, v in issue_counts.most_common(5)]

    return ScoreResult(
        name="quality",
        score=score,
        details={
            "gopher_pass_rate": round(gopher_pass_rate, 4),
            "fineweb_pass_rate": round(fineweb_pass_rate, 4),
            "quality_pass_rate": round(quality_pass_rate, 4),
            "diversity": round(diversity, 4),
            "gopher_failures": gopher_failures,
            "fineweb_failures": fineweb_failures,
            "quality_failures": quality_failures,
            "top_issues": top_issues,
            "total_samples": total,
        },
        warnings=warnings,
    )
