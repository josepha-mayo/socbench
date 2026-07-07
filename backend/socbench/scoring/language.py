"""Language detection scorer — purity, mixed-language detection.

Uses fasttext-langdetect for high-accuracy language identification.
"""

from __future__ import annotations

from collections import Counter

from socbench.scoring.base import ScoreResult


async def language_scorer(
    samples: list[dict],
    text_key: str = "text",
    expected_language: str = "en",
) -> ScoreResult:
    """Detect language distribution and purity."""
    try:
        import fasttext  # noqa: F401
        from fasttext_langdetect import detect

        model = None  # fasttext-langdetect handles loading internally
    except (ImportError, OSError):
        return await _heuristic_language(samples, text_key, expected_language)

    texts = [s.get(text_key, "")[:500] for s in samples if s.get(text_key)]  # Truncate for speed
    if not texts:
        return ScoreResult(name="language", score=0.0, warnings=["No text content"])

    total = len(texts)
    lang_counts: Counter = Counter()

    for text in texts:
        clean = text.replace("\n", " ")[:500]
        try:
            result = detect(clean, low_memory=False)
            lang = result.get("lang", "unknown") if isinstance(result, dict) else "unknown"
            lang_counts[lang] += 1
        except Exception:
            lang_counts["unknown"] += 1

    # Language purity: fraction in expected language
    expected_count = lang_counts.get(expected_language, 0)
    purity = expected_count / total if total > 0 else 0.0

    # Mixed language rate: fraction in non-dominant languages
    dominant_lang = lang_counts.most_common(1)[0][0] if lang_counts else expected_language
    mixed_rate = 1.0 - (lang_counts.get(dominant_lang, 0) / total) if total > 0 else 0.0

    # Score: high purity = high score
    score = purity

    warnings = []
    if purity < 0.9:
        warnings.append(f"Low language purity: {purity:.1%} {expected_language}")
    if mixed_rate > 0.1:
        top_non_dominant = [
            (lang, count)
            for lang, count in lang_counts.most_common(5)
            if lang != dominant_lang
        ][:3]
        warnings.append(
            f"Mixed languages: {top_non_dominant}"
        )

    return ScoreResult(
        name="language",
        score=score,
        details={
            "expected_language": expected_language,
            "purity": round(purity, 4),
            "mixed_rate": round(mixed_rate, 4),
            "language_distribution": dict(lang_counts.most_common(10)),
            "total_samples": total,
        },
        warnings=warnings,
    )


async def _heuristic_language(
    samples: list[dict],
    text_key: str,
    expected_language: str,
) -> ScoreResult:
    """Fallback heuristic when fasttext not available."""
    texts = [s.get(text_key, "") for s in samples if s.get(text_key)]
    total = len(texts)
    if not texts:
        return ScoreResult(name="language", score=0.0, warnings=["No text content"])

    # Simple heuristic: check for ASCII dominance (rough proxy for English)
    ascii_count = 0
    for text in texts:
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        if ascii_ratio > 0.9:
            ascii_count += 1

    purity = ascii_count / total
    return ScoreResult(
        name="language",
        score=purity,
        details={
            "expected_language": expected_language,
            "purity": round(purity, 4),
            "method": "ascii_heuristic",
            "total_samples": total,
        },
        warnings=["fasttext-langdetect not available, using ASCII heuristic"],
    )
