"""PII detection scorer — density by type, severity weighting.

Uses presidio-analyzer for regex + NER-based PII detection.
"""

from __future__ import annotations

from socbench.scoring.base import ScoreResult

# PII type severity weights (higher = more dangerous to have in training data)
SEVERITY_WEIGHTS = {
    "PERSON": 0.3,
    "EMAIL_ADDRESS": 0.5,
    "PHONE_NUMBER": 0.6,
    "CREDIT_CARD": 0.9,
    "US_SSN": 0.95,
    "IP_ADDRESS": 0.4,
    "LOCATION": 0.3,
    "MEDICAL_LICENSE": 0.7,
    "US_PASSPORT": 0.85,
    "US_DRIVER_LICENSE": 0.8,
    "UK_NHS": 0.85,
    "AU_ABN": 0.7,
}


async def pii_scorer(samples: list[dict], text_key: str = "text", use_presidio: bool = False) -> ScoreResult:
    """Detect PII density and compute safety score.

    Default: regex-based (fast, no model download).
    Set use_presidio=True for presidio-analyzer (requires 400MB spaCy model download).
    """
    if use_presidio:
        return await _presidio_pii(samples, text_key)
    return await _regex_pii(samples, text_key)


async def _presidio_pii(samples: list[dict], text_key: str) -> ScoreResult:
    """Full presidio-analyzer PII detection (requires spaCy en_core_web_lg)."""
    try:
        from presidio_analyzer import AnalyzerEngine
        analyzer = AnalyzerEngine()
    except (ImportError, OSError) as e:
        return ScoreResult(name="pii", score=0.5, warnings=[f"presidio unavailable: {e}"], details={"method": "presidio_fallback"})

    texts = [s.get(text_key, "") for s in samples if s.get(text_key)]
    if not texts:
        return ScoreResult(name="pii", score=1.0, warnings=["No text content"])

    total = len(texts)
    pii_counts_by_type: dict[str, int] = {}
    samples_with_pii = 0
    total_pii_entities = 0

    for text in texts:
        results = analyzer.analyze(text=text, language="en")
        if results:
            samples_with_pii += 1
        for r in results:
            pii_counts_by_type[r.entity_type] = pii_counts_by_type.get(r.entity_type, 0) + 1
            total_pii_entities += 1

    pii_rate = samples_with_pii / total if total > 0 else 0.0
    weighted_sum = sum(count * SEVERITY_WEIGHTS.get(pii_type, 0.5) for pii_type, count in pii_counts_by_type.items())
    density = weighted_sum / total if total > 0 else 0.0
    score = max(0.0, 1.0 - (density / 0.01))

    warnings = []
    if pii_rate > 0.05:
        warnings.append(f"High PII rate: {pii_rate:.1%}")
    if "CREDIT_CARD" in pii_counts_by_type:
        warnings.append(f"Credit card numbers: {pii_counts_by_type['CREDIT_CARD']}")
    if "US_SSN" in pii_counts_by_type:
        warnings.append(f"SSNs: {pii_counts_by_type['US_SSN']}")

    return ScoreResult(
        name="pii", score=score,
        details={"pii_rate": round(pii_rate, 4), "total_pii_entities": total_pii_entities, "pii_by_type": pii_counts_by_type, "weighted_density": round(density, 6), "total_samples": total, "method": "presidio"},
        warnings=warnings,
    )


async def _regex_pii(samples: list[dict], text_key: str) -> ScoreResult:
    """Fallback regex-based PII detection."""
    import re

    patterns = {
        "EMAIL_ADDRESS": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "PHONE_NUMBER": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
        "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "IP_ADDRESS": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    }

    texts = [s.get(text_key, "") for s in samples if s.get(text_key)]
    total = len(texts)
    if not texts:
        return ScoreResult(name="pii", score=1.0, warnings=["No text content"])

    pii_counts: dict[str, int] = {}
    samples_with_pii = 0

    for text in texts:
        found_in_sample = False
        for pii_type, pattern in patterns.items():
            matches = pattern.findall(text)
            if matches:
                pii_counts[pii_type] = pii_counts.get(pii_type, 0) + len(matches)
                found_in_sample = True
        if found_in_sample:
            samples_with_pii += 1

    pii_rate = samples_with_pii / total if total > 0 else 0.0
    score = max(0.0, 1.0 - pii_rate)

    return ScoreResult(
        name="pii",
        score=score,
        details={
            "pii_rate": round(pii_rate, 4),
            "pii_by_type": pii_counts,
            "method": "regex_fallback",
            "total_samples": total,
        },
        warnings=["presidio-analyzer not available, using regex fallback"],
    )
