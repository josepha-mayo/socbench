"""Stage 1: Run all automated quality scorers."""

from __future__ import annotations

from socbench.scoring.base import ScoreResult
from socbench.scoring.dedup import dedup_scorer
from socbench.scoring.format import format_scorer
from socbench.scoring.tokens import token_scorer
from socbench.scoring.language import language_scorer
from socbench.scoring.pii import pii_scorer
from socbench.scoring.quality import quality_scorer
from socbench.scoring.code import code_scorer

__all__ = [
    "run_all_scorers",
    "ScoreResult",
    "dedup_scorer",
    "format_scorer",
    "token_scorer",
    "language_scorer",
    "pii_scorer",
    "quality_scorer",
    "code_scorer",
]

# Composite weights (from PLAN.md)
WEIGHTS = {
    "dedup": 0.15,
    "format": 0.15,
    "tokens": 0.15,
    "language": 0.15,
    "pii": 0.15,
    "quality": 0.15,
    "code": 0.10,
}


async def run_all_scorers(
    samples: list[dict],
    text_key: str = "text",
    expected_language: str = "en",
    is_code: bool = False,
    category: str | None = None,
) -> tuple[float, list[ScoreResult]]:
    """Run all scorers and return composite score + individual results."""
    scorers = [
        dedup_scorer,
        format_scorer,
        token_scorer,
        lambda s, k: language_scorer(s, k, expected_language),
        pii_scorer,
        lambda s, k: quality_scorer(s, k, category=category),
    ]
    if is_code:
        scorers.append(code_scorer)

    results: list[ScoreResult] = []
    for scorer in scorers:
        try:
            result = await scorer(samples, text_key)
            results.append(result)
        except Exception as e:
            results.append(
                ScoreResult(
                    name=scorer.__name__.replace("_scorer", ""),
                    score=0.0,
                    warnings=[f"Scorer failed: {e}"],
                )
            )

    # Composite score
    composite = 0.0
    total_weight = 0.0
    for r in results:
        weight = WEIGHTS.get(r.name, 0.1)
        composite += r.score * weight
        total_weight += weight

    if total_weight > 0:
        composite /= total_weight
        composite *= total_weight  # Re-normalize to actual weight sum

    return composite, results
