"""Recommendations engine — "Best for:" ratings per category.

Instead of just a score, tell users what this dataset is useful for.
Actionable guidance, not abstract numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from socbench.categories import CATEGORIES, Category


@dataclass(frozen=True)
class Recommendation:
    category_key: str
    label: str
    rating: int  # 1-5 stars
    confidence: float  # 0.0-1.0
    reasoning: str = ""
    icon: str = ""


@dataclass(frozen=True)
class DatasetRecommendations:
    dataset_id: str
    best_for: list[Recommendation] = field(default_factory=list)
    good_for: list[Recommendation] = field(default_factory=list)
    not_for: list[Recommendation] = field(default_factory=list)


# Category-specific scoring heuristics
def _score_for_category(
    category_key: str,
    auto_scores: dict[str, float],
    coverage: dict[str, float],
    tags: list[str],
) -> tuple[int, float, str]:
    """Generate a recommendation rating and reasoning for a category."""

    cat = CATEGORIES.get(category_key)
    if not cat:
        return 3, 0.5, "Unknown category"

    avg_score = 0.0
    metrics = cat.metrics
    scores_available = 0

    for metric in metrics:
        if metric in auto_scores:
            avg_score += auto_scores[metric]
            scores_available += 1

    if scores_available > 0:
        avg_score /= scores_available

    # Convert to star rating
    if avg_score >= 0.85:
        stars = 5
    elif avg_score >= 0.70:
        stars = 4
    elif avg_score >= 0.50:
        stars = 3
    elif avg_score >= 0.30:
        stars = 2
    else:
        stars = 1

    confidence = min(scores_available / max(len(metrics), 1), 1.0)

    # Generate reasoning
    reasons = []
    if auto_scores.get("quality", 0) >= 0.8:
        reasons.append("high overall quality")
    if auto_scores.get("diversity", 0) >= 0.7:
        reasons.append("good diversity")
    if auto_scores.get("dedup_rate", 0) <= 0.05:
        reasons.append("well-deduplicated")
    if auto_scores.get("pii_rate", 0) <= 0.01:
        reasons.append("minimal PII")
    if auto_scores.get("parse_rate", 0) >= 0.9:
        reasons.append("high code parseability")
    reasoning = "; ".join(reasons) if reasons else "limited data"

    return stars, confidence, reasoning


def generate_recommendations(
    dataset_id: str,
    auto_scores: dict[str, float],
    coverage: dict[str, float],
    tags: list[str],
) -> DatasetRecommendations:
    """Generate "Best for:" / "Good for:" / "Not recommended for:" ratings.

    auto_scores should contain per-metric scores (quality, diversity, dedup_rate, etc.)
    """

    all_ratings: list[Recommendation] = []

    # Score every relevant category
    for cat_key, cat in CATEGORIES.items():
        if cat.parent:
            # Score leaf categories
            stars, confidence, reasoning = _score_for_category(
                cat_key, auto_scores, coverage, tags
            )
            all_ratings.append(
                Recommendation(
                    category_key=cat_key,
                    label=cat.label,
                    rating=stars,
                    confidence=confidence,
                    reasoning=reasoning,
                    icon=cat.icon,
                )
            )

    # Sort by rating, then confidence
    all_ratings.sort(key=lambda r: (r.rating, r.confidence), reverse=True)

    return DatasetRecommendations(
        dataset_id=dataset_id,
        best_for=[r for r in all_ratings if r.rating >= 4][:5],
        good_for=[r for r in all_ratings if r.rating == 3][:5],
        not_for=[r for r in all_ratings if r.rating <= 2][:5],
    )


def format_recommendations_markdown(recs: DatasetRecommendations) -> str:
    """Format recommendations as a Markdown table."""
    lines = [f"## Recommendations for {recs.dataset_id}\n"]

    if recs.best_for:
        lines.append("**Best for:**")
        for r in recs.best_for:
            stars_str = "★" * r.rating + "☆" * (5 - r.rating)
            lines.append(f"- {r.label}: {stars_str} ({r.reasoning})")
        lines.append("")

    if recs.good_for:
        lines.append("**Good for:**")
        for r in recs.good_for:
            stars_str = "★" * r.rating + "☆" * (5 - r.rating)
            lines.append(f"- {r.label}: {stars_str} ({r.reasoning})")
        lines.append("")

    if recs.not_for:
        lines.append("**Not recommended for:**")
        for r in recs.not_for:
            stars_str = "★" * r.rating + "☆" * (5 - r.rating)
            lines.append(f"- {r.label}: {stars_str} ({r.reasoning})")

    return "\n".join(lines)