"""Multi-dimension scoring — quality, diversity, utility, coverage.

Not one composite score. Separate dimensions for separate questions.
Like IMDb: you don't get one score, you get plot, acting, direction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DimensionScore:
    name: str
    score: float  # 0.0-1.0
    details: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SocbenchScore:
    """The full Socbench assessment for a dataset."""

    dataset_id: str
    category: str
    category_label: str

    # Core dimensions
    quality: DimensionScore
    diversity: DimensionScore
    utility: DimensionScore  # How useful/actionable this dataset is

    # Supporting dimensions
    documentation: DimensionScore
    popularity: DimensionScore
    freshness: DimensionScore
    coverage: dict = field(default_factory=dict)  # Language, domain distributions

    # Safety & ethics
    contamination_rate: float = 0.0
    pii_safety: DimensionScore = field(default_factory=lambda: DimensionScore(name="pii", score=1.0))

    # Category-specific dimensions
    category_scores: list[DimensionScore] = field(default_factory=list)


def compute_coverage(samples: list[str]) -> dict:
    """Compute language and domain coverage metrics."""
    from socbench.scoring.language import language_scorer

    # Approximate language coverage by counting non-ASCII
    total = len(samples)
    if total == 0:
        return {}

    # Language distribution approximation
    ascii_count = 0
    char_counts: Counter = Counter()
    for text in samples:
        chars = set(text)
        char_counts.update(chars)
        if sum(1 for c in text if ord(c) < 128) / max(len(text), 1) > 0.9:
            ascii_count += 1

    # Unique character count as rough diversity proxy
    unique_chars = len(char_counts)

    # Domain detection heuristic
    domains: Counter = Counter()
    for text in samples:
        lower = text.lower()
        if "def " in lower or "class " in lower or "import " in lower:
            domains["code"] += 1
        elif "proof" in lower or "theorem" in lower or "equation" in lower:
            domains["math"] += 1
        elif "image" in lower or "vision" in lower or "pixel" in lower:
            domains["vision"] += 1
        elif "patient" in lower or "diagnosis" in lower or "treatment" in lower:
            domains["medical"] += 1
        elif "translation" in lower or "translate" in lower:
            domains["translation"] += 1
        else:
            domains["general"] += 1

    # Compute domain distribution
    domain_dist = {
        k: round(v / total, 4) for k, v in domains.most_common(10)
    }

    # Language diversity: higher unique chars = more languages
    language_diversity = min(unique_chars / 200, 1.0)  # Normalize

    return {
        "ascii_ratio": round(ascii_count / total, 4),
        "language_diversity": round(language_diversity, 4),
        "unique_characters": unique_chars,
        "domain_distribution": domain_dist,
        "total_samples": total,
    }


async def compute_multi_dimension_score(
    dataset_id: str,
    samples: list[dict],
    category_key: str = "pretraining-web",
    text_key: str = "text",
) -> SocbenchScore:
    """Compute the full multi-dimension Socbench score for a dataset."""
    from socbench.scoring import run_all_scorers, ScoreResult
    from socbench.scoring.dedup import dedup_scorer
    from socbench.categories import CATEGORIES, get_category_metrics

    cat = CATEGORIES.get(category_key, CATEGORIES["pretraining-web"])
    cat_metrics = get_category_metrics(category_key)

    # Run all automated scorers
    _, scorer_results = await run_all_scorers(samples, text_key=text_key)

    # Map scorers to dimensions
    scorer_map: dict[str, ScoreResult] = {r.name: r for r in scorer_results}

    # Quality dimension
    quality_avg = sum(
        scorer_map.get(m, ScoreResult(name=m, score=0.0)).score
        for m in ["quality", "dedup", "format"]
    ) / 3
    quality = DimensionScore(
        name="quality",
        score=quality_avg,
        details={
            "gopher_pass": scorer_map.get("quality", ScoreResult(name="q", score=0)).details.get("gopher_pass_rate", 0),
            "dedup": scorer_map.get("dedup", ScoreResult(name="d", score=0)).score,
            "format": scorer_map.get("format", ScoreResult(name="f", score=0)).score,
        },
    )

    # Diversity dimension
    diversity_score = scorer_map.get("quality", ScoreResult(name="q", score=0)).details.get("diversity", 0)
    token_details = scorer_map.get("tokens", ScoreResult(name="t", score=0)).details
    token_spread = token_details.get("diversity_score", 0)

    diversity = DimensionScore(
        name="diversity",
        score=(diversity_score + token_spread) / 2,
        details={
            "text_diversity": diversity_score,
            "token_spread": token_spread,
            "unique_words_ratio": diversity_score,
        },
    )

    # Utility: how actionable this dataset is (documentation, format, popularity)
    format_score = scorer_map.get("format", ScoreResult(name="f", score=0)).details.get("schema_conformance", 0)
    utility = DimensionScore(
        name="utility",
        score=format_score * 0.6 + 0.4,  # Baseline + format quality
        details={"schema_conformance": format_score},
    )

    # Documentation score (placeholder — will be enriched from HF metadata)
    documentation = DimensionScore(
        name="documentation",
        score=0.7,  # Default, updated from HF dataset cards
        details={"has_card": True, "has_license": True},
    )

    # Popularity (from HF metadata)
    popularity_details = {}
    popularity_score = 0.5
    if "downloads" in scorer_map:
        pop = scorer_map["downloads"]
        popularity_score = pop.score
        popularity_details = pop.details

    popularity = DimensionScore(
        name="popularity",
        score=popularity_score,
        details=popularity_details,
    )

    # Freshness: how recently updated
    freshness = DimensionScore(
        name="freshness",
        score=0.8,  # Placeholder — set from HF metadata
        details={},
    )

    # PII safety
    pii = scorer_map.get("pii", ScoreResult(name="pii", score=1.0))
    pii_dim = DimensionScore(
        name="pii_safety",
        score=pii.score,
        details=pii.details,
        warnings=pii.warnings,
    )

    # Coverage
    texts = [s.get(text_key, "") for s in samples if s.get(text_key)]
    coverage = compute_coverage(texts)

    # Contamination rate (placeholder — computed separately)
    contamination_rate = 0.0

    # Category-specific scores
    category_scores = [
        DimensionScore(
            name=m,
            score=scorer_map.get(m, ScoreResult(name=m, score=0.0)).score,
            details=scorer_map.get(m, ScoreResult(name=m, score=0.0)).details,
        )
        for m in cat_metrics
        if m in scorer_map
    ]

    return SocbenchScore(
        dataset_id=dataset_id,
        category=category_key,
        category_label=cat.label,
        quality=quality,
        diversity=diversity,
        utility=utility,
        documentation=documentation,
        popularity=popularity,
        freshness=freshness,
        coverage=coverage,
        contamination_rate=contamination_rate,
        pii_safety=pii_dim,
        category_scores=category_scores,
    )


def format_score_table(score: SocbenchScore) -> str:
    """Format a SocbenchScore as a readable table."""
    lines = [
        f"# {score.dataset_id}",
        f"Category: {score.category_label} ({score.category})",
        "",
        "## Core Dimensions",
        f"- **Quality**:    {score.quality.score:.3f}",
        f"- **Diversity**:  {score.diversity.score:.3f}",
        f"- **Utility**:    {score.utility.score:.3f}",
        "",
        "## Supporting",
        f"- **Documentation**: {score.documentation.score:.3f}",
        f"- **Popularity**:    {score.popularity.score:.3f}",
        f"- **Freshness**:     {score.freshness.score:.3f}",
        "",
        "## Safety",
        f"- **PII Safety**:     {score.pii_safety.score:.3f}",
        f"- **Contamination**:  {score.contamination_rate:.3f}",
        "",
        "## Category Metrics",
    ]
    for cs in score.category_scores:
        lines.append(f"- **{cs.name}**: {cs.score:.3f}")
        if cs.warnings:
            lines.append(f"  ⚠ {', '.join(cs.warnings)}")

    lines.append("\n## Coverage")
    for k, v in score.coverage.items():
        if isinstance(v, dict):
            lines.append(f"  {k}:")
            for dk, dv in v.items():
                lines.append(f"    {dk}: {dv}")
        else:
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)