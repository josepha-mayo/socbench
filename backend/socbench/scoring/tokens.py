"""Token length scorer — distribution analysis, outlier detection.

Uses tiktoken cl100k_base for tokenization.
"""

from __future__ import annotations

import math
from collections import Counter

from socbench.scoring.base import ScoreResult


async def token_scorer(samples: list[dict], text_key: str = "text") -> ScoreResult:
    """Analyze token length distribution and detect outliers."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return ScoreResult(
            name="tokens",
            score=0.0,
            warnings=["tiktoken not installed"],
        )

    texts = [s.get(text_key, "") for s in samples if s.get(text_key)]
    if not texts:
        return ScoreResult(name="tokens", score=0.0, warnings=["No text content"])

    # Tokenize all texts
    token_counts = []
    for text in texts:
        tokens = enc.encode(text)
        token_counts.append(len(tokens))

    if not token_counts:
        return ScoreResult(name="tokens", score=0.0, warnings=["Empty dataset"])

    total = len(token_counts)
    mean_tokens = sum(token_counts) / total
    sorted_counts = sorted(token_counts)
    median_tokens = sorted_counts[total // 2]
    min_tokens = sorted_counts[0]
    max_tokens = sorted_counts[-1]
    std_tokens = (
        sum((t - mean_tokens) ** 2 for t in token_counts) / total
    ) ** 0.5

    # Percentiles
    def percentile(data: list[int], p: float) -> int:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    p5 = percentile(sorted_counts, 5)
    p25 = percentile(sorted_counts, 25)
    p75 = percentile(sorted_counts, 75)
    p95 = percentile(sorted_counts, 95)
    p99 = percentile(sorted_counts, 99)

    # Outlier rate: <10 tokens or >100K tokens
    outlier_count = sum(1 for t in token_counts if t < 10 or t > 100_000)
    outlier_rate = outlier_count / total

    # Usable range: percentage within 50-8000 tokens (reasonable for training)
    usable_count = sum(1 for t in token_counts if 50 <= t <= 8000)
    usable_range_pct = usable_count / total

    # Score: based on usable range and outlier rate
    score = usable_range_pct * (1.0 - outlier_rate)

    # Distribution health: how spread out are the token lengths
    cv = std_tokens / mean_tokens if mean_tokens > 0 else 0.0
    # Good diversity: CV between 0.3 and 1.5
    diversity_score = 1.0 - abs(cv - 0.8) / 0.8
    diversity_score = max(0.0, min(1.0, diversity_score))

    warnings = []
    if outlier_rate > 0.05:
        warnings.append(f"High outlier rate: {outlier_rate:.1%}")
    if mean_tokens < 50:
        warnings.append(f"Very short average: {mean_tokens:.0f} tokens")
    if mean_tokens > 5000:
        warnings.append(f"Very long average: {mean_tokens:.0f} tokens")

    return ScoreResult(
        name="tokens",
        score=score,
        details={
            "mean_tokens": round(mean_tokens, 1),
            "median_tokens": median_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "std_tokens": round(std_tokens, 1),
            "p5": p5,
            "p25": p25,
            "p75": p75,
            "p95": p95,
            "p99": p99,
            "outlier_rate": round(outlier_rate, 4),
            "usable_range_pct": round(usable_range_pct, 4),
            "diversity_score": round(diversity_score, 4),
            "total_samples": total,
        },
        warnings=warnings,
    )
