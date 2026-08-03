"""Per-language water-filling rebalancing for audit outputs."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RebalanceResult:
    """Result of water-filling rebalancing."""

    records: list[dict[str, Any]]
    language_counts: dict[str, int] = field(default_factory=dict)
    language_percentages: dict[str, float] = field(default_factory=dict)
    shortfalls: dict[str, int] = field(default_factory=dict)


def water_filling_rebalance(
    records: list[dict[str, Any]],
    target_ratios: dict[str, float] | None = None,
    output_size: int | None = None,
    seed: int = 42,
) -> RebalanceResult:
    """Rebalance records by language using a simple water-filling approach.

    If ``target_ratios`` is None, all accepted records are kept and their
    language distribution is reported.  If ``output_size`` is set, the
    records are shuffled and capped at that size.
    """
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)

    if output_size is not None:
        records = records[:output_size]

    counts = Counter(r.get("__audit_language", "unknown") for r in records)
    total = sum(counts.values()) or 1
    percentages = {lang: round(count / total, 4) for lang, count in counts.items()}

    shortfalls: dict[str, int] = {}
    if target_ratios:
        for lang, target in target_ratios.items():
            actual = counts.get(lang, 0)
            target_count = int(output_size * target) if output_size else 0
            if actual < target_count:
                shortfalls[lang] = target_count - actual

    return RebalanceResult(
        records=records,
        language_counts=dict(counts),
        language_percentages=percentages,
        shortfalls=shortfalls,
    )
