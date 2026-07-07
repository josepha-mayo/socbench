"""Dataset qualifier — run after scan to filter candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from socbench.discovery.scanner import (
    DiscoveredDataset,
    QualificationResult,
    qualify_dataset,
)


@dataclass
class QualifiedDataset:
    dataset: DiscoveredDataset
    qualification: QualificationResult
    stage1_score: Optional[float] = None
    stage2_contaminated: bool = False


async def qualify_all(
    datasets: list[DiscoveredDataset],
    token: Optional[str] = None,
) -> list[QualifiedDataset]:
    """Qualify a list of discovered datasets against thresholds."""
    results: list[QualifiedDataset] = []
    for ds in datasets:
        # Skip private/gated
        if ds.private or ds.gated:
            continue

        qual = await qualify_dataset(
            ds.hf_id,
            downloads=ds.downloads,
            likes=ds.likes,
            token=token,
        )
        if qual.qualified:
            results.append(QualifiedDataset(dataset=ds, qualification=qual))

    return results
