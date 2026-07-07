"""Runner — orchestrate all stages for a dataset.

Multi-dimension scoring, contamination checking, training prep.
Results feed the leaderboard and the dataset intelligence page.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx

from socbench.categories import classify_dataset, get_category_metrics
from socbench.score import compute_multi_dimension_score, SocbenchScore
from socbench.scoring.base import ScoreResult
from socbench.provenance import get_provenance

VIEWER_API = "https://datasets-server.huggingface.co"


async def fetch_samples(
    dataset_id: str,
    text_key: str = "text",
    sample_size: int = 10_000,
    token: Optional[str] = None,
) -> list[dict]:
    """Fetch sample rows from a HuggingFace dataset via the viewer API."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    samples: list[dict] = []
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{VIEWER_API}/splits",
                params={"dataset": dataset_id},
                headers=headers,
            )
            if resp.status_code != 200:
                return []
            splits = resp.json().get("splits", [])
        except Exception:
            return []

        for split_info in splits[:1]:
            split_name = split_info.get("split", "train")
            config = split_info.get("config", "default")
            num_rows = min(split_info.get("num_rows", 0), sample_size)

            offset = 0
            batch_size = 100
            while offset < num_rows:
                try:
                    resp = await client.get(
                        f"{VIEWER_API}/rows",
                        params={
                            "dataset": dataset_id,
                            "config": config,
                            "split": split_name,
                            "offset": offset,
                            "length": min(num_rows - offset, batch_size),
                        },
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        break
                    rows = resp.json().get("rows", [])
                    if not rows:
                        break
                    for row in rows:
                        samples.append(row.get("row", {}))
                    offset += len(rows)
                except Exception:
                    break

    return samples


async def fetch_metadata(
    dataset_id: str,
    token: Optional[str] = None,
) -> dict:
    """Fetch dataset metadata from HuggingFace API."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"https://huggingface.co/api/datasets/{dataset_id}",
            headers=headers,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "tags": data.get("tags", []) or [],
                "license": data.get("tags", []) or [],
                "downloads": data.get("downloads", 0) or 0,
                "likes": data.get("likes", 0) or 0,
                "last_modified": data.get("lastModified"),
                "created_at": data.get("createdAt"),
                "description": data.get("description", ""),
                "gated": bool(data.get("gated")),
                "private": bool(data.get("private")),
            }
        return {}


async def run_socbench_scoring(
    dataset_id: str,
    sample_size: int = 10_000,
    text_key: str = "text",
    token: Optional[str] = None,
) -> dict:
    """Run the full Socbench assessment for a dataset.

    Returns multi-dimension scores, provenance, category, recommendations.
    """
    start = time.time()

    # Fetch metadata
    metadata = await fetch_metadata(dataset_id, token=token)

    # Classify dataset
    tags = metadata.get("tags", [])
    category_key = classify_dataset(tags, metadata.get("description", ""))

    # Fetch samples
    samples = await fetch_samples(
        dataset_id, text_key=text_key, sample_size=sample_size, token=token
    )

    if not samples:
        return {
            "dataset_id": dataset_id,
            "error": "Could not fetch samples",
            "category": category_key,
        }

    # Run multi-dimension scoring
    score = await compute_multi_dimension_score(
        dataset_id, samples, category_key=category_key, text_key=text_key
    )

    # Get provenance
    provenance = get_provenance(dataset_id)

    # Get category metrics
    cat_metrics = get_category_metrics(category_key)

    elapsed = time.time() - start

    return {
        "dataset_id": dataset_id,
        "category": score.category,
        "category_label": score.category_label,
        "quality": {
            "score": score.quality.score,
            "details": score.quality.details,
        },
        "diversity": {
            "score": score.diversity.score,
            "details": score.diversity.details,
        },
        "utility": {
            "score": score.utility.score,
            "details": score.utility.details,
        },
        "documentation": {
            "score": score.documentation.score,
            "details": score.documentation.details,
        },
        "popularity": {
            "score": score.popularity.score,
            "details": {
                "downloads": metadata.get("downloads", 0),
                "likes": metadata.get("likes", 0),
            },
        },
        "freshness": {
            "score": score.freshness.score,
            "details": score.freshness.details,
        },
        "pii_safety": {
            "score": score.pii_safety.score,
            "details": score.pii_safety.details,
            "warnings": score.pii_safety.warnings,
        },
        "coverage": score.coverage,
        "contamination_rate": score.contamination_rate,
        "category_metrics": [
            {
                "name": cs.name,
                "score": cs.score,
                "details": cs.details,
                "warnings": cs.warnings,
            }
            for cs in score.category_scores
        ],
        "provenance": [
            {
                "model_name": p.model_name,
                "paper_title": p.paper_title,
                "paper_url": p.paper_url,
                "verified": p.verified,
            }
            for p in provenance
        ],
        "metadata": {
            "tags": tags,
            "license": metadata.get("license"),
            "downloads": metadata.get("downloads", 0),
            "likes": metadata.get("likes", 0),
            "last_modified": metadata.get("last_modified"),
        },
        "samples_fetched": len(samples),
        "elapsed_seconds": round(elapsed, 2),
    }