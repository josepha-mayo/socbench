"""Runner — orchestrate all stages for a dataset.

Multi-dimension scoring, contamination checking, training prep.
Results feed the leaderboard and the dataset intelligence page.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx

from socbench.categories import classify_dataset, get_category_metrics
from socbench.contamination.checker import check_texts_against_benchmarks
from socbench.score import compute_multi_dimension_score, SocbenchScore, _extract_text
from socbench.scoring.base import ScoreResult
from socbench.provenance import get_provenance

VIEWER_API = "https://datasets-server.huggingface.co"


async def fetch_samples(
    dataset_id: str,
    text_key: str = "text",
    sample_size: int = 10_000,
    token: Optional[str] = None,
) -> list[dict]:
    """Fetch sample rows from a HuggingFace dataset via the viewer API.

    Retries transient HTTP/network failures (the HF viewer API is flaky
    under load) so scoring doesn't fail on a single dropped request.
    """
    import asyncio

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    samples: list[dict] = []
    async with httpx.AsyncClient(timeout=30) as client:
        splits: list[dict] = []
        for attempt in range(3):
            try:
                resp = await client.get(
                    f"{VIEWER_API}/splits",
                    params={"dataset": dataset_id},
                    headers=headers,
                )
                if resp.status_code == 200:
                    splits = resp.json().get("splits", [])
                    break
            except Exception:
                pass
            await asyncio.sleep(2 * (attempt + 1))

        if not splits:
            return []

        for split_info in splits[:1]:
            split_name = split_info.get("split", "train")
            config = split_info.get("config", "default")
            num_rows = split_info.get("num_rows") or sample_size
            num_rows = min(num_rows, sample_size)

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


def _license_from_tags(tags: list[str], card_data: dict) -> Optional[str]:
    """Extract a license identifier from HF tags or card data."""
    for t in tags:
        if isinstance(t, str) and t.startswith("license:"):
            return t.split(":", 1)[1]
    lic = card_data.get("license")
    if isinstance(lic, list) and lic:
        return lic[0]
    if isinstance(lic, str):
        return lic
    return None


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
            params={"full": "true"},
            headers=headers,
        )
        if resp.status_code == 200:
            data = resp.json()
            tags = data.get("tags", []) or []
            card_data = data.get("cardData", {}) or {}
            description = data.get("description", "") or ""
            return {
                "tags": tags,
                "license": _license_from_tags(tags, card_data),
                "downloads": data.get("downloads", 0) or 0,
                "likes": data.get("likes", 0) or 0,
                "last_modified": data.get("lastModified"),
                "created_at": data.get("createdAt"),
                "description": description,
                "card_data": card_data,
                "has_card": bool(card_data),
                "gated": bool(data.get("gated")),
                "private": bool(data.get("private")),
            }
        return {}


def compute_documentation_score(metadata: dict) -> tuple[float, dict]:
    """Score dataset card completeness from real HF metadata."""
    card = metadata.get("card_data", {}) or {}
    description = metadata.get("description", "") or ""
    tags = metadata.get("tags", []) or []
    has_license = metadata.get("license") is not None

    checks = {
        "has_card": bool(card),
        "has_description": len(description.strip()) >= 50,
        "has_license": has_license,
        "has_tags": len(tags) >= 3,
        "has_task_categories": any(
            isinstance(t, str) and t.startswith("task_categories:") for t in tags
        ),
        "has_language": any(
            isinstance(t, str) and t.startswith("language:") for t in tags
        )
        or bool(card.get("language")),
    }
    weights = {
        "has_card": 0.25,
        "has_description": 0.25,
        "has_license": 0.20,
        "has_tags": 0.10,
        "has_task_categories": 0.10,
        "has_language": 0.10,
    }
    score = sum(weights[k] for k, v in checks.items() if v)
    return round(score, 4), checks


def compute_freshness_score(metadata: dict) -> tuple[float, dict]:
    """Score recency from real last-modified date (decay over ~2 years)."""
    from datetime import datetime, timezone

    last_modified = metadata.get("last_modified") or metadata.get("created_at")
    if not last_modified:
        return 0.5, {"reason": "no date available"}

    try:
        dt = datetime.fromisoformat(str(last_modified).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0.5, {"reason": "unparseable date"}

    now = datetime.now(timezone.utc)
    age_days = max((now - dt).days, 0)

    # Linear decay: <=30d -> 1.0, >=730d (2y) -> 0.2
    if age_days <= 30:
        score = 1.0
    elif age_days >= 730:
        score = 0.2
    else:
        score = 1.0 - 0.8 * ((age_days - 30) / 700)

    return round(score, 4), {
        "last_modified": last_modified,
        "age_days": age_days,
    }


def compute_popularity_score(metadata: dict) -> tuple[float, dict]:
    """Score community adoption from real downloads + likes (log-scaled)."""
    import math

    downloads = metadata.get("downloads", 0) or 0
    likes = metadata.get("likes", 0) or 0

    # log10(downloads): 1K->3, 100K->5, 10M->7. Normalize 3..7 -> 0..1
    dl_score = 0.0
    if downloads > 0:
        dl_score = max(0.0, min((math.log10(downloads) - 3) / 4, 1.0))
    # log10(likes): 10->1, 1000->3. Normalize 1..3 -> 0..1
    like_score = 0.0
    if likes > 0:
        like_score = max(0.0, min((math.log10(likes) - 1) / 2, 1.0))

    score = 0.6 * dl_score + 0.4 * like_score
    return round(score, 4), {
        "downloads": downloads,
        "likes": likes,
        "download_score": round(dl_score, 4),
        "like_score": round(like_score, 4),
    }


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
    category_key = classify_dataset(
        tags, metadata.get("description", ""), dataset_id=dataset_id
    )

    # Fetch samples
    samples = await fetch_samples(
        dataset_id, text_key=text_key, sample_size=sample_size, token=token
    )

    if not samples:
        return {
            "hf_id": dataset_id,
            "name": metadata.get("name", dataset_id),
            "description": metadata.get("description"),
            "license": metadata.get("license"),
            "tags": tags,
            "source_url": None,
            "category": category_key,
            "category_label": "",
            "quality": {"score": 0.0, "details": {}},
            "diversity": {"score": 0.0, "details": {}},
            "utility": {"score": 0.0, "details": {}},
            "documentation": {"score": 0.0, "details": {}},
            "popularity": {"score": 0.0, "details": {}},
            "freshness": {"score": 0.0, "details": {}},
            "pii_safety": {"score": 1.0, "details": {}},
            "coverage": {},
            "contamination_rate": 0.0,
            "provenance": [],
            "category_metrics": [],
            "recommendations": {"best_for": [], "good_for": [], "not_for": []},
            "metadata": {
                "tags": tags,
                "license": metadata.get("license"),
                "downloads": metadata.get("downloads", 0),
                "likes": metadata.get("likes", 0),
                "last_modified": metadata.get("last_modified"),
                "created_at": metadata.get("created_at"),
            },
            "training": None,
            "error": "Could not fetch samples",
        }

    # Run multi-dimension scoring
    score = await compute_multi_dimension_score(
        dataset_id, samples, category_key=category_key, text_key=text_key
    )

    # Extract repetition rate from dedup scorer details.
    dedup_details = score.quality.details
    repetition_pct = round(
        (1.0 - dedup_details.get("dedup", 1.0)) * 100, 2
    )

    # Run contamination checker (13-gram overlap vs eval benchmarks).
    # Reuses cached benchmark n-gram sets; first run fetches them.
    cont_texts = [_extract_text(s, text_key) for s in samples]
    cont_results, cont_rate = await check_texts_against_benchmarks(
        cont_texts, token=token
    )
    contamination_checks = [
        {
            "name": r.name,
            "score": r.score,
            "details": r.details,
            "warnings": r.warnings,
        }
        for r in cont_results
    ]

    # Get provenance
    provenance = get_provenance(dataset_id)

    # Get category metrics
    cat_metrics = get_category_metrics(category_key)

    # Compute supporting dimensions from REAL HF metadata (not placeholders)
    doc_score, doc_details = compute_documentation_score(metadata)
    fresh_score, fresh_details = compute_freshness_score(metadata)
    pop_score, pop_details = compute_popularity_score(metadata)

    # Generate "Best for / Good for / Not for" recommendations
    from socbench.recommendations import generate_recommendations

    auto_scores = {
        "quality": score.quality.score,
        "diversity": score.diversity.score,
        "utility": score.utility.score,
        "documentation": doc_score,
        "popularity": pop_score,
        "freshness": fresh_score,
        "pii_safety": score.pii_safety.score,
        "contamination": cont_rate,
    }
    for cs in score.category_scores:
        auto_scores[cs.name] = cs.score

    recs = generate_recommendations(dataset_id, auto_scores, score.coverage, tags)

    def _ser_recs(items):
        return [
            {
                "category_key": r.category_key,
                "label": r.label,
                "rating": r.rating,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
            }
            for r in items
        ]

    recommendations = {
        "best_for": _ser_recs(recs.best_for),
        "good_for": _ser_recs(recs.good_for),
        "not_for": _ser_recs(recs.not_for),
    }

    elapsed = time.time() - start

    return {
        "hf_id": dataset_id,
        "name": metadata.get("name", dataset_id),
        "description": metadata.get("description"),
        "license": metadata.get("license"),
        "tags": tags,
        "source_url": None,
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
            "score": doc_score,
            "details": doc_details,
        },
        "popularity": {
            "score": pop_score,
            "details": pop_details,
        },
        "freshness": {
            "score": fresh_score,
            "details": fresh_details,
        },
        "pii_safety": {
            "score": score.pii_safety.score,
            "details": score.pii_safety.details,
            "warnings": score.pii_safety.warnings,
        },
        "coverage": score.coverage,
        "contamination_rate": cont_rate,
        "contamination_checks": contamination_checks,
        "repetition_pct": repetition_pct,
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
        "recommendations": recommendations,
        "metadata": {
            "tags": tags,
            "license": metadata.get("license"),
            "downloads": metadata.get("downloads", 0),
            "likes": metadata.get("likes", 0),
            "last_modified": metadata.get("last_modified"),
            "created_at": metadata.get("created_at"),
        },
        "training": None,
        "samples_fetched": len(samples),
        "elapsed_seconds": round(elapsed, 2),
    }