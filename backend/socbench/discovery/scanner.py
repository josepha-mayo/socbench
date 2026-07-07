"""HuggingFace dataset discovery scanner."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

HF_API = "https://huggingface.co/api/datasets"
VIEWER_API = "https://datasets-server.huggingface.co"

# Minimum thresholds for "serious" datasets (research-backed)
MIN_DOWNLOADS = 1_000
MIN_LIKES = 10
MIN_ROWS = 1_000
MIN_BYTES = 1_000_000  # 1 MB


@dataclass
class DiscoveredDataset:
    hf_id: str
    name: str
    description: Optional[str] = None
    license: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    trending_score: float = 0.0
    row_count: Optional[int] = None
    byte_size: Optional[int] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    gated: bool = False
    private: bool = False


@dataclass
class QualificationResult:
    qualified: bool
    reason: str = ""
    row_count: Optional[int] = None
    byte_size: Optional[int] = None


def _extract_languages(tags: list[str]) -> list[str]:
    return [t.replace("language:", "") for t in tags if t.startswith("language:")]


def _extract_license(tags: list[str]) -> Optional[str]:
    for t in tags:
        if t.startswith("license:"):
            return t.replace("license:", "")
    return None


async def scan_datasets(
    search: str = "",
    task_categories: Optional[list[str]] = None,
    languages: Optional[list[str]] = None,
    sort: str = "trendingScore",
    limit: int = 100,
    days: Optional[int] = None,
    token: Optional[str] = None,
) -> list[DiscoveredDataset]:
    """Scan HuggingFace for datasets matching criteria."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    all_datasets: list[DiscoveredDataset] = []
    url = HF_API
    params: dict = {
        "sort": sort,
        "direction": "-1",
        "limit": min(limit, 500),
        "full": "true",
    }
    if search:
        params["search"] = search
    if task_categories:
        for tc in task_categories:
            params.setdefault("filter", [])
            if isinstance(params["filter"], list):
                params["filter"].append(f"task_categories:{tc}")
            else:
                params["filter"] = [params["filter"], f"task_categories:{tc}"]
    if languages:
        for lang in languages:
            params.setdefault("filter", [])
            if isinstance(params["filter"], list):
                params["filter"].append(f"languages:{lang}")
            else:
                params["filter"] = [params["filter"], f"languages:{lang}"]

    cutoff = None
    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with httpx.AsyncClient(timeout=30) as client:
        fetched = 0
        while url and fetched < limit:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            for ds in data:
                created = ds.get("createdAt") or ds.get("lastModified")
                if cutoff and created:
                    if isinstance(created, str):
                        try:
                            created_dt = datetime.fromisoformat(
                                created.replace("Z", "+00:00")
                            )
                            if created_dt < cutoff:
                                continue
                        except ValueError:
                            pass

                tags = ds.get("tags", []) or []
                all_datasets.append(
                    DiscoveredDataset(
                        hf_id=ds["id"],
                        name=ds.get("id", "").split("/")[-1],
                        description=ds.get("description"),
                        license=_extract_license(tags),
                        tags=tags,
                        languages=_extract_languages(tags),
                        downloads=ds.get("downloads", 0) or 0,
                        likes=ds.get("likes", 0) or 0,
                        trending_score=ds.get("trendingScore", 0) or 0,
                        created_at=ds.get("createdAt"),
                        last_modified=ds.get("lastModified"),
                        gated=bool(ds.get("gated")),
                        private=bool(ds.get("private")),
                    )
                )
                fetched += 1
                if fetched >= limit:
                    break

            # Pagination via Link header
            link = resp.headers.get("Link", "")
            url = None
            params = {}
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")

            time.sleep(0.5)

    return all_datasets


async def qualify_dataset(
    dataset_id: str,
    min_rows: int = MIN_ROWS,
    min_bytes: int = MIN_BYTES,
    min_downloads: int = MIN_DOWNLOADS,
    min_likes: int = MIN_LIKES,
    downloads: int = 0,
    likes: int = 0,
    token: Optional[str] = None,
) -> QualificationResult:
    """Check if a dataset meets quality thresholds."""
    # Quick checks from metadata
    if downloads < min_downloads:
        return QualificationResult(
            qualified=False, reason=f"Too few downloads: {downloads}"
        )
    if likes < min_likes:
        return QualificationResult(
            qualified=False, reason=f"Too few likes: {likes}"
        )

    # Fetch size from dataset viewer API
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(
                f"{VIEWER_API}/size",
                params={"dataset": dataset_id},
                headers=headers,
            )
            if resp.status_code != 200:
                return QualificationResult(
                    qualified=False, reason="Could not fetch dataset size"
                )
            size_data = resp.json()
            ds_size = size_data.get("size", {}).get("dataset", {})
            num_rows = ds_size.get("num_rows", 0)
            num_bytes = ds_size.get("num_bytes_parquet_files", 0)

            if num_rows < min_rows:
                return QualificationResult(
                    qualified=False,
                    reason=f"Too few rows: {num_rows}",
                    row_count=num_rows,
                    byte_size=num_bytes,
                )
            if num_bytes < min_bytes:
                return QualificationResult(
                    qualified=False,
                    reason=f"Too small: {num_bytes} bytes",
                    row_count=num_rows,
                    byte_size=num_bytes,
                )

            return QualificationResult(
                qualified=True,
                row_count=num_rows,
                byte_size=num_bytes,
            )
        except httpx.TimeoutException:
            return QualificationResult(qualified=False, reason="Size API timeout")
