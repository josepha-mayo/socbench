import asyncio

import pytest

from socbench.api import routes


def _reset_pending_cache(trending=None, most_used=None):
    routes._pending_scan_cache.update(
        {
            "expires_at": 0.0,
            "trending": trending or [],
            "most_used": most_used or [],
        }
    )


@pytest.mark.asyncio
async def test_training_pending_candidates_uses_fresh_cache(monkeypatch):
    _reset_pending_cache()
    calls = []

    async def fake_scan_datasets(*, sort, limit, days):
        calls.append((sort, limit, days))
        return [sort]

    monkeypatch.setattr(
        "socbench.discovery.scanner.scan_datasets",
        fake_scan_datasets,
    )

    trending, most_used = await routes._get_training_pending_candidates()
    cached_trending, cached_most_used = await routes._get_training_pending_candidates()

    assert trending == ["trendingScore"]
    assert most_used == ["downloads"]
    assert cached_trending == trending
    assert cached_most_used == most_used
    assert calls == [("trendingScore", 10, None), ("downloads", 10, None)]


@pytest.mark.asyncio
async def test_training_pending_candidates_falls_back_to_stale_cache(monkeypatch):
    _reset_pending_cache(trending=["old-trending"], most_used=["old-downloads"])
    monkeypatch.setattr(routes, "_PENDING_SCAN_TIMEOUT_SECONDS", 0.01)

    async def slow_scan_datasets(*, sort, limit, days):
        await asyncio.sleep(1)
        return [sort]

    monkeypatch.setattr(
        "socbench.discovery.scanner.scan_datasets",
        slow_scan_datasets,
    )

    trending, most_used = await routes._get_training_pending_candidates()

    assert trending == ["old-trending"]
    assert most_used == ["old-downloads"]
