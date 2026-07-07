"""Tests for discovery pipeline."""

import pytest
from socbench.discovery.scanner import (
    DiscoveredDataset,
    qualify_dataset,
    _extract_languages,
    _extract_license,
)


class TestHelpers:
    def test_extract_languages(self):
        tags = ["task_categories:text-generation", "language:en", "language:fr", "license:mit"]
        langs = _extract_languages(tags)
        assert "en" in langs
        assert "fr" in langs

    def test_extract_license(self):
        tags = ["language:en", "license:apache-2.0"]
        lic = _extract_license(tags)
        assert lic == "apache-2.0"

    def test_extract_license_missing(self):
        tags = ["language:en"]
        lic = _extract_license(tags)
        assert lic is None


class TestQualifyDataset:
    @pytest.mark.asyncio
    async def test_low_downloads(self):
        result = await qualify_dataset("test/ds", downloads=100, likes=50)
        assert not result.qualified
        assert "downloads" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_low_likes(self):
        result = await qualify_dataset("test/ds", downloads=10000, likes=3)
        assert not result.qualified
        assert "likes" in result.reason.lower()
