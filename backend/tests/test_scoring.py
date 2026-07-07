"""Tests for scoring pipeline."""

import asyncio
import pytest
from socbench.scoring.dedup import dedup_scorer
from socbench.scoring.format import format_scorer
from socbench.scoring.tokens import token_scorer
from socbench.scoring.pii import pii_scorer
from socbench.scoring.quality import quality_scorer
from socbench.scoring.code import code_scorer
from socbench.scoring.base import ScoreResult


SAMPLE_TEXTS = [
    {"text": "The quick brown fox jumps over the lazy dog. This is a sample sentence for testing."},
    {"text": "Machine learning models require large datasets for training. Data quality matters."},
    {"text": "def hello_world(): print('Hello, world!') This is a Python function."},
    {"text": "The quick brown fox jumps over the lazy dog. This is a sample sentence for testing."},
    {"text": "Data quality is crucial for training reliable machine learning models."},
]

SAMPLE_CODE = [
    {"text": "def add(a, b):\n    return a + b"},
    {"text": "class Calculator:\n    def __init__(self):\n        self.result = 0\n    def add(self, x):\n        self.result += x\n        return self"},
    {"text": "import os\nimport sys\nfor f in os.listdir('.'):\n    print(f)"},
]


class TestDedupScorer:
    @pytest.mark.asyncio
    async def test_basic_dedup(self):
        result = await dedup_scorer(SAMPLE_TEXTS)
        assert isinstance(result, ScoreResult)
        assert result.name == "dedup"
        assert 0.0 <= result.score <= 1.0
        assert "exact_dedup_rate" in result.details

    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await dedup_scorer([])
        assert result.score == 0.0
        assert any("No text" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_no_duplicates(self):
        texts = [{"text": f"Unique sentence number {i} with different content."} for i in range(20)]
        result = await dedup_scorer(texts)
        assert result.details["exact_dedup_rate"] == 0.0


class TestFormatScorer:
    @pytest.mark.asyncio
    async def test_consistent_format(self):
        result = await format_scorer(SAMPLE_TEXTS)
        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_code_parseability(self):
        result = await format_scorer(SAMPLE_CODE, text_key="text")
        assert result.details.get("code_parseability") is not None
        assert result.details["code_parseability"] > 0.8


class TestTokenScorer:
    @pytest.mark.asyncio
    async def test_token_analysis(self):
        result = await token_scorer(SAMPLE_TEXTS)
        assert isinstance(result, ScoreResult)
        assert "mean_tokens" in result.details
        assert "median_tokens" in result.details
        assert "outlier_rate" in result.details


class TestPIIScorer:
    @pytest.mark.asyncio
    async def test_clean_text(self):
        result = await pii_scorer(SAMPLE_TEXTS)
        assert isinstance(result, ScoreResult)
        assert result.score > 0.9  # No PII in sample texts

    @pytest.mark.asyncio
    async def test_pii_detection(self):
        pii_texts = [
            {"text": "Contact me at john@example.com for more info."},
            {"text": "My phone number is 555-123-4567."},
            {"text": "Normal text without any PII."},
        ]
        result = await pii_scorer(pii_texts)
        assert result.details["pii_rate"] > 0


class TestQualityScorer:
    @pytest.mark.asyncio
    async def test_quality_check(self):
        result = await quality_scorer(SAMPLE_TEXTS)
        assert isinstance(result, ScoreResult)
        assert "gopher_pass_rate" in result.details
        assert "fineweb_pass_rate" in result.details
        assert "diversity" in result.details


class TestCodeScorer:
    @pytest.mark.asyncio
    async def test_code_quality(self):
        result = await code_scorer(SAMPLE_CODE)
        assert isinstance(result, ScoreResult)
        assert "parse_rate" in result.details
        assert result.details["parse_rate"] > 0.9
