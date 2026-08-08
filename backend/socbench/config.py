"""Project-wide configuration defaults.

Environment variables can override most values; see each constant's docstring.
"""

from __future__ import annotations

import os

# HuggingFace APIs
HF_VIEWER_API_URL = os.environ.get("HF_VIEWER_API_URL", "https://datasets-server.huggingface.co")
HF_DATASETS_API_URL = os.environ.get("HF_DATASETS_API_URL", "https://huggingface.co/api/datasets")

# HTTP timeouts (seconds)
HF_SPLITS_TIMEOUT = int(os.environ.get("HF_SPLITS_TIMEOUT", "60"))
HF_ROWS_TIMEOUT = int(os.environ.get("HF_ROWS_TIMEOUT", "60"))
HF_METADATA_TIMEOUT = int(os.environ.get("HF_METADATA_TIMEOUT", "15"))

# HF API concurrency limit
HF_HTTP_CONCURRENCY = int(os.environ.get("HF_HTTP_CONCURRENCY", "10"))

# Data preparation
DATA_PREP_MAX_SAMPLES = int(os.environ.get("DATA_PREP_MAX_SAMPLES", "100000"))
DATA_PREP_FALLBACK_EMPTY_LIMIT = int(
    os.environ.get("DATA_PREP_FALLBACK_EMPTY_LIMIT", "1000")
)
