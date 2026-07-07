"""Socbench — 'The unexamined dataset is not worth training on.'

Scientific dataset intelligence. Multi-dimension scoring, provenance tracking,
contamination checking, and training impact measurement.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "assess_dataset",
    "discover_datasets",
    "check_contamination",
    "train_and_evaluate",
    "get_provenance",
    "generate_recommendations",
    "classify_dataset",
]


def assess_dataset(dataset_id: str, sample_size: int = 10_000):
    """Run full multi-dimension Socbench assessment on a dataset."""
    from socbench.runner import run_socbench_scoring

    return run_socbench_scoring(dataset_id, sample_size=sample_size)


def discover_datasets(search: str = "", limit: int = 50, days: int = None):
    """Discover new/trending datasets from HuggingFace."""
    from socbench.discovery.scanner import scan_datasets

    return scan_datasets(search=search, limit=limit, days=days)


def check_contamination(dataset_id: str):
    """Run contamination check against eval benchmarks."""
    from socbench.contamination.checker import check_dataset_contamination

    return check_dataset_contamination(dataset_id)


def train_and_evaluate(dataset_id: str, tokens: int = 1_000_000_000):
    """Run Stage 3: train GPT-2 124M and measure loss curves."""
    from socbench.training.trainer import train_on_dataset

    return train_on_dataset(dataset_id, tokens=tokens)


def get_provenance(dataset_id: str):
    """Get known provenance (dataset→model→paper mapping)."""
    from socbench.provenance import get_provenance as _gp

    return _gp(dataset_id)


def generate_recommendations(dataset_id: str, scores: dict, coverage: dict, tags: list[str]):
    """Generate 'Best for:' recommendations."""
    from socbench.recommendations import generate_recommendations as _gr

    return _gr(dataset_id, scores, coverage, tags)


def classify_dataset(tags: list[str], description: str = ""):
    """Classify a dataset into its hierarchical category."""
    from socbench.categories import classify_dataset as _cd

    return _cd(tags, description)