"""Stage 2: Contamination checker — n-gram overlap with eval benchmarks.

Method: 13-gram exact match (GPT-3 paper methodology).
Benchmarks: HumanEval, MBPP, GSM8K, MMLU, ARC, HellaSwag, TruthfulQA.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import httpx

from socbench.scoring.base import ScoreResult

VIEWER_API = "https://datasets-server.huggingface.co"

BENCHMARKS = {
    "humaneval": "openai/openai_humaneval",
    "mbpp": "google-research-datasets/mbpp",
    "gsm8k": "openai/gsm8k",
    "mmlu": "cais/mmlu",
    "arc": "allenai/ai2_arc",
    "hellaswag": "Rowan/hellaswag",
    "truthfulqa": "truthfulqa/truthful_qa",
}

NGRAM_SIZE = 13


def _get_ngrams(text: str, n: int = NGRAM_SIZE) -> set[str]:
    words = text.lower().split()
    if len(words) < n:
        return {text.lower()}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


async def _fetch_eval_samples(
    dataset_id: str,
    text_key: str = "text",
    limit: int = 500,
    token: Optional[str] = None,
) -> list[str]:
    """Fetch sample texts from a HuggingFace dataset via the viewer API."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    texts: list[str] = []
    async with httpx.AsyncClient(timeout=30) as client:
        # Get splits
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

        for split_info in splits[:1]:  # Use first split only
            split_name = split_info.get("split", "train")
            config = split_info.get("config", "default")

            # Fetch rows
            offset = 0
            while offset < limit:
                try:
                    resp = await client.get(
                        f"{VIEWER_API}/rows",
                        params={
                            "dataset": dataset_id,
                            "config": config,
                            "split": split_name,
                            "offset": offset,
                            "length": min(limit - offset, 100),
                        },
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        break
                    rows = resp.json().get("rows", [])
                    if not rows:
                        break

                    for row in rows:
                        row_data = row.get("row", {})
                        # Try common text keys
                        for key in [text_key, "question", "prompt", "problem", "input", "text"]:
                            if key in row_data and isinstance(row_data[key], str):
                                texts.append(row_data[key])
                                break
                    offset += len(rows)
                except Exception:
                    break

    return texts[:limit]


async def check_dataset_contamination(
    dataset_id: str,
    text_key: str = "text",
    token: Optional[str] = None,
    benchmarks: Optional[dict[str, str]] = None,
) -> list[ScoreResult]:
    """Check contamination of a dataset against standard eval benchmarks.

    Returns one ScoreResult per benchmark.
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    # Fetch dataset samples
    dataset_texts = await _fetch_eval_samples(
        dataset_id, text_key=text_key, limit=2000, token=token
    )
    if not dataset_texts:
        return [
            ScoreResult(
                name=f"contamination_{name}",
                score=0.0,
                warnings=[f"Could not fetch samples from {dataset_id}"],
            )
            for name in benchmarks
        ]

    # Build n-gram index from dataset
    dataset_ngrams: set[str] = set()
    for text in dataset_texts:
        dataset_ngrams.update(_get_ngrams(text))

    results: list[ScoreResult] = []

    for bench_name, bench_id in benchmarks.items():
        # Fetch eval samples
        eval_texts = await _fetch_eval_samples(
            bench_id, text_key=text_key, limit=500, token=token
        )
        if not eval_texts:
            results.append(
                ScoreResult(
                    name=f"contamination_{bench_name}",
                    score=0.0,
                    warnings=[f"Could not fetch {bench_name}"],
                )
            )
            continue

        # Check overlap
        overlap_count = 0
        total_eval = len(eval_texts)

        for eval_text in eval_texts:
            eval_ngrams = _get_ngrams(eval_text)
            # Check if any n-gram from eval text is in the dataset
            if eval_ngrams & dataset_ngrams:
                overlap_count += 1

        overlap_rate = overlap_count / total_eval if total_eval > 0 else 0.0

        # Score: 1.0 = no contamination, 0.0 = fully contaminated
        score = 1.0 - overlap_rate

        warnings = []
        if overlap_rate > 0.1:
            warnings.append(f"HIGH contamination: {overlap_rate:.1%} overlap")
        elif overlap_rate > 0.05:
            warnings.append(f"Moderate contamination: {overlap_rate:.1%} overlap")

        results.append(
            ScoreResult(
                name=f"contamination_{bench_name}",
                score=score,
                details={
                    "benchmark": bench_name,
                    "benchmark_id": bench_id,
                    "overlap_rate": round(overlap_rate, 4),
                    "overlap_count": overlap_count,
                    "total_eval": total_eval,
                    "method": f"ngram_{NGRAM_SIZE}",
                },
                warnings=warnings,
            )
        )

    return results
