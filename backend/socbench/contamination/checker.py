"""Stage 2: Contamination checker — n-gram overlap with eval benchmarks.

Method: 13-gram exact match (GPT-3 paper methodology).
Benchmarks: HumanEval, MBPP, GSM8K, MMLU, ARC, HellaSwag, TruthfulQA.

Benchmark reference n-gram sets are cached locally so repeated dataset
assessments don't refetch the eval corpora.
"""

from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
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
CACHE_DIR = Path(__file__).parent / ".cache"


def _get_ngrams(text: str, n: int = NGRAM_SIZE) -> set[str]:
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


async def _fetch_eval_samples(
    dataset_id: str,
    limit: int = 500,
    token: Optional[str] = None,
) -> list[str]:
    """Fetch sample texts from a HuggingFace dataset via the viewer API."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    texts: list[str] = []
    async with httpx.AsyncClient(timeout=30) as client:
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
        else:
            return []

        for split_info in splits[:1]:
            split_name = split_info.get("split", "train")
            config = split_info.get("config", "default")

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
                        # Try common text keys; concatenate multiple if present.
                        parts: list[str] = []
                        for key in ["question", "prompt", "problem", "input", "text", "content", "instruction"]:
                            val = row_data.get(key)
                            if isinstance(val, str) and val.strip():
                                parts.append(val)
                        if parts:
                            texts.append("\n".join(parts))
                        elif isinstance(row_data, str):
                            texts.append(row_data)
                    offset += len(rows)
                except Exception:
                    break

    return texts[:limit]


def _cache_path(bench_name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{bench_name}.pkl"


async def _load_benchmark_ngrams(
    bench_name: str,
    bench_id: str,
    token: Optional[str] = None,
    force_refresh: bool = False,
) -> Optional[set[str]]:
    """Load cached n-grams for a benchmark, fetching and caching if needed."""
    cache = _cache_path(bench_name)
    if not force_refresh and cache.exists():
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    texts = await _fetch_eval_samples(bench_id, limit=500, token=token)
    if not texts:
        return None

    ngrams: set[str] = set()
    for text in texts:
        ngrams.update(_get_ngrams(text))

    try:
        with open(cache, "wb") as f:
            pickle.dump(ngrams, f)
    except Exception:
        pass

    return ngrams


async def check_texts_against_benchmarks(
    dataset_texts: list[str],
    token: Optional[str] = None,
    benchmarks: Optional[dict[str, str]] = None,
) -> tuple[list[ScoreResult], float]:
    """Check a list of texts for contamination against eval benchmarks.

    Returns per-benchmark ScoreResults and an aggregate contamination rate
    (max overlap across benchmarks).
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    if not dataset_texts:
        return [], 0.0

    dataset_ngrams: set[str] = set()
    for text in dataset_texts:
        dataset_ngrams.update(_get_ngrams(text))

    results: list[ScoreResult] = []
    max_overlap = 0.0

    for bench_name, bench_id in benchmarks.items():
        bench_ngrams = await _load_benchmark_ngrams(bench_name, bench_id, token=token)
        if bench_ngrams is None:
            results.append(
                ScoreResult(
                    name=f"contamination_{bench_name}",
                    score=1.0,
                    details={"benchmark": bench_name, "benchmark_id": bench_id},
                    warnings=[f"Could not fetch {bench_name}"],
                )
            )
            continue

        if not bench_ngrams:
            results.append(
                ScoreResult(
                    name=f"contamination_{bench_name}",
                    score=1.0,
                    details={"benchmark": bench_name, "benchmark_id": bench_id},
                )
            )
            continue

        overlap_ngrams = dataset_ngrams & bench_ngrams
        overlap_rate = len(overlap_ngrams) / len(bench_ngrams)
        max_overlap = max(max_overlap, overlap_rate)

        score = 1.0 - min(overlap_rate, 1.0)
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
                    "overlap_rate": round(overlap_rate, 6),
                    "overlap_count": len(overlap_ngrams),
                    "total_eval_ngrams": len(bench_ngrams),
                    "method": f"ngram_{NGRAM_SIZE}",
                },
                warnings=warnings,
            )
        )

    return results, max_overlap


async def check_dataset_contamination(
    dataset_id: str,
    text_key: str = "text",
    token: Optional[str] = None,
    benchmarks: Optional[dict[str, str]] = None,
) -> list[ScoreResult]:
    """Legacy entry point: fetch dataset samples and check contamination."""
    from socbench.runner import fetch_samples
    from socbench.score import _extract_text

    samples = await fetch_samples(dataset_id, text_key=text_key, sample_size=2000, token=token)
    texts = [_extract_text(s, text_key) for s in samples]
    results, _ = await check_texts_against_benchmarks(texts, token=token, benchmarks=benchmarks)
    return results
