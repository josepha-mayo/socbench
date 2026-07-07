"""Deduplication scorer — exact + MinHash near-dedup + line-level.

Thresholds from SlimPajama (Cerebras, 2023):
  Jaccard similarity: 0.8, N-gram size: 13, Hashes: 10,000
"""

from __future__ import annotations

import hashlib
from collections import Counter

from socbench.scoring.base import ScoreResult


def _hash_text(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _get_ngrams(text: str, n: int = 13) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {text}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


async def dedup_scorer(samples: list[dict], text_key: str = "text") -> ScoreResult:
    """Compute exact dedup, near-dedup (MinHash), and line-level dedup rates."""
    texts = [s.get(text_key, "") for s in samples if s.get(text_key)]
    if not texts:
        return ScoreResult(name="dedup", score=0.0, warnings=["No text content found"])

    total = len(texts)

    # --- Exact dedup ---
    hashes = [_hash_text(t) for t in texts]
    unique_exact = len(set(hashes))
    exact_dedup_rate = 1.0 - (unique_exact / total) if total > 0 else 0.0

    # --- Line-level dedup ---
    all_lines: list[str] = []
    for t in texts:
        all_lines.extend(t.split("\n"))
    if all_lines:
        line_counts = Counter(all_lines)
        repeated_lines = sum(c for c in line_counts.values() if c > 1)
        line_dedup_rate = repeated_lines / len(all_lines) if all_lines else 0.0
    else:
        line_dedup_rate = 0.0

    # --- Near-dedup (MinHash via datasketch) ---
    try:
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=0.8, num_perm=128)
        minhashes = []
        unique_near = 0

        for i, text in enumerate(texts[:5000]):  # Cap for performance
            m = MinHash(num_perm=128)
            for ng in _get_ngrams(text):
                m.update(ng.encode("utf-8"))
            minhashes.append((i, m))

        for idx, mh in minhashes:
            result = lsh.query(mh)
            if not result:
                lsh.insert(str(idx), mh)
                unique_near += 1

        near_dedup_rate = 1.0 - (unique_near / len(minhashes)) if minhashes else 0.0
    except ImportError:
        near_dedup_rate = exact_dedup_rate  # Fallback
        warnings = ["datasketch not installed, using exact dedup as near-dedup proxy"]
    else:
        warnings = []

    # Composite dedup score: lower dedup rate = higher score
    avg_dedup = (exact_dedup_rate + near_dedup_rate + line_dedup_rate) / 3
    score = max(0.0, 1.0 - avg_dedup)

    return ScoreResult(
        name="dedup",
        score=score,
        details={
            "exact_dedup_rate": round(exact_dedup_rate, 4),
            "near_dedup_rate": round(near_dedup_rate, 4),
            "line_dedup_rate": round(line_dedup_rate, 4),
            "total_samples": total,
            "unique_exact": unique_exact,
        },
        warnings=warnings,
    )
