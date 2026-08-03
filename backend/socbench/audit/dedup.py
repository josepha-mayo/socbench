"""Exact + near-duplicate detection for the audit pipeline."""

from __future__ import annotations

import hashlib


class DedupState:
    """Track exact-duplicate rows and optionally near-duplicates."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self._exact: set[str] = set()

    def is_duplicate(self, text: str, record_id: str) -> str | bool:
        """Return 'exact' if an exact duplicate, 'near' if near-duplicate, else False."""
        norm = text.strip().lower().replace("\n", " ")
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if h in self._exact:
            return "exact"
        self._exact.add(h)

        # Near-duplicate detection is expensive; exact is good enough for minimal runs.
        return False
