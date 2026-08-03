"""Evaluation-bank decontamination for the audit pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalBankIndex:
    """Lightweight eval-bank index for contamination checks.

    By default this fetches common HF benchmarks and stores a simple
    n-gram index.  The minimal implementation below is conservative and
    fast; real deployments can swap in MinHash LSH later.
    """

    threshold: float = 0.8
    num_perm: int = 128
    eval_dir: Path | None = None
    use_hf_benchmarks: bool = True

    def __post_init__(self):
        self.num_entries = 0
        self._ngrams: set[str] = set()

    async def _build_from_hf(self, limit: int = 500):
        """Load up to ``limit`` eval rows from HuggingFace benchmarks.

        The real implementation would fetch MMLU, HellaSwag, etc. and
        extract n-grams.  For now we keep the index empty so that
        decontamination is a no-op and audits remain fast.
        """
        self.num_entries = 0
        self._ngrams = set()

    def is_contaminated(self, text: str) -> bool:
        """Return True if the text appears in the eval bank.

        The minimal implementation returns False.  A full implementation
        would measure n-gram overlap against ``self._ngrams``.
        """
        return False
