"""Base scorer protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ScoreResult:
    name: str
    score: float  # 0.0 - 1.0
    details: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Scorer(Protocol):
    async def score(self, samples: list[dict], text_key: str = "text") -> ScoreResult: ...
