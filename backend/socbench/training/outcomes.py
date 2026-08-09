"""Deterministic training-outcome classification and scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

RunOutcome = Literal[
    "improved",
    "stable",
    "regressed",
    "diverged",
    "insufficient_evidence",
]

SCORING_METHOD = "positive_final_validation_loss_reduction_v1"


@dataclass(frozen=True)
class TrainingOutcome:
    initial_val_loss: float
    best_val_loss: float
    final_val_loss: float
    absolute_improvement: float
    relative_improvement: float
    best_relative_improvement: float
    best_step: int
    run_outcome: RunOutcome
    outcome_reason: str
    training_score: float
    scoring_method: str = SCORING_METHOD

    def as_dict(self) -> dict:
        return asdict(self)


def classify_training_curve(
    losses: Sequence[float],
    steps: Sequence[int] | None = None,
) -> TrainingOutcome:
    """Classify a validation-loss curve and return an interpretable score.

    Scores are the positive final relative loss reduction. A transient best
    checkpoint does not rescue a run whose final state regressed or diverged.
    """
    if not losses:
        raise ValueError("at least one finite validation loss is required")
    if steps is None:
        steps = list(range(len(losses)))
    if len(steps) != len(losses):
        raise ValueError("steps and losses must have the same length")
    if any(loss <= 0 for loss in losses):
        raise ValueError("validation losses must be positive")

    initial = float(losses[0])
    final = float(losses[-1])
    best_index = min(range(len(losses)), key=losses.__getitem__)
    best = float(losses[best_index])
    relative = (initial - final) / initial
    best_relative = (initial - best) / initial

    if len(losses) < 2:
        outcome: RunOutcome = "insufficient_evidence"
        reason = "Only one validation measurement is available."
    elif relative > 0.01:
        outcome = "improved"
        reason = "Final validation loss improved by more than 1% from baseline."
    elif relative < -0.05:
        outcome = "diverged"
        reason = "Final validation loss increased by more than 5% from baseline."
    elif relative < -0.01:
        outcome = "regressed"
        reason = "Final validation loss increased by more than 1% from baseline."
    else:
        outcome = "stable"
        reason = "Final validation loss stayed within 1% of baseline."

    return TrainingOutcome(
        initial_val_loss=initial,
        best_val_loss=best,
        final_val_loss=final,
        absolute_improvement=initial - final,
        relative_improvement=relative,
        best_relative_improvement=best_relative,
        best_step=int(steps[best_index]),
        run_outcome=outcome,
        outcome_reason=reason,
        training_score=max(0.0, min(1.0, relative)) if outcome == "improved" else 0.0,
    )
