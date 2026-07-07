"""Training evaluation — load checkpoints, compute metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_loss_curve(checkpoint_dir: str) -> dict:
    """Load loss curve from training output."""
    loss_file = Path(checkpoint_dir) / "loss_curve.json"
    if loss_file.exists():
        with open(loss_file) as f:
            return json.load(f)
    return {"losses": [], "best_val_loss": None}


def compute_training_metrics(loss_curve: list[float], total_tokens: int = 0) -> dict:
    """Compute metrics from a loss curve.

    Returns:
        final_val_loss: Last validation loss
        convergence_speed: Steps to reach 90% of final loss
        loss_stability: Std dev of loss over last 10%
        relative_quality: Placeholder (computed across all datasets)
    """
    if not loss_curve:
        return {
            "final_val_loss": None,
            "convergence_speed": 0,
            "loss_stability": 0.0,
            "relative_quality": 0.0,
        }

    final_loss = loss_curve[-1]

    # Convergence speed: steps to reach 90% of final loss
    target_90 = loss_curve[0] - 0.9 * (loss_curve[0] - final_loss)
    convergence_steps = 0
    for i, loss in enumerate(loss_curve):
        if loss <= target_90:
            convergence_steps = i
            break

    # Loss stability: std dev of last 10% of curve
    last_10 = loss_curve[int(len(loss_curve) * 0.9) :]
    if len(last_10) > 1:
        mean_last = sum(last_10) / len(last_10)
        stability = (sum((l - mean_last) ** 2 for l in last_10) / len(last_10)) ** 0.5
    else:
        stability = 0.0

    return {
        "final_val_loss": final_loss,
        "convergence_steps": convergence_steps,
        "loss_stability": stability,
        "total_tokens": total_tokens,
    }


def compute_relative_quality(
    dataset_losses: dict[str, float],
) -> dict[str, float]:
    """Compute relative quality scores across all datasets.

    Lower loss = better data quality.
    relative_quality = avg_loss / this_loss (higher = better).
    """
    if not dataset_losses:
        return {}

    avg_loss = sum(dataset_losses.values()) / len(dataset_losses)
    return {
        ds_id: avg_loss / loss if loss > 0 else 0.0
        for ds_id, loss in dataset_losses.items()
    }
