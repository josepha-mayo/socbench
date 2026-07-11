"""Import v12 Kaggle training results into the leaderboard and training_runs table."""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./socbench.db")

from sqlalchemy import select
from socbench.db import async_session_factory
from socbench.models import DatasetRow, LeaderboardRow, TrainingRunRow
from socbench.training.evaluate import compute_relative_quality

RESULTS_DIR = Path(r"C:\Users\USER\.vscode\vibe\kaggle_socbench\results_v12")

async def main():
    # Load all results
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == "summary.json":
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        results.append(data)

    # Compute relative quality scores (higher = better, lower loss = better)
    losses = {r["dataset_id"]: r["final_val_loss"] for r in results}
    rel = compute_relative_quality(losses)

    # Normalize relative quality to 0-1 across all datasets so best=1, worst=0
    raw_values = list(rel.values())
    min_raw = min(raw_values) if raw_values else 0.0
    max_raw = max(raw_values) if raw_values else 1.0
    raw_range = max_raw - min_raw if max_raw > min_raw else 1.0
    norm = {
        ds_id: max(0.0, min(1.0, (v - min_raw) / raw_range))
        for ds_id, v in rel.items()
    }

    async with async_session_factory() as session:
        for r in results:
            ds_id = r["dataset_id"]
            ds = (await session.execute(select(DatasetRow).where(DatasetRow.hf_id == ds_id))).scalar_one_or_none()
            if not ds:
                print(f"[SKIP] {ds_id}: not in DB")
                continue

            # Check if training run already exists
            existing = (await session.execute(select(TrainingRunRow).where(TrainingRunRow.dataset_id == ds.id))).scalar_one_or_none()
            if existing:
                existing.final_val_loss = r["final_val_loss"]
                existing.loss_curve = [x["val_loss"] for x in r["loss_curve"]]
                existing.convergence_steps = r["loss_curve"][-1]["step"] if r["loss_curve"] else 0
                existing.tokens_seen = r["n_tokens"]
                existing.model_config = {
                    "parameters": r["parameters"],
                    "max_iters": r["max_iters"],
                    "gpu": r["gpu"],
                    "pytorch_version": r["pytorch_version"],
                    "batch_size": r["batch_size"],
                    "use_fp16": r["use_fp16"],
                }
                existing.eval_scores = {"relative_quality": rel.get(ds_id, 0.0), "normalized_score": norm.get(ds_id, 0.0)}
                print(f"[UPDATE] {ds_id}")
            else:
                run = TrainingRunRow(
                    dataset_id=ds.id,
                    final_val_loss=r["final_val_loss"],
                    loss_curve=[x["val_loss"] for x in r["loss_curve"]],
                    convergence_steps=r["loss_curve"][-1]["step"] if r["loss_curve"] else 0,
                    tokens_seen=r["n_tokens"],
                    model_config={
                        "parameters": r["parameters"],
                        "max_iters": r["max_iters"],
                        "gpu": r["gpu"],
                        "pytorch_version": r["pytorch_version"],
                        "batch_size": r["batch_size"],
                        "use_fp16": r["use_fp16"],
                    },
                    eval_scores={"relative_quality": rel.get(ds_id, 0.0), "normalized_score": norm.get(ds_id, 0.0)},
                )
                session.add(run)
                print(f"[INSERT] {ds_id}")

            # Update leaderboard training_score
            lb = (await session.execute(select(LeaderboardRow).where(LeaderboardRow.dataset_id == ds.id))).scalar_one_or_none()
            if lb:
                score = norm.get(ds_id, 0.0)
                lb.training_score = score
                lb.combined_score = (
                    (lb.quality or 0) * 0.35 +
                    (lb.diversity or 0) * 0.25 +
                    (lb.utility or 0) * 0.15 +
                    (lb.documentation or 0) * 0.05 +
                    (lb.popularity or 0) * 0.05 +
                    (lb.freshness or 0) * 0.05 +
                    score * 0.10
                )
                print(f"[LEADERBOARD] {ds_id}: training_score={score:.4f}")

        await session.commit()

    print("\nImport complete.")

if __name__ == "__main__":
    asyncio.run(main())
