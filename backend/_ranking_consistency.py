"""Check whether our leaderboard ranking is consistent with training losses."""
import asyncio, json, math, os
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./socbench.db")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select
from socbench.db import async_session_factory
from socbench.models import DatasetRow, LeaderboardRow

RESULTS_DIR = Path(r"C:\Users\USER\.vscode\vibe\kaggle_socbench\results_v12")

async def main():
    with open(RESULTS_DIR / "summary.json", encoding="utf-8") as f:
        results = json.load(f)

    async with async_session_factory() as session:
        rows = []
        for r in results:
            ds_id = r["dataset"]
            ds = (await session.execute(select(DatasetRow).where(DatasetRow.hf_id == ds_id))).scalar_one_or_none()
            if not ds:
                continue
            lb = (await session.execute(select(LeaderboardRow).where(LeaderboardRow.dataset_id == ds.id))).scalar_one_or_none()
            quality = (lb.quality or 0) * 100  # 0-100
            final_loss = r["final_val_loss"]
            rows.append({
                "dataset": ds_id,
                "quality": quality,
                "final_loss": final_loss,
                "perplexity": math.exp(final_loss),
                "training_score": (lb.training_score or 0) * 100,
                "combined_score": (lb.combined_score or 0) * 100,
            })

    # Sort by quality and training score
    by_quality = sorted(rows, key=lambda x: x["quality"], reverse=True)
    by_loss = sorted(rows, key=lambda x: x["final_loss"])

    print("Comparison: Quality rank vs. Training Loss rank (lower loss = better)\n")
    print(f"{'Dataset':<45} {'Quality':>8} {'TrainScr':>8} {'Loss':>10} {'PPL':>8}")
    print("-" * 85)
    for r in sorted(rows, key=lambda x: x["final_loss"]):
        print(f"{r['dataset']:<45} {r['quality']:>8.1f} {r['training_score']:>8.1f} {r['final_loss']:>10.4f} {r['perplexity']:>8.2f}")

    # Correlation (Spearman rank)
    def rank(arr, key, reverse=False):
        s = sorted(enumerate(arr), key=lambda x: x[1][key], reverse=reverse)
        r = [0] * len(arr)
        for i, (idx, _) in enumerate(s):
            r[idx] = i + 1
        return r

    quality_ranks = rank(rows, "quality", reverse=True)
    loss_ranks = rank(rows, "final_loss", reverse=False)

    n = len(rows)
    d2 = sum((q - l) ** 2 for q, l in zip(quality_ranks, loss_ranks))
    spearman = 1 - (6 * d2) / (n * (n * n - 1))

    print(f"\nSpearman rank correlation between quality score and final val loss: {spearman:.3f}")
    print("(1.0 = perfect agreement, -1.0 = perfect disagreement, 0 = no relationship)")

    # Find outliers
    quality_order = {r["dataset"]: i for i, r in enumerate(by_quality)}
    loss_order = {r["dataset"]: i for i, r in enumerate(by_loss)}
    diffs = []
    for r in rows:
        ds = r["dataset"]
        diffs.append((ds, quality_order[ds] - loss_order[ds], r["quality"], r["final_loss"]))

    print("\nOverranked by quality (quality rank much better than loss rank):")
    for ds, diff, q, loss in sorted(diffs, key=lambda x: -x[1])[:5]:
        print(f"  {ds}: quality rank {quality_order[ds]+1}, loss rank {loss_order[ds]+1}, diff {diff:+d}")

    print("\nUnderranked by quality (loss rank much better than quality rank):")
    for ds, diff, q, loss in sorted(diffs, key=lambda x: x[1])[:5]:
        print(f"  {ds}: quality rank {quality_order[ds]+1}, loss rank {loss_order[ds]+1}, diff {diff:+d}")

if __name__ == "__main__":
    asyncio.run(main())
