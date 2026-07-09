import asyncio
from sqlalchemy import select
from socbench.db import async_session_factory
from socbench.models import DatasetRow, LeaderboardRow
from socbench.categories import classify_dataset
async def main():
    async with async_session_factory() as s:
        rows = (await s.execute(select(DatasetRow))).scalars().all()
        for ds in rows:
            cat = classify_dataset(ds.tags or [], ds.description or "", ds.hf_id)
            lb = (await s.execute(select(LeaderboardRow).where(LeaderboardRow.dataset_id==ds.id))).scalar_one_or_none()
            if lb:
                lb.category = cat
            print(f"{cat:26s} <- {ds.hf_id}")
        await s.commit()
asyncio.run(main())
