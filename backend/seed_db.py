"""Seed the local SQLite DB with real Socbench examinations.

Scores a curated set of datasets spanning every dataset form
(pretraining, SFT, preference, evaluation, task, multimodal) so the
website has real content immediately. Re-runnable.

Usage:
    python seed_db.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select

from socbench.db import async_session_factory, init_db
from socbench.models import ContaminationRow, DatasetRow, LeaderboardRow, ScoreRow
from socbench.runner import run_socbench_scoring

# Curated datasets covering all forms of datasets.
SEED_DATASETS = [
    "databricks/databricks-dolly-15k",      # posttraining-sft
    "Anthropic/hh-rlhf",                     # posttraining-preference
    "Skylion007/openwebtext",                # pretraining-web
    "openai/gsm8k",                          # task-qa / evaluation
    "stanfordnlp/sst2",                      # task-classification
    "tatsu-lab/alpaca",                      # posttraining-sft
    # Expanded coverage across categories
    "teknium/OpenHermes-2.5",                # posttraining-sft (large)
    "openai/openai_humaneval",               # evaluation (code)
    "cais/mmlu",                             # evaluation (knowledge)
    "Open-Orca/OpenOrca",                    # posttraining-sft/reasoning
    "nvidia/HelpSteer2",                     # posttraining-preference
    "HuggingFaceM4/the_cauldron",            # multimodal (VLM)
    "lmms-lab/LLaVA-OneVision-Data",         # multimodal
    # Code / agent / summarization / translation / QA coverage
    "m-a-p/CodeFeedback-Filtered-Instruction",  # code (SFT)
    "google-research-datasets/mbpp",             # code (evaluation)
    "OpenAssistant/oasst2",                      # agent / assistant (SFT)
    "EdinburghNLP/xsum",                         # task-summarization
    "Helsinki-NLP/opus_books",                   # task-translation
    "rajpurkar/squad",                           # task-qa
    # --- Large expansion: training data across all categories ---
    # Pretraining (more web/math/wiki corpora)
    "HuggingFaceFW/fineweb-edu",                 # pretraining-web (edu-filtered)
    "open-web-math/open-web-math",               # pretraining-math
    "Salesforce/wikitext",                       # pretraining-web (wiki)
    "wikimedia/wikipedia",                       # pretraining-web (wiki, 2024)
    "scholarweave/arxiv-latex",                  # pretraining-science (arxiv)
    # SFT (high-quality instruction data)
    "HuggingFaceH4/ultrachat_200k",              # posttraining-sft (large, multi-turn)
    "HuggingFaceH4/no_robots",                   # posttraining-sft (human-written)
    "garage-bAInd/Open-Platypus",                # posttraining-sft (reasoning/STEM)
    "jondurbin/airoboros-2.2.1",                 # posttraining-sft (diverse instruct)
    "shibing624/sharegpt_gpt4",                  # posttraining-sft (ShareGPT GPT-4)
    "open-thoughts/OpenThoughts-114k",           # posttraining-sft (reasoning traces)
    "WithinUsAI/claude_mythos_distilled_25k",    # posttraining-sft (distilled)
    # Preference / DPO
    "HuggingFaceH4/ultrafeedback_binarized",     # posttraining-preference (large)
    "Intel/orca_dpo_pairs",                      # posttraining-preference
    "jondurbin/truthy-dpo-v0.1",                 # posttraining-preference (truthfulness)
    "Skywork/Skywork-Reward-Preference-80K-v0.2", # posttraining-preference (reward model)
    "argilla/distilabel-intel-orca-dpo-pairs",   # posttraining-preference
    # Agent traces (trending — let quality scorers separate real from slop)
    "Glint-Research/Fable-5-traces",             # posttraining-agent (trending #1)
    "nvidia/Open-SWE-Traces",                    # posttraining-agent (SWE traces)
    "AletheiaResearch/GLM-5.2-Agent",            # posttraining-agent (agent traces)
    "armand0e/claude-fable-5-claude-code",       # posttraining-agent (Fable-5 derived)
    "Crownelius/Complete-FABLE.5-traces-2M",     # posttraining-agent (aggregated traces)
    "CodeDevX/Vibe-Coding-Instruct",             # posttraining-agent (code+agent)
    "AletheiaResearch/GPT-5.5-Codex",            # posttraining-agent (code agent traces)
    # Reasoning / math
    "microsoft/orca-math-word-problems-200k",    # posttraining-reasoning (math)
    "nvidia/OpenMathInstruct-2",                 # posttraining-reasoning (math, large)
    "meta-math/MetaMathQA",                      # posttraining-reasoning (math QA)
    "TIGER-Lab/MathInstruct",                    # posttraining-reasoning (math instruct)
    # Code
    "HuggingFaceH4/instruction-dataset",         # posttraining-sft (code instruct)
    # Multimodal
    "HuggingFaceH4/llava-instruct-mix-vsft",     # multimodal (VLM instruct)
    # Safety
    "LLM-LAT/harmful-dataset",                   # safety (harmful prompts, DPO format)
    "JailbreakV-28K/JailBreakV-28k",             # safety (jailbreak, multimodal)
]

DIM_KEYS = [
    "quality",
    "diversity",
    "utility",
    "documentation",
    "popularity",
    "freshness",
    "pii_safety",
]


def _score_of(result: dict, key: str) -> float | None:
    block = result.get(key)
    if isinstance(block, dict) and "score" in block:
        return block["score"]
    return None


async def seed() -> None:
    await init_db()

    async with async_session_factory() as session:
        for idx, ds_id in enumerate(SEED_DATASETS):
            if idx > 0:
                await asyncio.sleep(8)  # avoid HF viewer API rate-limiting
            print(f"Examining {ds_id} ...", flush=True)

            result = None
            for attempt in range(3):
                try:
                    result = await run_socbench_scoring(ds_id, sample_size=2000)
                except Exception as exc:  # noqa: BLE001
                    print(f"  attempt {attempt+1} error {ds_id}: {exc}", flush=True)
                    result = None
                if result and "error" not in result:
                    break
                await asyncio.sleep(10 * (attempt + 1))

            if not result or "error" in result:
                reason = result.get("error") if result else "no result"
                print(f"  skipped {ds_id}: {reason}", flush=True)
                continue

            meta = result.get("metadata", {})
            tags = meta.get("tags", []) or []
            category = result.get("category")

            stmt = select(DatasetRow).where(DatasetRow.hf_id == ds_id)
            ds = (await session.execute(stmt)).scalar_one_or_none()
            if ds is None:
                ds = DatasetRow(hf_id=ds_id, name=ds_id)
                session.add(ds)
                await session.flush()

            ds.name = ds_id
            ds.description = meta.get("description", "")
            ds.license = (meta.get("license") or [None])[0] if isinstance(meta.get("license"), list) else meta.get("license")
            ds.tags = tags
            ds.downloads = meta.get("downloads", 0)
            ds.likes = meta.get("likes", 0)
            ds.last_scored = __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            )

            # Store each dimension as a ScoreRow.
            await session.execute(
                select(ScoreRow).where(ScoreRow.dataset_id == ds.id)
            )
            existing = (await session.execute(
                select(ScoreRow).where(ScoreRow.dataset_id == ds.id)
            )).scalars().all()
            for ex in existing:
                await session.delete(ex)
            await session.flush()

            for key in DIM_KEYS + ["coverage"]:
                val = _score_of(result, key)
                if val is None:
                    continue
                block = result.get(key, {}) if key != "coverage" else {"details": result.get("coverage")}
                session.add(ScoreRow(
                    dataset_id=ds.id,
                    scorer_name=key,
                    score=val,
                    details=block.get("details", {}) if isinstance(block, dict) else {},
                    warnings=block.get("warnings", []) if isinstance(block, dict) else [],
                ))

            # Persist per-benchmark contamination rows (replace any existing).
            existing_cont = (await session.execute(
                select(ContaminationRow).where(ContaminationRow.dataset_id == ds.id)
            )).scalars().all()
            for ex in existing_cont:
                await session.delete(ex)
            await session.flush()

            for check in result.get("contamination_checks", []):
                details = check.get("details", {}) or {}
                # Skip benchmarks that could not be fetched (no overlap_rate).
                if "overlap_rate" not in details:
                    continue
                session.add(ContaminationRow(
                    dataset_id=ds.id,
                    benchmark_name=details.get("benchmark", check.get("name", "").replace("contamination_", "")),
                    overlap_rate=details.get("overlap_rate", 0.0),
                    overlap_count=details.get("overlap_count", 0),
                    total_eval=details.get("total_eval_ngrams", 0),
                    method=details.get("method", "ngram_13"),
                ))

            # Leaderboard entry with all dimensions.
            lb_stmt = select(LeaderboardRow).where(LeaderboardRow.dataset_id == ds.id)
            lb = (await session.execute(lb_stmt)).scalar_one_or_none()
            if lb is None:
                lb = LeaderboardRow(dataset_id=ds.id)
                session.add(lb)

            dims = {k: _score_of(result, k) for k in DIM_KEYS}
            present = [v for v in dims.values() if v is not None]
            combined = round(sum(present) / len(present), 4) if present else None

            lb.category = category
            lb.quality = dims["quality"]
            lb.diversity = dims["diversity"]
            lb.utility = dims["utility"]
            lb.documentation = dims["documentation"]
            lb.popularity = dims["popularity"]
            lb.freshness = dims["freshness"]
            lb.pii_safety = dims["pii_safety"]
            lb.contamination_score = result.get("contamination_rate")
            lb.auto_score = combined
            lb.combined_score = combined
            lb.category = category

            await session.commit()
            cont_rate = result.get("contamination_rate")
            n_cont = len(result.get("contamination_checks", []))
            print(f"  done: q={dims['quality']} d={dims['diversity']} u={dims['utility']} combined={combined} cont={cont_rate} ({n_cont} benchmarks)", flush=True)

    # Assign ranks.
    async with async_session_factory() as session:
        entries = (await session.execute(
            select(LeaderboardRow).order_by(LeaderboardRow.combined_score.desc().nullslast())
        )).scalars().all()
        for i, entry in enumerate(entries, start=1):
            entry.rank = i
        await session.commit()
        print(f"Seeded {len(entries)} datasets.", flush=True)


if __name__ == "__main__":
    asyncio.run(seed())
