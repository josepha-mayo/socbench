# Socbench — "The unexamined dataset is not worth training on."

Scientific dataset intelligence. Multi-dimension scoring, provenance tracking,
hierarchical classification, contamination checking, and training impact measurement.

## What It Is

Not a model benchmark. Not a leaderboard. It's the IMDb for datasets.

When someone is about to build a model, Socbench answers:
**"Which datasets should I train on, and why?"**

## Architecture

```
backend/
├── socbench/
│   ├── score.py              # Multi-dimension scoring (quality/diversity/utility)
│   ├── categories.py         # 21 hierarchical categories
│   ├── provenance.py         # Dataset→model→paper mapping
│   ├── recommendations.py    # "Best for:" engine
│   ├── runner.py             # Orchestrate all stages
│   ├── scoring/              # Stage 1: Automated metrics
│   │   ├── dedup.py          # Exact + MinHash + line-level
│   │   ├── format.py         # Schema consistency, encoding, code parseability
│   │   ├── tokens.py         # Token length distribution
│   │   ├── language.py       # Language detection (fasttext)
│   │   ├── pii.py            # PII scanning (regex default, presidio opt-in)
│   │   ├── quality.py        # Gopher rules + FineWeb filters
│   │   └── code.py           # AST parse rate, complexity, boilerplate
│   ├── contamination/        # Stage 2: Eval benchmark overlap
│   ├── discovery/            # HF dataset scanner + qualifier
│   ├── training/             # Stage 3: GPT-2 124M training pipeline
│   ├── api/                  # FastAPI routes
│   └── cli.py                # Typer CLI
├── pyproject.toml
└── tests/
frontend/
├── src/app/
│   ├── page.tsx              # Leaderboard with multi-dimension columns
│   ├── datasets/[id]/page.tsx # Detail page with score bars, domain coverage, provenance
│   ├── discover/page.tsx     # HF discovery with category labels
│   └── about/page.tsx        # Methodology with research citations
├── tailwind.config.ts        # Arxiv-style: EB Garamond serif, red accents
└── package.json
```

## Multi-Dimension Scoring

Not one composite number. Separate dimensions for separate questions.

| Dimension | What It Measures |
|---|---|---|
| **Quality** | Gopher rules, FineWeb filters, DCLM criteria, dedup rate |
| **Diversity** | Text uniqueness, token spread, domain distribution |
| **Utility** | How actionable — format quality, schema conformance |
| **Documentation** | Dataset card completeness, license clarity |
| **Popularity** | Downloads, likes, community adoption |
| **Freshness** | Recency of updates |
| **PII Safety** | PII density, severity-weighted |
| **Contamination** | 13-gram overlap with HumanEval, MBPP, GSM8K, MMLU, ARC, HellaSwag, TruthfulQA |

## Hierarchical Categories (21)

Different metrics per category. A Ferrari isn't "better" than a truck.

```
Pretraining → Web, Code, Math, Science, Books, Multilingual
Post-Training → SFT, Preference/DPO, Tool Use, Agent, Safety, Reasoning
Evaluation
Task → Classification, Translation, QA, Summarization
Multimodal
```

## Provenance Tracking

Every dataset shows which models trained on it.

```
OpenHermes-2.5 → Nous Hermes 2, OpenHermes models
StarCoderData → StarCoder (15.5B)
The Stack v2 → StarCoder2 (15B)
FineWeb → FineWeb models
Dolma → OLMo (7B, 1B)
```

## Training Impact (Stage 3)

For top-tier datasets only. GPT-2 124M from scratch.
1B tokens per dataset. Research proves 125M models predict
data quality scaling for 3B models (Ankner et al., 2024).

## Discovery Pipeline

Scan HuggingFace daily for new/trending datasets.
Minimum thresholds: ≥1K downloads, ≥10 likes, ≥1K rows.
Automated scoring runs instantly. Top-tier datasets queue for training.

## Research Sources

- Gopher quality rules — Rae et al., 2021
- FineWeb filters — Penedo et al., NeurIPS 2024
- DCLM filtering + fastText classifier — Li et al., NeurIPS 2024
- SlimPajama deduplication (13-gram, Jaccard 0.8) — Cerebras, 2023
- RedPajama-V2 46 quality signals — Weber et al., NeurIPS 2024
- Dolma curation — Soldaini et al., ACL 2024
- 125M proxy models predict 3B scaling — Ankner et al., 2024
- DoReMi domain reweighting — Xie et al., NeurIPS 2023
- DataComp standardized evaluation — Gadre et al., NeurIPS 2023

## CLI Commands

```bash
socbench discover --search "code" --limit 30
socbench assess "bigcode/starcoderdata"
socbench classify "databricks/databricks-dolly-15k"
socbench provenance "bigcode/starcoderdata"
socbench recommendations "teknium/OpenHermes-2.5"
socbench leaderboard --top 20
socbench serve
```

## Tech Stack

- Backend: Python 3.11+, FastAPI, SQLAlchemy+asyncpg, Pydantic, Typer
- Scoring: datasketch, tiktoken, fasttext-langdetect, presidio-analyzer (opt-in)
- Training: PyTorch, NanoGPT architecture
- Frontend: Next.js 14, Tailwind CSS, EB Garamond (arxiv aesthetic)
- Infra: PostgreSQL 16, Docker Compose
