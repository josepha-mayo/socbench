# Socbench — Scientific Dataset Quality Benchmark

A registry of open datasets with standardized quality metrics, contamination checks,
and training impact scores. Scientific benchmarking for the ML community.

## Mission

Nobody is systematically benchmarking dataset quality. We build the infrastructure
to do it rigorously — not vibes, not opinions, not "this dataset feels good."
Measurable, reproducible, research-backed metrics.

---

## The 4 Stages

### Stage 1: Automated Quality Metrics (CPU, no GPU)
Run on every discovered dataset. No training required.
Results available instantly.

### Stage 2: Contamination Check (CPU)
Check overlap between dataset content and standard eval benchmarks.
Detect data leakage before it poisons research.

### Stage 3: Training Impact (Kaggle 2x T4)
Train GPT-2 124M from scratch on each dataset. Measure loss curves.
Relative comparison across datasets. The differentiator.

### Stage 4: Leaderboard + Provenance
Composite scores, license tracking, model-to-dataset mapping.
The actual product.

---

## Dataset Quality Criteria (Research-Backed)

Every metric below is sourced from peer-reviewed research. We do not invent
arbitrary thresholds. Citations are inline.

### Minimum Thresholds for "Serious" Datasets

| Metric        | Minimum   | Good      | Great     | Rationale                                      |
|---------------|-----------|-----------|-----------|------------------------------------------------|
| Downloads     | >1,000    | >10,000   | >100,000  | HF power law: top 82 = 80% of traffic          |
| Likes         | >10       | >50       | >200      | Trending datasets typically have 50-1000+      |
| Rows          | >1,000    | >10,000   | >100,000  | Statistical significance for quality metrics   |
| Bytes         | >1 MB     | >10 MB    | >100 MB   | Minimum for meaningful analysis                |
| Dataset card  | Present   | Complete  | Detailed  | 86% of top-100 have complete cards             |
| License       | Present   | Permissive| Clear     | MIT, Apache-2.0, CC-BY preferred               |
| Last modified | <1 year   | <6 months | <3 months | Recency indicates active maintenance           |

Datasets below these thresholds are skipped. No exceptions.

### Gopher Quality Rules (Rae et al., 2021)

The canonical web data quality filters. Applied to every text dataset.

| Metric                              | Threshold |
|-------------------------------------|-----------|
| Duplicate line fraction             | < 0.30    |
| Duplicate paragraph fraction        | < 0.30    |
| Duplicate line character fraction   | < 0.20    |
| Duplicate paragraph character fraction | < 0.20 |
| Top 2-gram character fraction       | < 0.20    |
| Top 3-gram character fraction       | < 0.18    |
| Top 4-gram character fraction       | < 0.16    |
| Duplicate 5-gram character fraction | < 0.15    |
| Duplicate 6-gram character fraction | < 0.14    |
| Duplicate 7-gram character fraction | < 0.13    |
| Duplicate 8-gram character fraction | < 0.12    |
| Duplicate 9-gram character fraction | < 0.11    |
| Duplicate 10-gram character fraction| < 0.10    |

### FineWeb Filters (Penedo et al., NeurIPS 2024)

Three custom filters that remove ~22% of tokens and boost benchmarks ~1%.

| Filter                                      | Threshold  |
|---------------------------------------------|------------|
| Fraction of lines ending with punctuation   | > 0.12     |
| Fraction of characters in duplicated lines  | < 0.10     |
| Fraction of lines shorter than 30 characters| < 0.67     |

Key finding: Per-crawl MinHash dedup outperforms global dedup.

### DCLM Filtering (Li et al., NeurIPS 2024)

Model-based filtering is king. A simple fastText bigram classifier trained on
~400K documents outperforms perplexity filtering, PageRank, AskLLM, SemDedup.

- Keep top 10% of documents scored by classifier
- Best positive examples: OpenHermes 2.5 + high-scoring ELI5 Reddit posts
- Human quality judgments have limited value for identifying training data quality

### RedPajama-V2 Quality Signals (Weber et al., NeurIPS 2024)

46 quality signals per document across categories:

1. **Natural language**: fraction of caps, ellipsis lines, unique words, word length
2. **Repetitiveness**: Gopher repetition thresholds (above)
3. **Content-based**: blocklist words, blocked URLs, topic clustering
4. **ML heuristics**: fastText classifier scores, importance weights, perplexity
5. **Deduplication**: exact duplicate IDs (Bloom filter), MinHash signatures

### Dolma Quality Approach (Soldaini et al., ACL 2024)

- Code from GitHub: filter files <100KB, small repositories
- Remove auto-generated code files
- Filter by programming language distribution
- Fuzzy deduplication using Bloom filters (Rust implementation)

---

## Deduplication Standards

### Exact Deduplication
- Hash-based: SHA256 of normalized text
- O(n) per shard, catches verbatim duplicates

### Near-Duplicate Deduplication (MinHash + LSH)
- From SlimPajama (Cerebras, 2023):
  - Jaccard similarity threshold: 0.8
  - N-gram size: 13
  - Number of hashes: 10,000
  - LSH bands: 9, range: 13
  - Connected components: keep one document per cluster

### Line-Level Deduplication
- Count repeated lines across corpus
- High line-dedup ratio (>0.40) indicates boilerplate
- Critical for code: license headers, imports repeat

### Per-Crawl vs Global
- FineWeb key finding: per-crawl dedup outperforms global dedup
- Use per-crawl for web data, global for curated collections

---

## Contamination Detection

### Benchmarks to Check Against

| Benchmark       | HuggingFace ID                     | What to check              |
|-----------------|-------------------------------------|----------------------------|
| HumanEval       | openai/openai_humaneval             | Python function prompts    |
| MBPP            | google-research-datasets/mbpp       | Python problem descriptions|
| GSM8K           | openai/gsm8k                        | Math word problems         |
| MMLU            | cais/mmlu                           | MCQ questions              |
| ARC             | allenai/ai2_arc                     | Science questions          |
| HellaSwag       | Rowan/hellaswag                      | Sentence completions       |
| TruthfulQA      | truthfulqa/truthful_qa               | QA pairs                   |

### Method
- 13-gram exact match (GPT-3 paper methodology)
- Hash every 13-gram from eval items
- Check presence in training corpus
- Contamination rate = eval_items_with_match / total_eval_items

---

## Training Impact (Stage 3)

### Model: GPT-2 124M

Why 124M (not 350M):
- Research proves 125M predicts data quality scaling for 3B models (Ankner et al., 2024)
- DoReMi used 280M proxy for 8B target
- 3x faster training than 350M on T4
- NanoGPT optimized specifically for this architecture

### Architecture Config

```
GPT-2 124M:
  - Layers: 12
  - Heads: 12
  - d_model: 768
  - Context: 1024 tokens
  - Vocab: 50,257 (GPT-2 BPE)
  - Params: 124,439,808
```

### Training Protocol

```
Optimizer: AdamW (beta1=0.9, beta2=0.95, wd=0.1)
LR: 6e-4 peak, cosine decay to 6e-5
Warmup: 375M tokens (linear)
Batch size: 0.5M → 4M tokens (warmup then fixed)
Tokens per dataset: 1B (1 epoch)
Mixed precision: FP16 (T4 has no BF16)
Flash Attention: enabled
Framework: NanoGPT (modded-nanogpt for speed)
```

### Kaggle 2x T4 Budget

| Parameter           | Value                    |
|---------------------|--------------------------|
| GPUs                | 2x T4 (16GB each)       |
| Max runtime         | 12 hours per session     |
| Weekly quota        | ~30 hours                |
| Effective throughput| ~25,000-40,000 tok/s     |
| Max tokens in 12h   | ~0.8-1.5 billion         |
| 1B tokens possible? | Yes, fits in 12 hours    |

### What We Measure Per Dataset

| Metric               | Description                                    |
|----------------------|------------------------------------------------|
| final_val_loss       | How well model compressed this data            |
| loss_curve           | Full curve for visualization                   |
| convergence_speed    | Steps to 90% of final loss                     |
| loss_stability       | Std dev of loss over last 10% of training      |
| downstream_accuracy  | Pass@1 on HumanEval, MBPP (optional)           |
| relative_quality     | Loss ratio vs average across all datasets      |

### Composite Score Formula

```python
# Combined score: weighted blend of automated + training impact
combined = (
    0.40 * auto_score +          # Stage 1 composite
    0.40 * training_score +      # Stage 3 relative quality
    0.20 * (1 - contamination)   # Stage 2 penalty
)
```

---

## Dataset Discovery Pipeline

### Sources

| Source               | API                                         | What it finds              |
|----------------------|---------------------------------------------|----------------------------|
| HuggingFace Hub      | sort=createdAt&direction=-1                 | Brand new datasets         |
| HuggingFace Hub      | sort=trendingScore&direction=-1             | Rising datasets            |
| HuggingFace Hub      | sort=likes30d&direction=-1                  | Expert-approved recent     |
| CodeSOTA (PwC)       | GET /api/sota/{task}                        | Benchmark-linked datasets  |
| HuggingFace Papers   | Scrape paper cards                          | Paper-linked datasets      |

### Categories (All Levels)

| Category              | HuggingFace Tag                              |
|-----------------------|----------------------------------------------|
| Text Classification   | task_categories:text-classification           |
| Code Generation       | task_categories:text-generation + code tag    |
| Instruction Following | task_categories:text-generation               |
| Conversation/Chat     | task_categories:text-generation               |
| Math/Reasoning        | task_categories:text-generation               |
| Question Answering    | task_categories:question-answering            |
| Summarization         | task_categories:summarization                 |
| Translation           | task_categories:translation                   |
| Retrieval/RAG         | task_categories:text-retrieval                |
| Image-Text            | task_categories:image-text-to-text            |
| Audio/Speech          | task_categories:automatic-speech-recognition  |
| Tabular               | task_categories:tabular-classification        |

### Qualification Pipeline

```
New dataset found
  → Check: private? gated? → skip if yes
  → Check: <1K downloads, <10 likes, <1K rows → skip if yes
  → Check: no dataset card → flag
  → Check: no license → flag
  → Run Stage 1 (automated metrics) → score it
  → Run Stage 2 (contamination) → flag if contaminated
  → Queue for Stage 3 (training) if score > threshold
```

---

## Agent Trace Quality (Research-Backed)

### From AgentBank (Song et al., 2024) and AgentTrek (Xu et al., 2024)

High-quality agent trajectory contains:
1. High-level goal description
2. Interleaved observations (what agent sees at each step)
3. Natural language reasoning (chain-of-thought rationale)
4. Grounded actions (specific, executable steps)

### Quality Criteria

- Step-by-step CoT rationale for every action
- Task completion verification
- Error recovery traces (not just perfect trajectories)
- Diverse solution paths
- Realistic difficulty distribution

### Filtering Repetitive/Sloppy Traces

From LAM Simulator (Hoang et al., ACL 2025):
1. Answer correctness: compare final response to ground truth
2. Action-level monitoring: verify each tool call
3. Error rectification: include corrected error trajectories
4. Completion without critical errors

### Loop/Repetition Detection

From STeCa (ACL 2025):
- Monitor state-action pairs for repeats
- Detect cycles in state transition graph
- Flag trajectories with >N identical consecutive actions
- Trajectory Deviation Distance quantifies deviation from optimal

---

## Kaggle Multi-Account System

### Existing Profiles (from model_ablation)

11 isolated profiles in `.kaggle_profiles/`:
alexcathe, hiolyjo, holyjow, holykeys, holykeyz10,
ippojoe, jjwhich, josephayanda, josephmayo, josephmayok, makanouchi

Each has isolated `.kaggle/credentials.json` with OAuth tokens.
Isolation via `KAGGLE_CONFIG_DIR` environment variable.

### Concurrency

- 11 accounts × 2 GPU slots each = 22 concurrent training slots
- Each slot runs one 12-hour session
- Batch: 3-5 datasets per session (pre-tokenized binary uploaded as Kaggle dataset)
- Queue manager assigns datasets to available accounts

### Workflow

```
Server-side:
  1. Discover datasets via HF API
  2. Run Stage 1 + 2 (automated metrics, contamination)
  3. For qualified datasets: tokenize → NanoGPT binary format
  4. Upload binary to Kaggle dataset (private)

Kaggle-side (notebook):
  1. Load pre-tokenized binary from /kaggle/input/
  2. Train GPT-2 124M with DDP on 2x T4
  3. Log loss curve every 500 steps
  4. Save checkpoint + loss curve to /kaggle/working/
  5. Upload results to HF Hub

Server-side:
  6. Pull results from HF Hub
  7. Compute relative quality scores
  8. Update leaderboard
```

---

## Tech Stack

### Backend
- Python >=3.11
- FastAPI (API)
- SQLAlchemy + asyncpg (PostgreSQL)
- Pydantic (schemas)
- Typer (CLI)
- huggingface_hub + datasets (discovery)
- datasketch (MinHash dedup)
- tiktoken (tokenization)
- fasttext-langdetect (language detection)
- presidio-analyzer (PII detection)

### Training
- NanoGPT (training framework)
- PyTorch (FP16, DDP)
- Kaggle 2x T4 (compute)

### Frontend (arxiv-style UI)
- Next.js 14 (App Router)
- Tailwind CSS
- EB Garamond serif font (arxiv aesthetic)
- Minimal color palette (red accents, white bg, gray text)
- Table-heavy leaderboard layout
- Score badges: green (≥0.7), yellow (≥0.4), red (<0.4)
- Loss curve visualization
- Category tabs for dataset filtering

### Infrastructure
- PostgreSQL 16
- Docker Compose
- Kaggle notebooks (compute)

---

## Implementation Steps

| Step | What                                           | Files |
|------|------------------------------------------------|-------|
| 1    | Python scaffold + pyproject.toml               | 4     |
| 2    | SQLAlchemy + Pydantic models                   | 3     |
| 3    | HuggingFace discovery scanner + qualifier      | 2     |
| 4    | Stage 1: Automated scorers (7 scorers)         | 8     |
| 5    | Stage 2: Contamination checker                 | 2     |
| 6    | Stage 3: Training pipeline (NanoGPT + T4)      | 4     |
| 7    | Kaggle orchestration (accounts + queue + notebook) | 4  |
| 8    | Runner: orchestrate all stages                 | 1     |
| 9    | FastAPI app + routes                           | 3     |
| 10   | CLI (Typer)                                    | 1     |
| 11   | Kaggle training notebook                       | 1     |
| 12   | Tests                                          | 2     |
| 13   | Docker Compose                                 | 2     |
| 14   | Next.js frontend                               | 8-10  |
| **Total** |                                        | **~45**|

---

## Running the First Batch

1. Discover 50-100 new datasets from HuggingFace (last 30 days, all categories)
2. Qualify → filter to ~30-40 that meet thresholds
3. Stage 1 → automated scores for all
4. Stage 2 → contamination check for all
5. Stage 3 → train GPT-2 124M on each (Kaggle 2x T4, ~12h per session)
6. Stage 4 → populate leaderboard

Each subsequent release: discover new → run pipeline → add to leaderboard.
