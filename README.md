# Socbench

Socbench is scientific dataset intelligence for model builders. It is designed to answer:

> Which datasets should I train on, and why?

It is not just a model leaderboard. Socbench scores datasets across quality, diversity,
utility, documentation, popularity, freshness, PII safety, contamination risk, and GPT-2
124M proxy-training impact.

## What It Does

- Discovers and classifies Hugging Face datasets.
- Scores datasets with automated quality, format, token, PII, code, and metadata checks.
- Tracks known dataset provenance, including models and papers associated with datasets.
- Exposes a FastAPI backend for leaderboards, dataset details, discovery, eval analysis,
  trending datasets, and evaluation requests.
- Serves a Next.js frontend with leaderboard, training, eval, trending, discover, and
  dataset detail views.
- Imports validated GPT-2 proxy-training results into the SQLite leaderboard.
- Keeps pending training candidates visible from live Hugging Face trending and downloads
  scans, with caching so the training page stays responsive.

## Current Verified Data State

The local development database is `backend/socbench.db`. It is runtime state and is ignored
by Git.

As of the latest recovery pass:

- `datasets`: 55
- `leaderboard`: 55
- `training_runs`: 16
- `/api/stats`: returns HTTP 200
- `/api/training-leaderboard?limit=20`: returns 16 trained rows plus pending candidates
- `/training`: renders the recovered training leaderboard in the frontend

The recovered trained rows include `tatsu-lab/alpaca`, `EdinburghNLP/xsum`,
`m-a-p/COIG-CQIA`, `garage-bAInd/Open-Platypus`, `yahma/alpaca-cleaned`,
`teknium/OpenHermes-2.5`, `WizardLMTeam/WizardLM_evol_instruct_70k`,
`LDJnr/Capybara`, `HuggingFaceH4/ultrachat_200k`, and `OpenAssistant/oasst1`.
The recovered proxy results also include `Salesforce/wikitext`,
`HuggingFaceCode/stack-v3-train`, `r0b0tlab/qwen3.8-max-distillation-50k`,
`allenai/c4`, `HuggingFaceFW/fineweb-edu`, and
`NousResearch/hermes-function-calling-v1`.

The latest validated real-run artifacts are the v23 results for `allenai/c4`,
`HuggingFaceCode/stack-v3-train`, `HuggingFaceFW/fineweb-edu`, and
`NousResearch/hermes-function-calling-v1`, plus
`r0b0tlab/qwen3.8-max-distillation-50k`. Each reached 999,817,216 observed tokens
with two Tesla T4 devices and distributed world size 2. All five are retained as
negative results because their validation curves ultimately diverged. Incomplete
campaign runs are not published as results.

Pending rows are dynamic: the API pulls the current top Hugging Face trending and
most-downloaded datasets, removes anything already trained, and marks the rest as
`pending`. Those are the datasets left to train next. Check them with:

```powershell
cd C:\Users\USER\.vscode\vibe
python -c "import requests; data=requests.get('http://localhost:8000/api/training-leaderboard?limit=100', timeout=20).json(); print([x['hf_id'] for x in data if x['status']=='pending'])"
```

## Repository Layout

```text
backend/
  socbench/
    api/              FastAPI app and routes
    audit/            Seven-stage dataset cleaning and decontamination audit
    contamination/    N-gram contamination checker
    discovery/        Hugging Face scanner and qualifier
    evals/            Eval benchmark intelligence
    scoring/          Automated dataset scorers
    training/         GPT-2 training config, script generator, importer
  tests/              Backend unit tests
frontend/
  src/app/            Next.js App Router UI
training_results/
  validated/          Canonical loss-curve and evaluation artifacts
PLAN.md               Product/architecture notes
docker-compose.yml    Local Postgres/API/frontend composition
```

## Local Setup

### Backend

```powershell
cd C:\Users\USER\.vscode\vibe\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m uvicorn socbench.api.app:app --host 127.0.0.1 --port 8000
```

The backend defaults to SQLite at `backend/socbench.db` when launched from the
`backend` directory. For Postgres, set:

```powershell
$env:DATABASE_URL = "postgresql+asyncpg://socbench:socbench@localhost:5432/socbench"
```

### Frontend

```powershell
cd C:\Users\USER\.vscode\vibe\frontend
npm install
npm run dev
```

The frontend rewrites `/api/*` to the backend. Configure the target with either:

```powershell
$env:SOCBENCH_API_URL = "http://localhost:8000"
```

or:

```powershell
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
```

Then open:

```text
http://localhost:3000
http://localhost:3000/training
```

## Verification

Run the backend checks:

```powershell
cd C:\Users\USER\.vscode\vibe\backend
python -m compileall socbench
python -m pytest
```

Run the frontend checks:

```powershell
cd C:\Users\USER\.vscode\vibe\frontend
npm audit --audit-level=high
npm run lint
npx tsc --noEmit
npm run build
```

Check live API health:

```powershell
python -c "import requests; print(requests.get('http://localhost:8000/api/stats', timeout=10).json())"
python -c "import requests; print(len(requests.get('http://localhost:8000/api/training-leaderboard?limit=20', timeout=20).json()))"
python -c "import requests; print(len(requests.get('http://localhost:3000/api/training-leaderboard?limit=20', timeout=20).json()))"
```

## Scoring A Dataset

Anyone can run a Socbench score for a Hugging Face dataset ID from the CLI or API.
The command fetches live metadata/samples, classifies the dataset, computes the
multi-dimension score, checks contamination, and prints the result.

CLI:

```powershell
cd C:\Users\USER\.vscode\vibe\backend
python -m socbench score Salesforce/wikitext --sample-size 1000
```

Equivalent legacy command:

```powershell
python -m socbench assess Salesforce/wikitext --sample-size 1000
```

Live API:

```powershell
Invoke-RestMethod -Method Post "http://localhost:8000/api/datasets/Salesforce/wikitext/score?sample_size=1000"
```

Open a dataset page or score on demand through the frontend:

```text
http://localhost:3000/datasets/Salesforce/wikitext
```

## Deep Dataset Audit

Use `audit` when the objective is a trainable, cleaned dataset rather than a
leaderboard score. It runs all seven audit stages and writes both the balanced
JSONL output and an auditable JSON summary:

1. license policy
2. language detection
3. code syntax validation when applicable
4. evaluation-bank decontamination
5. exact and near-duplicate removal
6. token-length filtering
7. seeded per-language water-filling rebalancing

The default audit uses the built-in evaluation-bank benchmarks. Give it a local
`--eval-bank-dir` to add or replace those checks with your own JSON/JSONL eval
corpora.

```powershell
cd C:\Users\USER\.vscode\vibe\backend
python -m socbench audit Salesforce/wikitext --output-dir audit_outputs/wikitext --max-rows 100000
```

For a bounded local verification pass:

```powershell
python -m socbench audit Salesforce/wikitext --output-dir audit_outputs/wikitext-smoke --max-rows 1000 --output-size 500
```

The command creates:

```text
audit_outputs/wikitext/
  audit_balanced.jsonl
  audit_summary.json
```

## Eval-Proof Export

Generate reproducible proof JSON files for every dataset currently in the database:

```powershell
cd C:\Users\USER\.vscode\vibe\backend
python -m socbench export-proofs --output-dir ..\eval-proof
```

This writes:

```text
eval-proof/
  manifest.json
  <org>/<dataset>/dataset.json
```

Each `dataset.json` contains the dataset metadata, leaderboard dimensions,
individual scorer details, contamination rows, and latest training-run provenance
when available. The export directory is local runtime output and is ignored by Git.

## Training Result Import

Validated external training results are compact JSON provenance files under
`training_results/validated`, not model checkpoints. One canonical artifact per
dataset/campaign is kept. Account credentials, launch configuration, logs, and
generated runner files stay outside this repository.

Dry-run an import plan:

```powershell
cd C:\Users\USER\.vscode\vibe\backend
python -m socbench.training.import_results
```

Apply recovered results:

```powershell
python -m socbench.training.import_results --include-orphans --create-missing-datasets --apply
```

The importer is idempotent. It validates artifact completeness, keeps provenance in
`model_config.source_artifact`, upserts `training_runs`, and globally recomputes
`training_score` across every current complete run so incremental imports cannot leave
mixed normalization states. It preserves the combined score formula:

```text
combined_score = 0.9 * auto_score + 0.1 * training_score
```

## External Training Result Recovery

Socbench publishes only validated training results and their proof exports. GPU
accounts, kernel configuration, generated training files, and run logs are
local operational concerns and are excluded from the product repository.

Import the compact result artifacts and regenerate public evidence with:

```powershell
cd C:\Users\USER\.vscode\vibe\backend
python -m socbench.training.import_results --include-orphans --create-missing-datasets --apply
python -m socbench export-proofs --output-dir ..\eval-proof
```

The importer validates loss curves and evaluation metrics, records provenance in
`training_runs`, recomputes the training and combined leaderboard scores, and
the proof export contains the latest verified training evidence for each dataset.

## Catalog Curation

The canonical seed catalog is intentionally training-data first. Trained rows are
protected when curating the live SQLite database. The August 3, 2026 cleanup
dropped five untrained low-signal rows from the local runtime catalog:

```text
HuggingFaceH4/llava-instruct-mix-vsft
LLM-LAT/harmful-dataset
Skywork/Skywork-Reward-Preference-80K-v0.2
jondurbin/airoboros-2.2.1
princeton-nlp/llama3-ultrafeedback-armorm
```

They were replaced with stronger live candidates:

```text
allenai/c4
HuggingFaceCode/stack-v3-train
Qyrou/reasoning-corpus-4K-5M-v1
XYZAILab/XYZ-Aquila-SFT
r0b0tlab/qwen3.8-max-distillation-50k
```

Before live catalog mutation, back up `backend/socbench.db`; the cleanup created
`backend/socbench.before_catalog_swap_20260803T103614Z.db`.

## Docker Compose

```powershell
cd C:\Users\USER\.vscode\vibe
Copy-Item .env.example .env
# Replace both password placeholders. URL-encode reserved characters in DATABASE_URL.
docker compose up --build
```

Services:

- Postgres is private to the Compose network.
- API: `http://localhost:8000`
- Frontend: `http://localhost:3000`

Readiness checks:

```powershell
Invoke-RestMethod http://localhost:8000/healthz
Invoke-RestMethod http://localhost:8000/readyz
```

The production images use non-root users, the frontend runs the standalone Next.js
server, and the API waits for a healthy database. Set `CORS_ORIGINS` to browser
origins allowed to call the API and `TRUSTED_HOSTS` to the API host names. Both are
comma-separated and production defaults are fail-closed.

## Production Notes

- Do not commit `backend/socbench.db`; it is local runtime state.
- Commit source, tests, docs, and small recovered provenance JSONs.
- Keep all external compute credentials and operations outside the repository.
- Set `DATABASE_URL` for production Postgres.
- Set `SOCBENCH_API_URL` for frontend deployments.
- Set `APP_ENV=production`, `TRUSTED_HOSTS`, and the narrowest practical `CORS_ORIGINS`.
- Use `/healthz` for liveness and `/readyz` for database-backed readiness.
- Run backend tests, frontend lint/typecheck/build, and npm audit before release.

## Known Gaps

These are not finished yet:

- Audit decontamination now has a local n-gram eval-bank index for `.json`, `.jsonl`,
  `.txt`, and `.md` benchmark files, plus a bounded Hugging Face benchmark fallback.
  Full MinHash/LSH scale-out and curated eval-bank coverage are still future work.
- Eval-proof export exists for scored datasets, but proofs should be regenerated after
  every catalog curation/training import batch.
- `teknium/OpenHermes-2.5` and `LDJnr/Capybara` have real recovered training scores,
  but their automated scoring rows were restored minimally because Hugging Face sample
  fetching failed during recovery.

The restored training leaderboard is displaying and the local catalog is curated.
Remaining research work is larger-scale decontamination coverage and regenerating
proof exports whenever a validated training-result batch is imported.
