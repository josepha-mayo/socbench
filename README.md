# Socbench

Socbench is dataset intelligence for model builders. It examines training data before
compute is committed, scores it across independent quality dimensions, audits and cleans
raw corpora, records contamination evidence, and publishes validated proxy-training
outcomes.

The product consists of a FastAPI service, a Next.js interface, a Python CLI, a canonical
55-dataset scored catalog, and compact validated result summaries with evaluation proofs. A
clean database bootstraps from tracked JSON, so production does not depend on a developer's
SQLite file.

## Current Evidence

- 55 scored catalog datasets
- 55 leaderboard records
- 16 validated GPT-2 124M proxy-training records
- 10 improved runs and 6 divergent runs
- Six completed dual-T4, approximately 1B-token negative results retained as evidence
- SQLite for local development and PostgreSQL for production

Training score is not a cross-run min-max ranking. It is the positive final validation-loss
reduction for that dataset:

```text
relative_improvement = (initial_val_loss - final_val_loss) / initial_val_loss
training_score = relative_improvement when outcome == improved, otherwise 0
combined_score = 0.9 * auto_score + 0.1 * training_score
```

Outcomes are `improved`, `stable`, `regressed`, `diverged`, or
`insufficient_evidence`. A transient best checkpoint never turns a regressed final run into
a positive result.

## Capabilities

- Discover and classify Hugging Face datasets by training purpose.
- Score quality, diversity, utility, documentation, popularity, freshness, PII safety,
  contamination, and repetition.
- Run a seven-stage deep audit that emits cleaned JSONL and a machine-readable report.
- Track dataset provenance and evaluation benchmark risk.
- Accept public or private evaluation requests through the API and UI.
- Import and validate compact training results, including exact-run evidence hashes.
- Reconstruct a clean database from the canonical catalog and validated results.
- Expose dataset, leaderboard, training, discovery, evaluation, and health APIs.

## Repository Layout

```text
backend/
  socbench/
    api/                 FastAPI application and routes
    audit/               Deep cleaning and decontamination pipeline
    contamination/       Benchmark overlap checks
    discovery/           Hugging Face discovery and qualification
    evals/               Evaluation benchmark intelligence
    scoring/             Automated scoring dimensions
    training/            Data preparation, guarded trainer, outcomes, importer
    bootstrap.py         Canonical clean-database bootstrap and catalog export
  tests/                 Backend test suite
catalog/catalog.json     Canonical scored catalog, without runtime IDs
frontend/src/            Next.js application
training_results/
  validated/             Compact canonical result summaries and evaluation proofs
docker-compose.yml       PostgreSQL, API, and frontend production-like stack
```

Only source, tests, documentation, canonical catalog data, validated result summaries, and
evaluation proofs are versioned. Runtime databases, credentials, raw logs, checkpoints,
caches, and downloads are ignored.

## Local Setup

Requirements:

- Python 3.11 or newer
- Node.js 20 or newer
- npm

Install and start the API:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m uvicorn socbench.api.app:app --host 127.0.0.1 --port 8000
```

On first startup, an empty local `backend/socbench.db` is populated from
`catalog/catalog.json` and `training_results/validated`. Existing databases are left intact.

In another terminal, install and start the UI:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. The UI rewrites `/api/*` to
`SOCBENCH_API_URL`, which defaults to `http://localhost:8000`.

Portable Windows launchers are also available after dependencies are installed:

```powershell
.\run_api.bat
.\run_fe.bat
```

## Score Any Dataset

Score a public Hugging Face dataset from the CLI:

```powershell
cd backend
python -m socbench score Salesforce/wikitext --sample-size 1000
```

Use the HTTP API:

```powershell
Invoke-RestMethod -Method Post `
  "http://localhost:8000/api/datasets/Salesforce/wikitext/score?sample_size=1000"
```

The live scoring path fetches metadata and bounded samples, classifies the dataset, runs
the scoring suites, checks contamination, and returns structured JSON. `sample_size` is
bounded by the API to 100 through 100,000 rows.

## Deep Dataset Audit

The audit command is for producing a cleaned training corpus, not merely a leaderboard
score. Its stages are:

1. License policy enforcement
2. Language detection
3. Code syntax validation when applicable
4. Evaluation-bank decontamination
5. Exact and near-duplicate removal
6. Token-length filtering
7. Seeded per-language water-filling rebalance

Run a full bounded audit:

```powershell
cd backend
python -m socbench audit Salesforce/wikitext `
  --output-dir audit_outputs/wikitext `
  --max-rows 100000
```

Run a small verification audit:

```powershell
python -m socbench audit Salesforce/wikitext `
  --output-dir audit_outputs/wikitext-check `
  --max-rows 1000 `
  --output-size 500
```

Outputs:

```text
audit_outputs/wikitext/
  audit_balanced.jsonl
  audit_summary.json
```

Pass `--eval-bank-dir` to use local `.json`, `.jsonl`, `.txt`, or `.md` evaluation
corpora in addition to the built-in bounded benchmark sources.

## API

Important routes:

```text
GET  /healthz
GET  /readyz
GET  /api/stats
GET  /api/leaderboard
GET  /api/training-leaderboard
GET  /api/datasets/{owner}/{dataset}
POST /api/datasets/{owner}/{dataset}/score
GET  /api/discover
POST /api/request-evaluation
```

`/healthz` is process liveness. `/readyz` verifies both database connectivity and a
non-empty canonical catalog. Interactive OpenAPI docs are available at `/docs` in
development and disabled in production.

## Training Methodology

Socbench generates a standalone GPT-2 124M DDP trainer. The guarded defaults are:

- learning rate `3e-4`, AdamW epsilon `1e-8`
- per-device batch `8`, gradient accumulation `64`
- FP16 autocast with gradient scaling
- `torch.compile` disabled by default
- validation under inference mode
- finite-gradient checks and synchronized DDP validation
- maximum planned dataset repetition `8x` unless explicitly reviewed
- calibration at 100 optimizer iterations
- minimum calibration improvement `0.5%`
- abort after two validation checks more than `5%` above baseline

Generated training scripts stop after calibration by default. A full token-budget run
requires explicit authorization in the execution environment:

```powershell
$env:SOCBENCH_FULL_RUN = "1"
```

Excessive dataset repetition is a separate explicit override:

```powershell
$env:SOCBENCH_ALLOW_EXCESSIVE_REPETITION = "1"
```

Do not set either flag until the calibration result and token/repetition math have been
reviewed. Only compact validated result summaries and evaluation proofs belong in the
repository; runtime credentials, execution code, and raw logs remain local.

## Validated Result Import

Canonical artifacts live at:

```text
training_results/validated/<dataset>_v<campaign>/result.json
training_results/validated/<dataset>_v<campaign>/eval_results.json
```

Preview an import:

```powershell
cd backend
python -m socbench.training.import_results --include-orphans
```

Apply it to the local SQLite database:

```powershell
python -m socbench.training.import_results `
  --include-orphans `
  --create-missing-datasets `
  --apply
```

The importer accepts canonical files only by default. It validates curve completeness,
final metrics, declared outcomes, real-run proof files, and SHA-256 digest format. It is
idempotent and records the best checkpoint separately from completed steps.

## Catalog And Proofs

Export the current non-training catalog after an intentional curation change:

```powershell
cd backend
python -m socbench.bootstrap export --output ..\catalog\catalog.json
```

Generate local per-dataset evaluation proofs:

```powershell
python -m socbench export-proofs --output-dir ..\eval-proof
```

The catalog is tracked and bootstraps production. `eval-proof/` is generated runtime
output and ignored; compact canonical training proofs remain under `training_results`.

## Docker Compose

Create local production settings and replace both password placeholders:

```powershell
Copy-Item .env.example .env
docker compose config
docker compose up --build
```

Services:

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- PostgreSQL: private to the Compose network

The API image includes the canonical catalog and validated result/proof data. A new
PostgreSQL volume is populated automatically and idempotently. Both application images
run as non-root users and expose health checks.

Production environment variables:

```text
APP_ENV=production
DATABASE_URL=postgresql+asyncpg://...
TRUSTED_HOSTS=api.example.com
CORS_ORIGINS=https://app.example.com
SOCBENCH_CATALOG_PATH=/app/catalog/catalog.json
SOCBENCH_TRAINING_RESULTS_ROOT=/app
HF_TOKEN=optional-private-dataset-token
```

`TRUSTED_HOSTS` is required in production. Wildcard production CORS is rejected.

## Verification

Backend:

```powershell
cd backend
python -m compileall socbench
python -m ruff check socbench tests
python -m pytest
```

Frontend:

```powershell
cd frontend
npm audit --audit-level=high
npm run lint
npx tsc --noEmit
npm run build
```

Live end-to-end checks:

```powershell
Invoke-RestMethod http://localhost:8000/healthz
Invoke-RestMethod http://localhost:8000/readyz
Invoke-RestMethod http://localhost:8000/api/stats
Invoke-RestMethod "http://localhost:8000/api/training-leaderboard?limit=100"
Invoke-WebRequest http://localhost:3000/training -UseBasicParsing
```

## Production Boundary

Commit source, tests, documentation, the canonical catalog, compact validated result
summaries, and evaluation proofs. Keep local databases, model checkpoints, credentials,
execution code, raw logs, downloads, and temporary audit corpora outside the repository.

The six approximately 1B-token dual-T4 campaign results are negative evidence, not
successful model improvements. They remain visible, explicitly labeled `diverged`, and
score zero. No additional GPU run should be launched without a reviewed calibration and
explicit authorization.
