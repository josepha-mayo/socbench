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

- 55 catalog datasets: 53 complete automated assessments and 2 training-only records
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
    dataset_intelligence.py  Cache, readiness, and comparison policy
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

To verify the generated production server locally:

```powershell
cd frontend
npm run build
npm run start -- --hostname 127.0.0.1 --port 3000
```

`npm run start` prepares the static assets inside Next.js standalone output and launches
the generated server. It intentionally fails when a production build is missing.

Portable Windows launchers are also available after dependencies are installed:

```powershell
.\run_api.bat
.\run_fe.bat
```

## Cache-First Dataset Decisions

The CLI checks the local Socbench database by exact, case-sensitive Hugging Face dataset
ID before it makes a scoring request. A cache entry is reusable only when it has all seven
automated dimensions, the aggregate contamination rate, at least one per-benchmark
contamination record, an automated score, and a scoring timestamp. An incomplete row is
never treated as training-ready.

Default behavior:

1. Return complete evidence from the database without a network request.
2. If complete evidence is absent, fetch bounded metadata and samples, run every scoring
   suite, and persist the result in one transaction.
3. Reject a live result instead of caching it when any required dimension or verified
   per-benchmark contamination evidence is missing.

Use `--refresh` to deliberately replace complete cached evidence. Use `--cached-only` to
forbid network access and fail with exit code `2` when complete evidence is unavailable.
The two flags are mutually exclusive. `--token` or `HF_TOKEN` is optional and is needed
only when the source dataset requires authentication.

Get a direct training-readiness decision:

```powershell
cd backend
python -m socbench readiness Salesforce/wikitext --cached-only
```

Score only on a cache miss:

```powershell
python -m socbench score owner/dataset --sample-size 1000
```

Deliberately refresh an existing assessment:

```powershell
python -m socbench score owner/dataset --sample-size 10000 --refresh
```

Return machine-readable evidence. The banner is written to stderr, so stdout remains valid
JSON; `--no-banner` suppresses it entirely:

```powershell
python -m socbench --no-banner readiness owner/dataset --format json > readiness.json
```

Compare two to twenty datasets. Missing assessments are scored sequentially unless
`--cached-only` is supplied:

```powershell
python -m socbench compare `
  HuggingFaceFW/fineweb-edu `
  allenai/c4 `
  Salesforce/wikitext `
  --cached-only
```

Comparison order is `PROCEED`, then `REVIEW`, then `DO NOT PROCEED`; ties are ordered by
automated score. Missing evidence never ranks as `PROCEED`.

### Readiness Policy

The automated score is the arithmetic mean of quality, diversity, utility,
documentation, popularity, freshness, and PII safety. It is useful for comparison, but
hard safety and evidence gates take precedence.

`DO NOT PROCEED` is returned when any blocker is present:

- quality is below `40%`
- utility is below `40%`
- PII safety is below `85%`
- maximum benchmark overlap is above `5%`
- estimated repetition is above `50%`
- the latest validated proxy-training outcome is `diverged` or `regressed`

`REVIEW` is returned when evidence is incomplete or any review condition is present:

- automated score is below `65%`
- quality is below `60%`
- diversity is below `15%`
- documentation is below `50%`
- PII safety is below `95%`
- maximum benchmark overlap is above `1%`
- estimated repetition is above `25%`
- the license is missing or ambiguous
- the cached assessment is older than `180` days
- the latest proxy-training outcome is `stable` or `insufficient_evidence`

`PROCEED` means the stored evidence is complete and no gate fired. It is permission to run
the bounded seven-stage audit, not permission to train raw data without inspection.

Use the HTTP API for an explicit live score when database persistence is not needed:

```powershell
Invoke-RestMethod -Method Post `
  "http://localhost:8000/api/datasets/Salesforce/wikitext/score?sample_size=1000"
```

The API bounds `sample_size` to 100 through 100,000 rows.

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

## Semantic Benchmark Audit

`benchmark-audit` evaluates the internal quality of conversational-agent benchmarks. It
measures description-to-expected-behavior alignment, policy-to-expected-behavior
alignment, restrictive-policy violations per task, policy-violation coverage, and an
aggregate semantic-quality score. This audits benchmark construction; it does not score a
training dataset or claim model performance.

The metric family is based on Koren, Bar-Haim, and Goldsteen,
[Benchmarking the Benchmarks](https://arxiv.org/abs/2608.06329). Socbench's local engine is
a deterministic, reference-free adaptation for reproducible offline screening, not a claim
of reproducing the paper's LLM-judge results.

The deterministic local semantic engine is always available and requires no API key. A
remote OpenAI-compatible semantic judge is optional and only runs when it is explicitly
enabled and fully configured. `--local-only` prevents remote use even when configured.

Canonical input:

```json
{
  "name": "Customer support policy benchmark",
  "policy_items": [
    "The agent must verify identity before changing a delivery address.",
    "The agent must not issue a refund after the refund window closes."
  ],
  "coverage_threshold": 3,
  "tasks": [
    {
      "id": "address-change-without-verification",
      "description": "A customer asks to change an address before identity verification.",
      "initial_state": {"identity_verified": false},
      "expected_behavior": "Refuse the change until the customer verifies identity."
    }
  ]
}
```

`policy` may be supplied as one prose string instead of `policy_items`. Inputs are bounded
to 200 tasks and 200 policy items. Run locally and save the full result:

```powershell
cd backend
python -m socbench benchmark-audit .\benchmark.json `
  --local-only `
  --output .\audit_outputs\benchmark-audit.json
```

Send the same canonical JSON to the API:

```powershell
$body = Get-Content .\benchmark.json -Raw
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/api/evals/benchmark-audit" `
  -ContentType "application/json" `
  -Body $body
```

Optional remote enhancement:

```powershell
$env:SOCBENCH_SEMANTIC_REMOTE_ENABLED = "1"
$env:SOCBENCH_SEMANTIC_API_URL = "https://judge.example.com/v1/chat/completions"
$env:SOCBENCH_SEMANTIC_MODEL = "judge-model"
$env:SOCBENCH_SEMANTIC_API_KEY = "..."
python -m socbench benchmark-audit .\benchmark.json --output .\audit.json
```

The URL and model do not imply a particular provider. The API key is never required for
local scoring, is read only from the process environment, and must not be committed. If
remote enhancement is enabled but unavailable or incomplete, Socbench records a warning
and retains local results.

## CLI Reference

Run `python -m socbench --help` or `python -m socbench <command> --help` for generated
help. The startup mark is emitted on stderr. Set `SOCBENCH_NO_BANNER=1` or pass the global
`--no-banner` option before the command to hide it.

Dataset scoring and decisions:

- `assess DATASET_ID`: cache-first full dimension table plus readiness decision.
- `score DATASET_ID`: cache-first scoring entry point; functionally equivalent to
  `assess`, with wording suited to automation.
- `readiness DATASET_ID`: concise `PROCEED`, `REVIEW`, or `DO NOT PROCEED` decision with
  explicit reasons and next step.
- `compare DATASET_ID...`: compare 2 to 20 unique datasets by readiness and stored score.
- `cache-status DATASET_ID`: inspect exact-ID evidence without any scoring request.

`assess`, `score`, and `readiness` accept `--sample-size 100..100000`, `--refresh`,
`--cached-only`, optional `--token`/`HF_TOKEN`, and `--format table|json`. `compare`
accepts the same options and applies the sample bound to each cache miss. `cache-status`
accepts `--format table|json` and never uses the network.

Discovery and interpretation:

- `discover`: find public datasets. Options are `--search`, `--limit` (default `50`), and
  optional `--days`.
- `classify DATASET_ID`: fetch metadata and show the hierarchical training-purpose
  category and its metrics.
- `qualify DATASET_ID`: test discovery eligibility; optional `--downloads` and `--likes`
  seed the qualification check.
- `provenance DATASET_ID`: print locally recorded model and paper provenance.
- `recommendations DATASET_ID`: generate category-use recommendations from cache-first
  evidence. It accepts the same sample, refresh, cached-only, and token options as `score`.

Audit, ranking, and proof workflows:

- `audit DATASET_ID`: run the seven-stage cleaning pipeline. Options include
  `--output-dir`, `--max-rows`, `--output-size`, `--text-key`, `--eval-bank-dir`,
  `--min-tokens`, and `--max-tokens`.
- `leaderboard`: query stored records only. Options are `--top 1..500`, exact
  `--category`, `--sort combined|auto|quality|diversity|utility|training`, and
  `--format table|json`.
- `benchmark-audit INPUT_PATH`: audit semantic benchmark construction. Use `--local-only`
  to prohibit the optional remote judge and `--output` for the complete JSON result.
- `export-proofs`: export local dataset proof JSON. Options are `--output-dir` and
  optional `--limit`.

Service workflow:

- `serve`: start the development API server. Options are `--host` (default `0.0.0.0`)
  and `--port` (default `8000`). Use the documented Uvicorn or container entry point for
  production rather than the development reloader.

Exit codes are `0` for success, `1` when `cache-status` finds no exact dataset record, and
`2` for invalid arguments, unavailable required cache evidence, or failed live scoring.

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
