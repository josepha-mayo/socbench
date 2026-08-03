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
- Recovers historical Kaggle GPT-2 proxy-training artifacts into the SQLite leaderboard.
- Keeps pending training candidates visible from live Hugging Face trending and downloads
  scans, with caching so the training page stays responsive.

## Current Verified Data State

The local development database is `backend/socbench.db`. It is runtime state and is ignored
by Git.

As of the latest recovery pass:

- `datasets`: 55
- `leaderboard`: 55
- `training_runs`: 13
- `/api/stats`: returns HTTP 200
- `/api/training-leaderboard?limit=20`: returns 13 trained rows plus pending candidates
- `/training`: renders the recovered training leaderboard in the frontend

The recovered trained rows include `tatsu-lab/alpaca`, `EdinburghNLP/xsum`,
`m-a-p/COIG-CQIA`, `garage-bAInd/Open-Platypus`, `yahma/alpaca-cleaned`,
`teknium/OpenHermes-2.5`, `WizardLMTeam/WizardLM_evol_instruct_70k`,
`LDJnr/Capybara`, `HuggingFaceH4/ultrachat_200k`, and `OpenAssistant/oasst1`.

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
    audit/            Dataset audit pipeline scaffolding
    contamination/    N-gram contamination checker
    discovery/        Hugging Face scanner and qualifier
    evals/            Eval benchmark intelligence
    kaggle/           Kaggle account, notebook, and queue helpers
    scoring/          Automated dataset scorers
    training/         GPT-2 training config, script generator, importer
  tests/              Backend unit tests
frontend/
  src/app/            Next.js App Router UI
kaggle_socbench/
  results*/           Recovered historical Kaggle loss-curve artifacts
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

## Training Result Recovery

Historical Kaggle training artifacts are kept under `kaggle_socbench/results*`.
They are small JSON provenance files, not model checkpoints.

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
`model_config.source_artifact`, upserts `training_runs`, recomputes `training_score`,
and preserves the combined score formula:

```text
combined_score = 0.9 * auto_score + 0.1 * training_score
```

## Kaggle Training Workflow

Socbench has a safe offline bundle step for new Kaggle runs. The bundle step does
not push anything; it creates reviewable files for the prepared Kaggle dataset and
training kernel.

List configured Kaggle profiles without printing credentials:

```powershell
cd C:\Users\USER\.vscode\vibe\backend
python -m socbench kaggle accounts
```

Prepare a dataset binary first:

```powershell
python -c "import asyncio; from socbench.training.data_prep import prepare_dataset_binary; asyncio.run(prepare_dataset_binary('Salesforce/wikitext', 'kaggle_prepared/salesforce-wikitext', max_samples=100000))"
```

Create the local Kaggle bundle:

```powershell
python -m socbench kaggle bundle Salesforce/wikitext kaggle_prepared/salesforce-wikitext --output-root kaggle_train --account holykeys
```

The command prints exact push commands like:

```powershell
$env:KAGGLE_CONFIG_DIR = "C:\Users\USER\.vscode\model_ablation\.kaggle_profiles\holykeys\.kaggle"
kaggle datasets create -p "kaggle_train\salesforce-wikitext\data" --dir-mode zip
kaggle kernels push -p "kaggle_train\salesforce-wikitext\kernel"
```

If the Kaggle dataset already exists, use `kaggle datasets version` instead of
`kaggle datasets create`.

To upload/version the prepared Kaggle dataset and push the GPU kernel in one
step, use:

```powershell
python -m socbench kaggle launch Salesforce/wikitext kaggle_prepared/salesforce-wikitext --output-root kaggle_train --account holykeys --tokens 5000000
```

`launch` reuses the saved Kaggle OAuth profile from
`C:\Users\USER\.vscode\model_ablation\.kaggle_profiles\<account>\.kaggle\credentials.json`.
It mounts those credentials into a temporary Kaggle home for each subprocess so
the global login state is not changed and secrets are not written to manifests.
Each launch writes:

```text
backend/kaggle_train/<dataset-slug>/launch-manifest.json
```

Review the generated `data/` and `kernel/` directories before starting long GPU
training. The notebook expects Kaggle to mount data at:

```text
/kaggle/input/<dataset-slug>/train.bin
```

The generated notebook is designed to write `train.log`, `train.err`,
`loss_curve.json`, and `eval_results.json` under `/kaggle/working`, then copy
compact result files to the output root for download/import.

### Current launched batch

On August 3, 2026, the first new 1M-token Kaggle smoke/proxy runs were launched,
reached Kaggle `COMPLETE` status, downloaded result artifacts, and were imported
into the local training leaderboard:

```text
Salesforce/wikitext                         holykeys/socbench-train-salesfoc13c
HuggingFaceCode/stack-v3-train              alexcathe/socbench-train-huggingd774
r0b0tlab/qwen3.8-max-distillation-50k       ippojoe/socbench-train-r0b0tlabed1
```

Use `kaggle kernels status <owner>/<slug>` with the matching profile credentials
to poll them, then `kaggle kernels output <owner>/<slug> -p <output-dir> -o` to
pull artifacts before running the importer. The imported smoke runs use the
script-kernel path, force CPU on Kaggle's current P100 image, keep checkpoints
under `/kaggle/temp`, and publish compact JSON/log artifacts under
`/kaggle/working`.

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
docker compose up --build
```

Services:

- Postgres: `localhost:5432`
- API: `localhost:8000`
- Frontend: `localhost:3000`

## Production Notes

- Do not commit `backend/socbench.db`; it is local runtime state.
- Commit source, tests, docs, and small recovered provenance JSONs.
- Keep Kaggle credentials outside the repository.
- Set `DATABASE_URL` for production Postgres.
- Set `SOCBENCH_API_URL` for frontend deployments.
- Run backend tests, frontend lint/typecheck/build, and npm audit before release.

## Known Gaps

These are not finished yet:

- Audit decontamination now has a local n-gram eval-bank index for `.json`, `.jsonl`,
  `.txt`, and `.md` benchmark files, plus a bounded Hugging Face benchmark fallback.
  Full MinHash/LSH scale-out and curated eval-bank coverage are still future work.
- Kaggle bundle creation and launch orchestration are implemented and tested, including
  OAuth profile isolation, dataset create/version fallback, kernel push, and launch
  manifests. Output collection still needs a hardened importer path for newly launched
  notebooks across Kaggle CLI versions.
- Eval-proof export exists for scored datasets, but proofs should be regenerated after
  every catalog curation/training import batch.
- `teknium/OpenHermes-2.5` and `LDJnr/Capybara` have real recovered training scores,
  but their automated scoring rows were restored minimally because Hugging Face sample
  fetching failed during recovery.

So: the restored training leaderboard is real and displaying, new Kaggle training
runs can be launched from the repo, and the local catalog has been curated. The
remaining work is mostly operational hardening around post-run artifact import,
proof regeneration after each batch, and larger-scale decontamination coverage.
