# Socbench Kaggle Training Results

This directory keeps small recovered Kaggle training-result artifacts used by
`python -m socbench.training.import_results`.

It intentionally contains JSON loss curves and summary files, not model checkpoints,
Kaggle credentials, generated notebooks, or local logs.

## Contents

- `results/*/loss_curve.json`: recovered historical campaign result files.
- `results_v12/*.json`: recovered v12 campaign result files.

## Recovery Import

From the backend directory:

```powershell
python -m socbench.training.import_results
python -m socbench.training.import_results --include-orphans --create-missing-datasets --apply
```

The importer validates completeness and records each selected artifact path in
`training_runs.model_config.source_artifact`.
