# Socbench Training Results

This directory keeps small validated training-result artifacts used by
`python -m socbench.training.import_results`.

It intentionally contains JSON loss curves and summary files, not model checkpoints,
compute credentials, generated runners, or local logs.

## Contents

- `validated/*_v*/result.json`: one canonical validated artifact per selected
  dataset/campaign.

## Recovery Import

From the backend directory:

```powershell
python -m socbench.training.import_results
python -m socbench.training.import_results --include-orphans --create-missing-datasets --apply
```

The importer validates completeness and records each selected artifact path in
`training_runs.model_config.source_artifact`.
