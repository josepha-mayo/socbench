# Socbench Training Results

This directory keeps small validated training-result artifacts used by
`python -m socbench.training.import_results`.

It intentionally contains JSON loss curves and summary files, not model checkpoints,
compute credentials, generated runners, or local logs.

## Contents

- `validated/*_v*/result.json`: one canonical validated artifact per selected
  dataset/campaign.
- `validated/*_v*/eval_results.json`: a compact evaluation proof, including explicit
  negative outcomes when a completed run diverged.

The v23 C4 and Hermes artifacts are completed real 1B-token-target runs with verified
two-device distributed execution. Their validation curves diverged, so they are kept
as negative evidence rather than presented as successful training improvements.

## Recovery Import

From the backend directory:

```powershell
python -m socbench.training.import_results
python -m socbench.training.import_results --include-orphans --create-missing-datasets --apply
```

The importer validates completeness, records each selected artifact path in
`training_runs.model_config.source_artifact`, and globally normalizes training scores
across all current complete runs after each applied import.
