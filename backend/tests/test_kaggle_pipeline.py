"""Tests for Kaggle training bundle creation."""

from __future__ import annotations

import json
from pathlib import Path

from socbench.kaggle.pipeline import build_dataset_metadata, create_training_bundle


def test_build_dataset_metadata_uses_owner_and_safe_slug():
    metadata = build_dataset_metadata("Org/My Dataset!", "holykeys")

    assert metadata["id"] == "holykeys/org-my-dataset"
    assert metadata["title"] == "Socbench prepared data - Org/My Dataset!"
    assert metadata["licenses"] == [{"name": "CC0-1.0"}]


def test_create_training_bundle_writes_data_and_kernel_files(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "train.bin").write_bytes(b"\x01\x00\x02\x00")
    (prepared / "metadata.json").write_text(
        json.dumps({"dataset_id": "Org/My Dataset!", "total_tokens": 2}),
        encoding="utf-8",
    )

    bundle = create_training_bundle(
        dataset_id="Org/My Dataset!",
        prepared_data_dir=prepared,
        output_root=tmp_path / "bundles",
        kaggle_owner="holykeys",
        tokens=1234,
    )

    assert bundle.kaggle_dataset_ref == "holykeys/org-my-dataset"
    assert bundle.train_bin.exists()
    assert bundle.train_bin.read_bytes() == b"\x01\x00\x02\x00"

    dataset_metadata = json.loads((bundle.data_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert dataset_metadata["id"] == "holykeys/org-my-dataset"
    assert (bundle.data_dir / "metadata.json").exists()

    kernel_metadata = json.loads((bundle.kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert kernel_metadata["dataset_sources"] == ["holykeys/org-my-dataset"]

    notebook = json.loads((bundle.kernel_dir / "notebook.ipynb").read_text(encoding="utf-8"))
    verify_source = "".join(notebook["cells"][2]["source"])
    assert "/kaggle/input/org-my-dataset/train.bin" in verify_source
