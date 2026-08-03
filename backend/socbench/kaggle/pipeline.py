"""Offline Kaggle training bundle builder.

This module deliberately separates bundle creation from network-side Kaggle
pushes. That makes the training pipeline reviewable: first create a directory
containing dataset metadata, the prepared binary, notebook, and kernel metadata;
then push those artifacts with the Kaggle CLI from a chosen account.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from socbench.kaggle.notebook import dataset_safe_id, save_notebook

DEFAULT_LICENSE = "CC0-1.0"


@dataclass(frozen=True)
class KaggleTrainingBundle:
    dataset_id: str
    kaggle_owner: str
    kaggle_dataset_slug: str
    kaggle_dataset_ref: str
    kernel_slug: str
    bundle_dir: Path
    data_dir: Path
    kernel_dir: Path
    train_bin: Path


def build_dataset_metadata(
    dataset_id: str,
    kaggle_owner: str,
    kaggle_dataset_slug: str | None = None,
    license_name: str = DEFAULT_LICENSE,
) -> dict:
    """Return Kaggle dataset metadata for a prepared Socbench binary."""
    slug = kaggle_dataset_slug or dataset_safe_id(dataset_id)
    return {
        "title": f"Socbench prepared data - {dataset_id}",
        "id": f"{kaggle_owner}/{slug}",
        "licenses": [{"name": license_name}],
    }


def link_or_copy_file(source: Path, destination: Path) -> None:
    """Hard-link a file when possible, otherwise copy it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def create_training_bundle(
    dataset_id: str,
    prepared_data_dir: str | Path,
    output_root: str | Path,
    kaggle_owner: str,
    tokens: int = 1_000_000_000,
    binary_filename: str = "train.bin",
    license_name: str = DEFAULT_LICENSE,
) -> KaggleTrainingBundle:
    """Create a local Kaggle dataset+kernel bundle for one training run."""
    prepared_dir = Path(prepared_data_dir)
    source_bin = prepared_dir / binary_filename
    if not source_bin.exists():
        raise FileNotFoundError(f"Missing prepared binary: {source_bin}")

    dataset_slug = dataset_safe_id(dataset_id)
    bundle_dir = Path(output_root) / dataset_slug
    data_dir = bundle_dir / "data"
    kernel_dir = bundle_dir / "kernel"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_bin = data_dir / binary_filename
    link_or_copy_file(source_bin, train_bin)

    metadata_path = prepared_dir / "metadata.json"
    if metadata_path.exists():
        shutil.copy2(metadata_path, data_dir / "metadata.json")

    dataset_metadata = build_dataset_metadata(
        dataset_id=dataset_id,
        kaggle_owner=kaggle_owner,
        kaggle_dataset_slug=dataset_slug,
        license_name=license_name,
    )
    (data_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    notebook_result = save_notebook(
        str(kernel_dir),
        dataset_id,
        binary_filename=binary_filename,
        tokens=tokens,
        dataset_owner=kaggle_owner,
        kaggle_dataset_slug=dataset_slug,
    )

    return KaggleTrainingBundle(
        dataset_id=dataset_id,
        kaggle_owner=kaggle_owner,
        kaggle_dataset_slug=dataset_slug,
        kaggle_dataset_ref=f"{kaggle_owner}/{dataset_slug}",
        kernel_slug=notebook_result["slug"],
        bundle_dir=bundle_dir,
        data_dir=data_dir,
        kernel_dir=kernel_dir,
        train_bin=train_bin,
    )
