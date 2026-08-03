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
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from socbench.kaggle.notebook import dataset_safe_id, save_script_kernel

DEFAULT_LICENSE = "CC0-1.0"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
MAX_KAGGLE_TITLE_LENGTH = 50


@dataclass(frozen=True)
class KaggleTrainingBundle:
    dataset_id: str
    kaggle_owner: str
    kaggle_dataset_slug: str
    kaggle_dataset_ref: str
    kernel_slug: str
    train_device: str
    allow_cpu_fallback: bool
    bundle_dir: Path
    data_dir: Path
    kernel_dir: Path
    train_bin: Path


@dataclass(frozen=True)
class KaggleLaunchResult:
    bundle: KaggleTrainingBundle
    dataset_action: str
    dataset_output: str
    kernel_output: str
    manifest_path: Path


def build_dataset_metadata(
    dataset_id: str,
    kaggle_owner: str,
    kaggle_dataset_slug: str | None = None,
    license_name: str = DEFAULT_LICENSE,
) -> dict:
    """Return Kaggle dataset metadata for a prepared Socbench binary."""
    slug = kaggle_dataset_slug or dataset_safe_id(dataset_id)
    title = f"Socbench data - {dataset_id}"
    if len(title) > MAX_KAGGLE_TITLE_LENGTH:
        title = f"Socbench data - {slug}"[:MAX_KAGGLE_TITLE_LENGTH].rstrip("- ")
    return {
        "title": title,
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
    train_device: str = "cuda",
    allow_cpu_fallback: bool = False,
) -> KaggleTrainingBundle:
    """Create a local Kaggle dataset+kernel bundle for one training run."""
    train_device = train_device.lower()
    if train_device not in {"cuda", "cpu"}:
        raise ValueError("train_device must be 'cuda' or 'cpu'")
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

    kernel_result = save_script_kernel(
        str(kernel_dir),
        dataset_id,
        binary_filename=binary_filename,
        tokens=tokens,
        dataset_owner=kaggle_owner,
        kaggle_dataset_slug=dataset_slug,
        train_device=train_device,
        allow_cpu_fallback=allow_cpu_fallback,
    )

    return KaggleTrainingBundle(
        dataset_id=dataset_id,
        kaggle_owner=kaggle_owner,
        kaggle_dataset_slug=dataset_slug,
        kaggle_dataset_ref=f"{kaggle_owner}/{dataset_slug}",
        kernel_slug=kernel_result["slug"],
        train_device=train_device,
        allow_cpu_fallback=allow_cpu_fallback,
        bundle_dir=bundle_dir,
        data_dir=data_dir,
        kernel_dir=kernel_dir,
        train_bin=train_bin,
    )


def _run_kaggle_command(
    args: list[str],
    config_dir: str,
    timeout: int = 180,
    api_token: str | None = None,
    credentials_file: str | None = None,
) -> str:
    env = os.environ.copy()
    if api_token:
        env["KAGGLE_API_TOKEN"] = api_token

    temp_home: tempfile.TemporaryDirectory | None = None
    if credentials_file:
        temp_home = tempfile.TemporaryDirectory()
        kaggle_home = Path(temp_home.name) / ".kaggle"
        kaggle_home.mkdir(parents=True, exist_ok=True)
        shutil.copy2(credentials_file, kaggle_home / "credentials.json")
        env["USERPROFILE"] = temp_home.name
        env["KAGGLE_CONFIG_DIR"] = str(kaggle_home)
    else:
        env["KAGGLE_CONFIG_DIR"] = config_dir

    try:
        result = subprocess.run(
            args,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        if temp_home is not None:
            temp_home.cleanup()

    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    if result.returncode != 0:
        raise RuntimeError(output or f"Kaggle command failed: {' '.join(args)}")
    return output


def push_training_bundle(
    bundle: KaggleTrainingBundle,
    kaggle_config_dir: str,
    kaggle_api_token: str | None = None,
    kaggle_credentials_file: str | None = None,
    message: str = "Update prepared Socbench data",
    accelerator: str | None = DEFAULT_ACCELERATOR,
) -> KaggleLaunchResult:
    """Upload/refresh the prepared dataset, push the kernel, and write a manifest."""
    try:
        dataset_output = _run_kaggle_command(
            ["kaggle", "datasets", "create", "-p", str(bundle.data_dir), "--dir-mode", "zip"],
            config_dir=kaggle_config_dir,
            timeout=600,
            api_token=kaggle_api_token,
            credentials_file=kaggle_credentials_file,
        )
        dataset_action = "create"
    except RuntimeError as exc:
        if "already exists" not in str(exc).lower():
            raise
        dataset_output = _run_kaggle_command(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(bundle.data_dir),
                "-m",
                message,
                "--dir-mode",
                "zip",
            ],
            config_dir=kaggle_config_dir,
            timeout=600,
            api_token=kaggle_api_token,
            credentials_file=kaggle_credentials_file,
        )
        dataset_action = "version"

    kernel_args = ["kaggle", "kernels", "push", "-p", str(bundle.kernel_dir)]
    if accelerator:
        kernel_args.extend(["--accelerator", accelerator])

    kernel_output = _run_kaggle_command(
        kernel_args,
        config_dir=kaggle_config_dir,
        timeout=240,
        api_token=kaggle_api_token,
        credentials_file=kaggle_credentials_file,
    )

    manifest = {
        "dataset_id": bundle.dataset_id,
        "kaggle_owner": bundle.kaggle_owner,
        "kaggle_dataset_ref": bundle.kaggle_dataset_ref,
        "kernel_id": f"{bundle.kaggle_owner}/{bundle.kernel_slug}",
        "kernel_slug": bundle.kernel_slug,
        "train_device": bundle.train_device,
        "allow_cpu_fallback": bundle.allow_cpu_fallback,
        "accelerator": accelerator,
        "dataset_action": dataset_action,
        "bundle_dir": str(bundle.bundle_dir),
        "data_dir": str(bundle.data_dir),
        "kernel_dir": str(bundle.kernel_dir),
        "train_bin": str(bundle.train_bin),
        "launched_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = bundle.bundle_dir / "launch-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return KaggleLaunchResult(
        bundle=bundle,
        dataset_action=dataset_action,
        dataset_output=dataset_output,
        kernel_output=kernel_output,
        manifest_path=manifest_path,
    )
