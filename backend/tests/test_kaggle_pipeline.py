"""Tests for Kaggle training bundle creation."""

from __future__ import annotations

import json
from pathlib import Path

from socbench.kaggle.pipeline import build_dataset_metadata, create_training_bundle, push_training_bundle


def test_build_dataset_metadata_uses_owner_and_safe_slug():
    metadata = build_dataset_metadata("Org/My Dataset!", "holykeys")

    assert metadata["id"] == "holykeys/org-my-dataset"
    assert metadata["title"] == "Socbench data - Org/My Dataset!"
    assert metadata["licenses"] == [{"name": "CC0-1.0"}]


def test_build_dataset_metadata_clamps_long_titles():
    metadata = build_dataset_metadata("SomeOrg/" + "very-long-dataset-name-" * 5, "holykeys")

    assert 6 <= len(metadata["title"]) <= 50
    assert metadata["title"].startswith("Socbench data - ")


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
    assert bundle.train_device == "cuda"
    assert bundle.allow_cpu_fallback is False
    assert bundle.train_batch_size == 8
    assert bundle.gradient_accumulation_steps == 64
    assert bundle.train_compile is False
    assert bundle.train_bin.exists()
    assert bundle.train_bin.read_bytes() == b"\x01\x00\x02\x00"

    dataset_metadata = json.loads((bundle.data_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert dataset_metadata["id"] == "holykeys/org-my-dataset"
    assert (bundle.data_dir / "metadata.json").exists()

    kernel_metadata = json.loads((bundle.kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert kernel_metadata["id"] == "holykeys/socbench-train-org-my-b7e6"
    assert kernel_metadata["dataset_sources"] == ["holykeys/org-my-dataset"]
    assert kernel_metadata["kernel_type"] == "script"
    assert kernel_metadata["code_file"] == "kernel.py"

    kernel_source = (bundle.kernel_dir / "kernel.py").read_text(encoding="utf-8")
    compile(kernel_source, "<generated-kaggle-script>", "exec")
    assert "/kaggle/input/org-my-dataset/train.bin" in kernel_source
    assert "os.environ['SOCBENCH_TRAIN_DEVICE'] = \"cuda\"" in kernel_source
    assert "os.environ['SOCBENCH_ALLOW_CPU_FALLBACK'] = '0'" in kernel_source
    assert "os.environ['SOCBENCH_TRAIN_BATCH_SIZE'] = '8'" in kernel_source
    assert "os.environ['SOCBENCH_GRADIENT_ACCUMULATION_STEPS'] = '64'" in kernel_source
    assert "os.environ['SOCBENCH_TRAIN_COMPILE'] = '0'" in kernel_source


def test_push_training_bundle_versions_existing_dataset(tmp_path: Path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "train.bin").write_bytes(b"\x01\x00\x02\x00")
    bundle = create_training_bundle(
        dataset_id="Org/My Dataset!",
        prepared_data_dir=prepared,
        output_root=tmp_path / "bundles",
        kaggle_owner="holykeys",
        tokens=1234,
    )
    calls = []

    def fake_run(args, config_dir, timeout=180, api_token=None, credentials_file=None):
        calls.append(args)
        assert api_token == "secret-token"
        assert credentials_file == "C:/fake/.kaggle/credentials.json"
        if args[:3] == ["kaggle", "datasets", "create"]:
            raise RuntimeError("Dataset already exists")
        return "ok"

    monkeypatch.setattr("socbench.kaggle.pipeline._run_kaggle_command", fake_run)

    result = push_training_bundle(
        bundle=bundle,
        kaggle_config_dir="C:/fake/.kaggle",
        kaggle_api_token="secret-token",
        kaggle_credentials_file="C:/fake/.kaggle/credentials.json",
    )

    assert result.dataset_action == "version"
    assert calls[0][:3] == ["kaggle", "datasets", "create"]
    assert calls[1][:3] == ["kaggle", "datasets", "version"]
    assert calls[2][:3] == ["kaggle", "kernels", "push"]
    assert calls[2][-2:] == ["--accelerator", "NvidiaTeslaT4"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["kaggle_dataset_ref"] == "holykeys/org-my-dataset"
    assert manifest["kernel_id"] == "holykeys/socbench-train-org-my-b7e6"
    assert manifest["train_device"] == "cuda"
    assert manifest["allow_cpu_fallback"] is False
    assert manifest["train_batch_size"] == 8
    assert manifest["gradient_accumulation_steps"] == 64
    assert manifest["train_compile"] is False
    assert manifest["accelerator"] == "NvidiaTeslaT4"
