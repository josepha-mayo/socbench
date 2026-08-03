"""Tests for socbench.kaggle.notebook."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from socbench.kaggle.notebook import generate_kernel_script, generate_notebook, kernel_slug_for, save_notebook


def test_kernel_slug_for_stability_and_collision():
    ds = "user/my-dataset-name"
    slug1 = kernel_slug_for(ds)
    slug2 = kernel_slug_for(ds)
    assert slug1.startswith("socbench-train-")
    assert len(slug1) <= len("socbench-") + 22  # prefix + short_id up to 11 chars
    assert slug1 == slug2

    # Different dataset IDs should not collide (high probability due to md4 digest)
    a = kernel_slug_for("abcdefgh-foo")
    b = kernel_slug_for("abcdefgh-bar")
    assert a != b


def test_generate_notebook_returns_valid_json_and_expected_strings():
    result = generate_notebook("user/my-dataset", tokens=1_000_000_000)
    notebook = result["notebook"]
    kernel_metadata = result["kernel_metadata"]
    kernel_slug = result["kernel_slug"]

    # Valid JSON serialization
    json.dumps(notebook)
    json.dumps(kernel_metadata)

    assert notebook["nbformat"] == 4
    assert kernel_metadata["id"] == f"socbench/{kernel_slug}"

    # Training execution cell (fourth cell, index 3)
    source = "".join(notebook["cells"][3]["source"])
    assert "subprocess.run" in source
    assert "train.log" in source
    assert "Training exited with code" in source
    assert "NCCL_P2P_DISABLE" in source
    assert "TOKENIZERS_PARALLELISM" in source
    assert "SOCBENCH_TRAIN_DEVICE'] = \"cuda\"" in source
    assert "SOCBENCH_ALLOW_CPU_FALLBACK'] = '0'" in source
    assert "RuntimeError" in source
    compile(source, "<generated-notebook-cell>", "exec")

    # Setup cell uses subprocess.run instead of check_call
    setup_source = "".join(notebook["cells"][1]["source"])
    assert "subprocess.run" in setup_source
    assert "subprocess.check_call" not in setup_source

    # Data verification cell contains Kaggle's mounted dataset path.
    verify_source = "".join(notebook["cells"][2]["source"])
    assert "/kaggle/input/user-my-dataset/" in verify_source

    assert kernel_metadata["dataset_sources"] == ["<owner>/user-my-dataset"]


def test_generate_notebook_uses_selected_kaggle_owner_for_kernel_id():
    result = generate_notebook("user/my-dataset", dataset_owner="holykeys")
    kernel_metadata = result["kernel_metadata"]

    assert kernel_metadata["id"] == f"holykeys/{result['kernel_slug']}"
    assert kernel_metadata["dataset_sources"] == ["holykeys/user-my-dataset"]


def test_save_notebook_writes_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_notebook(tmpdir, "user/my-dataset", tokens=1_000_000)
        nb_path = Path(tmpdir) / "notebook.ipynb"
        meta_path = Path(tmpdir) / "kernel-metadata.json"
        assert nb_path.exists()
        assert meta_path.exists()

        notebook = json.loads(nb_path.read_text(encoding="utf-8"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        assert notebook["nbformat"] == 4
        assert metadata["id"].startswith("socbench/")
        assert metadata["code_file"] == "notebook.ipynb"


def test_generate_kernel_script_compiles_and_uses_script_metadata():
    result = generate_kernel_script("user/my-dataset", dataset_owner="holykeys", tokens=5_000_000)

    compile(result["source"], "<generated-kaggle-script>", "exec")
    assert result["kernel_metadata"]["id"] == f"holykeys/{result['kernel_slug']}"
    assert result["kernel_metadata"]["kernel_type"] == "script"
    assert result["kernel_metadata"]["code_file"] == "kernel.py"
    assert "SOCBENCH_RESULT_JSON=" in result["source"]
    assert "os.environ['SOCBENCH_TRAIN_DEVICE'] = \"cuda\"" in result["source"]
    assert "os.environ['SOCBENCH_ALLOW_CPU_FALLBACK'] = '0'" in result["source"]
    assert "'torch'," not in result["source"]
    assert "/kaggle/input/datasets/holykeys/user-my-dataset/train.bin" in result["source"]


def test_generate_kernel_script_can_opt_into_cpu_smoke_mode():
    result = generate_kernel_script(
        "user/my-dataset",
        dataset_owner="holykeys",
        train_device="cpu",
        allow_cpu_fallback=True,
    )

    assert "os.environ['SOCBENCH_TRAIN_DEVICE'] = \"cpu\"" in result["source"]
    assert "os.environ['SOCBENCH_ALLOW_CPU_FALLBACK'] = '1'" in result["source"]


def test_kernel_slug_for_empty_prefix_fallback():
    # An all-non-ascii-ish id that would yield empty safe_id still produces a slug
    slug = kernel_slug_for("!!!")
    assert slug.startswith("socbench-train-")
