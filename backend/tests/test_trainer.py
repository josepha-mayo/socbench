"""Tests for socbench.training.trainer."""

from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path

import pytest

from socbench.training.trainer import generate_train_script, save_training_script


@pytest.fixture
def trainer_script():
    return generate_train_script("/tmp/data.bin", output_dir="/tmp/output", tokens=1_000_000_000)


def test_generate_train_script_returns_string(trainer_script):
    assert isinstance(trainer_script, str)
    assert trainer_script.strip()


def test_generate_train_script_contains_required_markers(trainer_script):
    required = [
        "def main(",
        "if __name__ == \"__main__\":",
        "NCCL_P2P_DISABLE",
        "destroy_process_group",
        "loss_curve.json",
        "ckpt_final.pt",
        "eval_results.json",
        "save_loss_curve",
        "configure_optimizers",
        "compile = False",
    ]
    for marker in required:
        assert marker in trainer_script, marker
    assert "batch_size * block_size * gradient_accumulation_steps" in trainer_script
    assert "tokens_per_iter =" in trainer_script
    assert "actual_tokens_seen = iter_num * tokens_per_iter" in trainer_script
    assert "loss_curve=comparable_curve" in trainer_script
    assert "SOCBENCH_ALLOW_CPU_FALLBACK" in trainer_script
    assert "SOCBENCH_TRAIN_BATCH_SIZE" in trainer_script
    assert "SOCBENCH_GRADIENT_ACCUMULATION_STEPS" in trainer_script
    assert "SOCBENCH_TRAIN_COMPILE" in trainer_script
    assert "Set SOCBENCH_ALLOW_CPU_FALLBACK=1 only for explicit smoke/debug runs" in trainer_script
    assert "with torch.inference_mode():" in trainer_script
    assert 'required_world_size = int(os.environ.get("SOCBENCH_REQUIRED_WORLD_SIZE", "1"))' in trainer_script
    assert "torch.distributed.all_reduce(val_loss_tensor" in trainer_script
    assert "tokens_per_sec = tokens_per_iter / dt" in trainer_script
    assert "SOCBENCH_FULL_RUN" in trainer_script
    assert "SOCBENCH_ALLOW_EXCESSIVE_REPETITION" in trainer_script
    assert "torch.amp.GradScaler" in trainer_script
    assert "scaler.unscale_(optimizer)" in trainer_script
    assert "Non-finite gradient norm" in trainer_script
    assert "calibration improved only" in trainer_script
    assert "validation loss exceeded baseline" in trainer_script
    assert "torch.distributed.barrier()" in trainer_script
    assert 'init_process_group("gloo")' not in trainer_script
    assert "{str(TRAIN.compile)}" not in trainer_script


def test_generated_train_script_compiles(trainer_script):
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "train.py"
        script_path.write_text(trainer_script, encoding="utf-8")
        py_compile.compile(str(script_path), doraise=True)


def test_generated_train_script_has_expected_data_validation(trainer_script):
    assert "if n_train <= block_size or n_val <= block_size:" in trainer_script
    assert "raise ValueError" in trainer_script
    assert "planned token budget repeats" in trainer_script.lower()


def test_generated_train_script_requires_nccl_ddp(trainer_script):
    assert 'init_process_group("nccl"' in trainer_script
    assert 'init_process_group("gloo"' not in trainer_script


def test_save_training_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "train.py"
        path = save_training_script(str(output), dataset_bin_path="/tmp/data.bin")
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "def main(" in content
