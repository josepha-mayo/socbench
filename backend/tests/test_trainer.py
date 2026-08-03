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
        "compile = True",
    ]
    for marker in required:
        assert marker in trainer_script, marker
    assert "block_size * 2 *" in trainer_script
    assert "loss_curve=comparable_curve" in trainer_script
    assert "{str(TRAIN.compile)}" not in trainer_script


def test_generated_train_script_compiles(trainer_script):
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "train.py"
        script_path.write_text(trainer_script, encoding="utf-8")
        py_compile.compile(str(script_path), doraise=True)


def test_generated_train_script_has_expected_data_validation(trainer_script):
    assert "if n_total < 2:" in trainer_script
    assert "raise ValueError" in trainer_script
    assert "the dataset will repeat" in trainer_script.lower()


def test_generated_train_script_has_ddp_fallback(trainer_script):
    assert 'init_process_group("nccl"' in trainer_script
    assert 'init_process_group("gloo"' in trainer_script
    assert "falling back to gloo backend" in trainer_script.lower()


def test_save_training_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "train.py"
        path = save_training_script(str(output), dataset_bin_path="/tmp/data.bin")
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "def main(" in content
