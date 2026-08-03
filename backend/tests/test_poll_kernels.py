"""Tests for poll_kernels helpers."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import poll_kernels as pk
from poll_kernels import _find_account, _get_active_refs, _retry_call
from socbench.kaggle.accounts import KaggleAccount, MultiAccountManager

# ---------------------------------------------------------------------------
# _find_account
# ---------------------------------------------------------------------------


def _manager_with_accounts(monkeypatch: pytest.MonkeyPatch, accounts):
    monkeypatch.setattr("socbench.kaggle.accounts.load_accounts", lambda *a, **k: accounts)
    return MultiAccountManager()


def test_find_account_by_name(monkeypatch: pytest.MonkeyPatch):
    acc = KaggleAccount(name="acc1", config_dir="/x", username="user1")
    manager = _manager_with_accounts(monkeypatch, [acc])
    assert _find_account(manager, "acc1") is acc


def test_find_account_by_username(monkeypatch: pytest.MonkeyPatch):
    acc = KaggleAccount(name="acc1", config_dir="/x", username="user1")
    manager = _manager_with_accounts(monkeypatch, [acc])
    assert _find_account(manager, "user1") is acc


def test_find_account_not_found(monkeypatch: pytest.MonkeyPatch):
    acc = KaggleAccount(name="acc1", config_dir="/x", username="user1")
    manager = _manager_with_accounts(monkeypatch, [acc])
    assert _find_account(manager, "missing") is None


# ---------------------------------------------------------------------------
# _get_active_refs
# ---------------------------------------------------------------------------


def _write_train_log(path: Path, *lines: str) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_poll_log(path: Path, *lines: str) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_get_active_refs_no_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pk, "TRAIN_LOG", tmp_path / "missing_train.log")
    monkeypatch.setattr(pk, "POLL_LOG", tmp_path / "missing_poll.log")
    assert _get_active_refs() == set()


def test_get_active_refs_no_poll_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_log = tmp_path / "train_pending.log"
    _write_train_log(
        train_log,
        "Pushing kernel for owner/ds on holykeys...",
        "Kernel version 1 successfully pushed. Please check progress at "
        "https://www.kaggle.com/code/holykeys/socbench-train-abc123",
    )
    monkeypatch.setattr(pk, "TRAIN_LOG", train_log)
    monkeypatch.setattr(pk, "POLL_LOG", tmp_path / "missing_poll.log")
    assert _get_active_refs() == {"holykeys/socbench-train-abc123"}


def test_get_active_refs_running_queued_stay_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_log = tmp_path / "train_pending.log"
    _write_train_log(
        train_log,
        "Pushing kernel for owner/ds on holykeys...",
        "Kernel version 1 successfully pushed. Please check progress at "
        "https://www.kaggle.com/code/holykeys/socbench-train-abc123",
        "Pushing kernel for other/data on makanouchi...",
        "Kernel version 2 successfully pushed. Please check progress at "
        "https://www.kaggle.com/code/makanouchi/socbench-train-def456",
    )
    poll_log = tmp_path / "poll_status.log"
    _write_poll_log(
        poll_log,
        "[2026-07-19 00:00:00] holykeys/socbench-train-abc123 : KernelWorkerStatus.RUNNING",
        "[2026-07-19 00:00:00] makanouchi/socbench-train-def456 : KernelWorkerStatus.QUEUED",
    )
    monkeypatch.setattr(pk, "TRAIN_LOG", train_log)
    monkeypatch.setattr(pk, "POLL_LOG", poll_log)
    assert _get_active_refs() == {
        "holykeys/socbench-train-abc123",
        "makanouchi/socbench-train-def456",
    }


def test_get_active_refs_complete_error_inactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_log = tmp_path / "train_pending.log"
    _write_train_log(
        train_log,
        "Pushing kernel for owner/ds on holykeys...",
        "Kernel version 1 successfully pushed. Please check progress at "
        "https://www.kaggle.com/code/holykeys/socbench-train-abc123",
        "Pushing kernel for other/data on makanouchi...",
        "Kernel version 2 successfully pushed. Please check progress at "
        "https://www.kaggle.com/code/makanouchi/socbench-train-def456",
    )
    poll_log = tmp_path / "poll_status.log"
    _write_poll_log(
        poll_log,
        "[2026-07-19 00:00:00] holykeys/socbench-train-abc123 : KernelWorkerStatus.COMPLETE",
        "[2026-07-19 00:00:00] makanouchi/socbench-train-def456 : KernelWorkerStatus.ERROR",
    )
    monkeypatch.setattr(pk, "TRAIN_LOG", train_log)
    monkeypatch.setattr(pk, "POLL_LOG", poll_log)
    assert _get_active_refs() == set()


def test_get_active_refs_ignores_unmatched_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_log = tmp_path / "train_pending.log"
    _write_train_log(
        train_log,
        "Pushing kernel for owner/ds on holykeys...",
        "Pushing kernel for other/data on makanouchi...",
        "Kernel version 2 successfully pushed. Please check progress at "
        "https://www.kaggle.com/code/makanouchi/socbench-train-def456",
    )
    monkeypatch.setattr(pk, "TRAIN_LOG", train_log)
    monkeypatch.setattr(pk, "POLL_LOG", tmp_path / "missing_poll.log")
    assert _get_active_refs() == {"makanouchi/socbench-train-def456"}


# ---------------------------------------------------------------------------
# _retry_call
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pk, "time", types.SimpleNamespace(sleep=lambda *_: None))


def test_retry_call_success_first_try():
    def fn(x):
        return x + 1

    assert _retry_call(fn, 1) == 2


def test_retry_call_success_on_retry():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("fail")
        return "ok"

    assert _retry_call(fn, max_attempts=3) == "ok"
    assert len(calls) == 2


def test_retry_call_failure_after_retries():
    def fn():
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        _retry_call(fn, max_attempts=2)
