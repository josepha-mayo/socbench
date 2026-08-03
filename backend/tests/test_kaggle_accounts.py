"""Tests for Kaggle account profile loading."""

from __future__ import annotations

import json
from pathlib import Path

from socbench.kaggle.accounts import load_accounts


def _write_profile(root: Path, name: str) -> None:
    profile = root / name / ".kaggle"
    profile.mkdir(parents=True)
    (profile / "credentials.json").write_text(
        json.dumps({"username": name, "access_token": "token"}),
        encoding="utf-8",
    )


def test_load_accounts_excludes_disabled_profiles(tmp_path: Path):
    _write_profile(tmp_path, "alexcathe")
    _write_profile(tmp_path, "josephayanda")

    accounts = load_accounts(str(tmp_path))

    assert [account.name for account in accounts] == ["alexcathe"]
