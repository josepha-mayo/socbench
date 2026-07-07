"""Kaggle multi-account manager — reuse existing profiles from model_ablation."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class KaggleAccount:
    name: str
    config_dir: str
    username: str
    active_kernels: int = 0
    max_kernels: int = 2
    last_used: Optional[float] = None


# Default profile paths (model_ablation structure)
DEFAULT_PROFILES_DIR = r"C:\Users\USER\.vscode\model_ablation\.kaggle_profiles"

ACCOUNT_NAMES = [
    "alexcathe", "hiolyjo", "holyjow", "holykeys", "holykeyz10",
    "ippojoe", "jjwhich", "josephayanda", "josephmayo", "josephmayok", "makanouchi",
]


def load_accounts(profiles_dir: str = DEFAULT_PROFILES_DIR) -> list[KaggleAccount]:
    """Load all Kaggle accounts from profile directories."""
    accounts = []
    profiles_path = Path(profiles_dir)

    for name in ACCOUNT_NAMES:
        profile_dir = profiles_path / name
        cred_file = profile_dir / ".kaggle" / "credentials.json"
        kaggle_file = profile_dir / ".kaggle" / "kaggle.json"

        # Try credentials.json first, then kaggle.json
        config_file = cred_file if cred_file.exists() else kaggle_file
        if not config_file.exists():
            continue

        try:
            with open(config_file) as f:
                creds = json.load(f)
            username = creds.get("username", name)
            accounts.append(
                KaggleAccount(
                    name=name,
                    config_dir=str(profile_dir / ".kaggle"),
                    username=username,
                )
            )
        except (json.JSONDecodeError, KeyError):
            continue

    return accounts


class MultiAccountManager:
    """Manage multiple Kaggle accounts for parallel kernel execution."""

    def __init__(self, profiles_dir: str = DEFAULT_PROFILES_DIR):
        self.accounts = load_accounts(profiles_dir)
        self.results: dict[str, dict] = {}

    def get_available_account(self) -> Optional[KaggleAccount]:
        """Find account with available kernel slots."""
        now = time.time()
        available = [
            a for a in self.accounts
            if a.active_kernels < a.max_kernels
            and (a.last_used is None or now - a.last_used > 10)
        ]
        if not available:
            return None
        # Prefer least recently used
        return min(available, key=lambda a: a.last_used or 0)

    def get_total_slots(self) -> int:
        return sum(a.max_kernels for a in self.accounts)

    def get_active_slots(self) -> int:
        return sum(a.active_kernels for a in self.accounts)

    def push_kernel(
        self,
        account: KaggleAccount,
        kernel_path: str,
        kernel_id: Optional[str] = None,
    ) -> dict:
        """Push a kernel using a specific account."""
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = account.config_dir

        # Push
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", kernel_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return {"success": False, "error": result.stderr, "account": account.name}

        account.active_kernels += 1
        account.last_used = time.time()

        return {"success": True, "account": account.name, "output": result.stdout}

    def check_status(self, account: KaggleAccount, kernel_id: str) -> str:
        """Check kernel status."""
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = account.config_dir

        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        return result.stdout.strip() if result.returncode == 0 else "unknown"

    def wait_for_kernel(
        self,
        account: KaggleAccount,
        kernel_id: str,
        timeout: int = 43200,  # 12 hours
        poll_interval: int = 60,
    ) -> dict:
        """Wait for kernel to complete."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.check_status(account, kernel_id)
            if "complete" in status.lower():
                account.active_kernels -= 1
                return {"success": True, "status": "complete"}
            elif "error" in status.lower() or "failed" in status.lower():
                account.active_kernels -= 1
                return {"success": False, "status": status}
            time.sleep(poll_interval)

        account.active_kernels -= 1
        return {"success": False, "status": "timeout"}

    def push_with_fallback(
        self,
        kernel_path: str,
        max_attempts: int = 3,
    ) -> dict:
        """Try pushing to accounts until one succeeds."""
        attempts = 0
        while attempts < max_attempts:
            account = self.get_available_account()
            if account is None:
                time.sleep(30)
                attempts += 1
                continue

            result = self.push_kernel(account, kernel_path)
            if result["success"]:
                return result
            attempts += 1

        return {"success": False, "error": "No accounts available"}
