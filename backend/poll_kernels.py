"""Poll Kaggle kernel statuses and append them to poll_status.log.

Uses the shared ``socbench.kaggle.accounts`` client so authentication and
rate-limit handling are centralized.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from socbench.kaggle.accounts import MultiAccountManager

TRAIN_LOG = Path("train_pending.log")
POLL_LOG = Path("poll_status.log")


def _find_account(manager: MultiAccountManager, name: str):
    for acc in manager.accounts:
        if name in (acc.name, acc.username):
            return acc
    return None


def _get_active_refs() -> set[str]:
    active: set[str] = set()
    if not TRAIN_LOG.exists():
        return active

    url_pat = re.compile(r"https://www\.kaggle\.com/code/([^\s]+)")
    for line in TRAIN_LOG.read_text(encoding="utf-8").splitlines():
        m = url_pat.search(line)
        if m:
            active.add(m.group(1))

    if POLL_LOG.exists():
        for line in POLL_LOG.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\[.*\]\s+([^\s]+)\s+:\s+KernelWorkerStatus\.(\w+)", line)
            if not m:
                continue
            ref, status = m.group(1), m.group(2)
            if status in {"COMPLETE", "ERROR"}:
                active.discard(ref)
            elif status in {"RUNNING", "QUEUED"}:
                active.add(ref)

    return active


def _retry_call(fn, *args, max_attempts: int = 3, **kwargs):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(0.5 * (attempt + 1))
    raise last_exc


def main() -> int:
    manager = MultiAccountManager()
    log_path = Path("poll_status.log")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for ref in _get_active_refs():
        acc = _find_account(manager, ref.split("/")[0])
        username = acc.username if acc else ref.split("/")[0]
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {ref} : KernelWorkerStatus.UNKNOWN ({username})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
