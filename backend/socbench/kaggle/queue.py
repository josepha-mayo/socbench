"""Job queue for dataset training — assigns datasets to Kaggle accounts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from socbench.kaggle.accounts import MultiAccountManager


@dataclass
class TrainingJob:
    job_id: str
    dataset_id: str
    binary_path: str
    status: str = "pending"  # pending, preparing, running, complete, failed
    account_name: Optional[str] = None
    kernel_id: Optional[str] = None
    result: Optional[dict] = None
    created_at: float = field(default_factory=time.time)


class JobQueue:
    """Simple file-based job queue for training jobs."""

    def __init__(self, queue_dir: str = "./job_queue"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.manager = MultiAccountManager()

    def enqueue(
        self,
        dataset_id: str,
        binary_path: str,
    ) -> TrainingJob:
        """Add a training job to the queue."""
        job_id = f"{dataset_id.replace('/', '_')}_{int(time.time())}"
        job = TrainingJob(
            job_id=job_id,
            dataset_id=dataset_id,
            binary_path=binary_path,
        )
        job_file = self.queue_dir / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(
                {
                    "job_id": job.job_id,
                    "dataset_id": job.dataset_id,
                    "binary_path": job.binary_path,
                    "status": job.status,
                    "created_at": job.created_at,
                },
                f,
                indent=2,
            )
        return job

    def get_pending(self) -> list[TrainingJob]:
        """Get all pending jobs."""
        jobs = []
        for f in sorted(self.queue_dir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            if data.get("status") == "pending":
                jobs.append(
                    TrainingJob(
                        job_id=data["job_id"],
                        dataset_id=data["dataset_id"],
                        binary_path=data["binary_path"],
                        status=data["status"],
                    )
                )
        return jobs

    def update_status(self, job_id: str, status: str, **kwargs) -> None:
        """Update job status."""
        job_file = self.queue_dir / f"{job_id}.json"
        if job_file.exists():
            with open(job_file) as f:
                data = json.load(f)
            data["status"] = status
            data.update(kwargs)
            with open(job_file, "w") as f:
                json.dump(data, f, indent=2)

    def get_stats(self) -> dict:
        """Get queue statistics."""
        counts = {"pending": 0, "running": 0, "complete": 0, "failed": 0}
        for f in self.queue_dir.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
            status = data.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts
