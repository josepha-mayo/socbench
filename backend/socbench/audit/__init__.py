"""Socbench dataset audit pipeline.

Public API for running the 7-stage audit on any HuggingFace dataset:

    from socbench.audit import audit_dataset
    summary = asyncio.run(audit_dataset("owner/name", output_dir="audit_out"))

Or from the CLI:

    socbench audit owner/name --output-dir audit_out --max-rows 100000
"""

from __future__ import annotations

from socbench.audit.pipeline import AuditConfig, AuditResult, audit_dataset

__all__ = ["AuditResult", "AuditConfig", "audit_dataset"]
