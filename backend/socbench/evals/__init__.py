"""Evaluation benchmark intelligence - contamination and saturation detection."""

from socbench.evals.analysis import analyze_evals
from socbench.evals.registry import EVALS, get_all_evals, get_eval
from socbench.evals.semantic_audit import BenchmarkAuditInput, audit_benchmark

__all__ = ["BenchmarkAuditInput", "EVALS", "analyze_evals", "audit_benchmark", "get_all_evals", "get_eval"]
