"""Evaluation benchmark intelligence - contamination and saturation detection."""

from socbench.evals.analysis import analyze_evals
from socbench.evals.registry import EVALS, get_all_evals, get_eval

__all__ = ["EVALS", "analyze_evals", "get_all_evals", "get_eval"]
