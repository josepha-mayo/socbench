"""Eval analysis — contamination risk, saturation detection, and quality scoring.

For each eval benchmark we compute:
1. Saturation Index: Based on SOTA-to-ceiling ratio (ICML 2026 methodology).
   Saturation Index = SOTA / Ceiling. Values > 0.9 indicate saturation.
2. Contamination Risk: Computed from data availability, known incidents,
   open-source status, and saturation level. Weighted scoring.
3. Discriminative Power: How well the eval separates models (from registry).
4. Effective Quality: Composite score combining discrimination, annotation,
   coverage, and contamination resistance.
5. Status: fresh / maturing / saturated / contaminated / deprecated

Methodology based on:
- Saturation Index (arXiv:2602.16763, ICML 2026)
- Benchmark Health Index (arXiv:2602.11674)
- Contamination detection (GPT-3 paper, ConTAM arXiv:2411.03923)
- "Leak, Cheat, Repeat" (EACL 2024)
- Deprecation framework (arXiv:2507.06434)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from socbench.evals.registry import EVALS, EvalBenchmark, get_all_evals


@dataclass
class EvalAnalysis:
    key: str
    name: str
    category: str
    open_source: bool
    hf_id: Optional[str]
    description: str
    paper: str
    num_examples: int
    sota_pass_rate: float       # 0-100
    ceiling: float              # 0-100
    # Computed metrics
    saturation_index: float     # 0-100, SOTA/ceiling ratio
    contamination_risk: float   # 0-100, weighted risk score
    discriminative_power: float # 0-100, from registry
    annotation_quality: float   # 0-100, from registry
    coverage_breadth: float     # 0-100, from registry
    effective_quality: float    # 0-100, composite quality score
    headroom: float             # 0-100, ceiling - SOTA
    status: str                 # fresh, maturing, saturated, contaminated, deprecated
    # Metadata
    data_publicly_available: bool
    deprecated: bool
    successor: Optional[str]
    year: int
    actively_maintained: bool
    contamination_incidents: list[str]
    # Overlap data from our contamination checker (if available)
    overlap_with_training: float
    overlap_benchmarks: list[dict]


def _compute_saturation_index(eval_bench: EvalBenchmark) -> float:
    """Saturation Index = SOTA / Ceiling (ICML 2026 methodology).
    0 = fresh (no progress), 1.0 = fully saturated (SOTA at ceiling).
    Values > 0.9 indicate the eval is losing discriminative power."""
    if eval_bench.ceiling <= 0:
        return 0.0
    ratio = eval_bench.sota_pass_rate / eval_bench.ceiling
    return min(max(ratio, 0.0), 1.0)


def _compute_contamination_risk(eval_bench: EvalBenchmark) -> float:
    """Contamination risk score (0 = safe, 1 = high risk).

    Weighted factors based on research:
    - Data publicly available: +0.30 (can be scraped into training data)
    - Known contamination incidents: +0.15 per incident (capped at 0.45)
    - Open source: +0.10 (easier to include in training)
    - High saturation with public data: +0.15 (suggests memorization)
    - Not actively maintained: +0.05 (stale eval = more time for contamination)
    - Small dataset (<500 examples): +0.10 (easier to memorize)
    """
    risk = 0.0
    if eval_bench.data_publicly_available:
        risk += 0.30
    risk += min(len(eval_bench.contamination_incidents) * 0.15, 0.45)
    if eval_bench.open_source:
        risk += 0.10
    sat = _compute_saturation_index(eval_bench)
    if sat > 0.9 and eval_bench.data_publicly_available:
        risk += 0.15
    if not eval_bench.actively_maintained:
        risk += 0.05
    if eval_bench.num_examples > 0 and eval_bench.num_examples < 500:
        risk += 0.10
    return min(risk, 1.0)


def _compute_effective_quality(eval_bench: EvalBenchmark) -> float:
    """Composite quality score (0-1) combining multiple dimensions.

    Weights:
    - Discriminative power: 35% (most important — can it separate models?)
    - Contamination resistance: 25% (inverse of contamination risk)
    - Annotation quality: 15%
    - Coverage breadth: 15%
    - Active maintenance: 10%
    """
    cont_risk = _compute_contamination_risk(eval_bench)
    cont_resistance = 1.0 - cont_risk

    score = (
        0.35 * eval_bench.discriminative_power +
        0.25 * cont_resistance +
        0.15 * eval_bench.annotation_quality +
        0.15 * eval_bench.coverage_breadth +
        0.10 * (1.0 if eval_bench.actively_maintained else 0.0)
    )
    return min(max(score, 0.0), 1.0)


def _compute_status(eval_bench: EvalBenchmark, saturation: float, contamination: float) -> str:
    """Determine eval status label based on deprecation, saturation, and contamination."""
    if eval_bench.deprecated:
        return "deprecated"
    if contamination > 0.6:
        return "contaminated"
    if saturation > 0.9:
        return "saturated"
    if saturation > 0.75:
        return "maturing"
    return "fresh"


async def _get_overlap_from_db(eval_hf_id: str) -> tuple[float, list[dict]]:
    """Check our contamination table for overlap data with this eval benchmark."""
    if not eval_hf_id:
        return 0.0, []
    try:
        import sqlite3
        from pathlib import Path
        db_path = Path(__file__).parent.parent.parent / "socbench.db"
        if not db_path.exists():
            return 0.0, []
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Check if any datasets have contamination overlap with this benchmark
        eval_name = eval_hf_id.split("/")[-1].lower()
        rows = cur.execute(
            """SELECT c.benchmark_name, c.overlap_rate, c.overlap_count,
                      c.total_eval, d.hf_id as dataset_hf_id
               FROM contamination c
               JOIN datasets d ON c.dataset_id = d.id
               WHERE c.benchmark_name LIKE ? OR c.benchmark_name = ?""",
            (f"%{eval_name}%", eval_name)
        ).fetchall()
        conn.close()
        if not rows:
            return 0.0, []
        max_overlap = max(r["overlap_rate"] for r in rows)
        details = [
            {
                "dataset": r["dataset_hf_id"],
                "overlap_rate": r["overlap_rate"],
                "overlap_count": r["overlap_count"],
                "total_eval": r["total_eval"],
            }
            for r in rows
        ]
        return max_overlap, details
    except Exception:
        return 0.0, []


async def analyze_evals() -> list[EvalAnalysis]:
    """Analyze all registered evals for saturation, contamination, and quality."""
    results = []
    for eb in get_all_evals():
        sat = _compute_saturation_index(eb)
        cont_risk = _compute_contamination_risk(eb)
        eff_quality = _compute_effective_quality(eb)
        overlap, overlap_details = await _get_overlap_from_db(eb.hf_id or eb.key)
        # Blend our DB overlap data with heuristic risk
        if overlap > 0:
            cont_risk = max(cont_risk, min(overlap * 10, 1.0))
        headroom = max(eb.ceiling - eb.sota_pass_rate, 0.0)
        status = _compute_status(eb, sat, cont_risk)
        results.append(EvalAnalysis(
            key=eb.key,
            name=eb.name,
            category=eb.category,
            open_source=eb.open_source,
            hf_id=eb.hf_id,
            description=eb.description,
            paper=eb.paper,
            num_examples=eb.num_examples,
            sota_pass_rate=round(eb.sota_pass_rate * 100, 1),
            ceiling=round(eb.ceiling * 100, 1),
            saturation_index=round(sat * 100, 1),
            contamination_risk=round(cont_risk * 100, 1),
            discriminative_power=round(eb.discriminative_power * 100, 1),
            annotation_quality=round(eb.annotation_quality * 100, 1),
            coverage_breadth=round(eb.coverage_breadth * 100, 1),
            effective_quality=round(eff_quality * 100, 1),
            headroom=round(headroom * 100, 1),
            status=status,
            data_publicly_available=eb.data_publicly_available,
            deprecated=eb.deprecated,
            successor=eb.successor,
            year=eb.year,
            actively_maintained=eb.actively_maintained,
            contamination_incidents=eb.contamination_incidents,
            overlap_with_training=round(overlap, 6),
            overlap_benchmarks=overlap_details,
        ))
    # Sort by effective_quality desc (best evals first)
    results.sort(key=lambda x: -x.effective_quality)
    return results


def get_eval_summary() -> dict:
    """Get aggregate statistics about all evals."""
    evals = get_all_evals()
    total = len(evals)
    open_count = sum(1 for e in evals if e.open_source)
    closed_count = total - open_count
    saturated = sum(1 for e in evals if _compute_saturation_index(e) > 0.9)
    contaminated = sum(1 for e in evals if _compute_contamination_risk(e) > 0.5)
    deprecated = sum(1 for e in evals if e.deprecated)
    fresh = sum(1 for e in evals if _compute_saturation_index(e) < 0.6 and _compute_contamination_risk(e) < 0.4 and not e.deprecated)
    categories = {}
    for e in evals:
        cat = e.category
        if cat not in categories:
            categories[cat] = {"total": 0, "open": 0, "closed": 0}
        categories[cat]["total"] += 1
        if e.open_source:
            categories[cat]["open"] += 1
        else:
            categories[cat]["closed"] += 1
    return {
        "total": total,
        "open_source": open_count,
        "closed_source": closed_count,
        "saturated": saturated,
        "contaminated": contaminated,
        "deprecated": deprecated,
        "fresh": fresh,
        "categories": categories,
    }
