"""Cache-first dataset evidence, readiness decisions, and comparisons."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import delete, select

from socbench.db import async_session_factory
from socbench.models import (
    ContaminationRow,
    DatasetRow,
    LeaderboardRow,
    ScoreRow,
    TrainingRunRow,
)

SCORE_DIMENSIONS = (
    "quality",
    "diversity",
    "utility",
    "documentation",
    "popularity",
    "freshness",
    "pii_safety",
)

Decision = Literal["PROCEED", "REVIEW", "DO NOT PROCEED"]
EvidenceSource = Literal["cache", "live"]
Scorer = Callable[..., Awaitable[dict[str, Any]]]


class DatasetIntelligenceError(RuntimeError):
    """Base class for user-facing dataset intelligence failures."""


class CacheMissError(DatasetIntelligenceError):
    """Raised when complete cached evidence was required but unavailable."""


class ScoringFailedError(DatasetIntelligenceError):
    """Raised when live scoring did not produce complete, persistable evidence."""


@dataclass(frozen=True)
class DimensionEvidence:
    score: float | None
    details: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    sample_size: int | None = None
    scored_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "details": self.details,
            "warnings": list(self.warnings),
            "sample_size": self.sample_size,
            "scored_at": _iso(self.scored_at),
        }


@dataclass(frozen=True)
class ReadinessAssessment:
    decision: Decision
    reasons: tuple[str, ...]
    next_step: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "next_step": self.next_step,
        }


@dataclass(frozen=True)
class DatasetEvidence:
    dataset_id: str
    name: str
    category: str | None
    license: str | None
    dimensions: dict[str, DimensionEvidence]
    auto_score: float | None
    contamination_rate: float | None
    contamination_checks: tuple[dict[str, Any], ...]
    repetition_pct: float | None
    training_score: float | None
    training_outcome: str | None
    training_reason: str | None
    last_scored: datetime | None
    complete: bool
    missing_evidence: tuple[str, ...]
    source: EvidenceSource

    def to_dict(self) -> dict[str, Any]:
        readiness = assess_readiness(self)
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "category": self.category,
            "license": self.license,
            "source": self.source,
            "complete": self.complete,
            "missing_evidence": list(self.missing_evidence),
            "last_scored": _iso(self.last_scored),
            "auto_score": self.auto_score,
            "dimensions": {
                name: self.dimensions[name].to_dict()
                for name in SCORE_DIMENSIONS
            },
            "contamination_rate": self.contamination_rate,
            "contamination_checks": list(self.contamination_checks),
            "repetition_pct": self.repetition_pct,
            "training_score": self.training_score,
            "training_outcome": self.training_outcome,
            "training_reason": self.training_reason,
            "readiness": readiness.to_dict(),
        }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def _score_value(result: dict[str, Any], dimension: str) -> float:
    block = result.get(dimension)
    if not isinstance(block, dict):
        raise ScoringFailedError(f"Live scoring omitted the {dimension} evidence block.")
    value = block.get("score")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScoringFailedError(f"Live scoring returned an invalid {dimension} score.")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ScoringFailedError(f"Live scoring returned {dimension} outside the 0-1 range.")
    return score


def _validated_live_result(result: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if result.get("error"):
        raise ScoringFailedError(str(result["error"]))

    dimensions = {name: _score_value(result, name) for name in SCORE_DIMENSIONS}
    contamination = result.get("contamination_rate")
    if not isinstance(contamination, (int, float)) or isinstance(contamination, bool):
        raise ScoringFailedError("Live scoring omitted aggregate contamination evidence.")
    if not 0.0 <= float(contamination) <= 1.0:
        raise ScoringFailedError("Live scoring returned contamination outside the 0-1 range.")

    valid_checks: list[dict[str, Any]] = []
    for check in result.get("contamination_checks", []):
        details = check.get("details", {}) if isinstance(check, dict) else {}
        overlap_rate = details.get("overlap_rate")
        if isinstance(overlap_rate, (int, float)) and not isinstance(overlap_rate, bool):
            valid_checks.append(check)
    if not valid_checks:
        raise ScoringFailedError(
            "Live scoring did not produce any verified per-benchmark contamination check."
        )
    return dimensions, valid_checks


def assess_readiness(evidence: DatasetEvidence, *, now: datetime | None = None) -> ReadinessAssessment:
    """Apply transparent, conservative gates to persisted scoring evidence."""
    blockers: list[str] = []
    reviews: list[str] = []

    if evidence.missing_evidence:
        reviews.append(
            "Evidence is incomplete: " + ", ".join(evidence.missing_evidence) + "."
        )

    scores = {
        name: dimension.score
        for name, dimension in evidence.dimensions.items()
    }
    quality = scores.get("quality")
    diversity = scores.get("diversity")
    utility = scores.get("utility")
    documentation = scores.get("documentation")
    pii_safety = scores.get("pii_safety")

    if quality is not None:
        if quality < 0.40:
            blockers.append(f"Quality {_percentage(quality)} is below the 40.0% blocker.")
        elif quality < 0.60:
            reviews.append(f"Quality {_percentage(quality)} is below the 60.0% proceed threshold.")
    if utility is not None and utility < 0.40:
        blockers.append(f"Utility {_percentage(utility)} is below the 40.0% blocker.")
    if diversity is not None and diversity < 0.15:
        reviews.append(f"Diversity {_percentage(diversity)} is below the 15.0% review threshold.")
    if documentation is not None and documentation < 0.50:
        reviews.append(
            f"Documentation {_percentage(documentation)} is below the 50.0% review threshold."
        )
    if pii_safety is not None:
        if pii_safety < 0.85:
            blockers.append(
                f"PII safety {_percentage(pii_safety)} is below the 85.0% blocker."
            )
        elif pii_safety < 0.95:
            reviews.append(
                f"PII safety {_percentage(pii_safety)} is below the 95.0% proceed threshold."
            )

    if evidence.contamination_rate is not None:
        if evidence.contamination_rate > 0.05:
            blockers.append(
                "Maximum benchmark overlap "
                f"{_percentage(evidence.contamination_rate)} exceeds the 5.0% blocker."
            )
        elif evidence.contamination_rate > 0.01:
            reviews.append(
                "Maximum benchmark overlap "
                f"{_percentage(evidence.contamination_rate)} exceeds the 1.0% review threshold."
            )

    if evidence.repetition_pct is not None:
        if evidence.repetition_pct > 50.0:
            blockers.append(
                f"Estimated repetition {evidence.repetition_pct:.1f}% exceeds the 50.0% blocker."
            )
        elif evidence.repetition_pct > 25.0:
            reviews.append(
                f"Estimated repetition {evidence.repetition_pct:.1f}% exceeds the 25.0% review threshold."
            )

    if evidence.auto_score is not None and evidence.auto_score < 0.65:
        reviews.append(
            f"Automated score {_percentage(evidence.auto_score)} is below the 65.0% proceed threshold."
        )

    license_name = (evidence.license or "").strip().lower()
    if license_name in {"", "none", "other", "unknown"}:
        reviews.append("No unambiguous dataset license is recorded.")

    training_outcome = (evidence.training_outcome or "").lower()
    if training_outcome in {"diverged", "regressed"}:
        blockers.append(
            f"The latest validated proxy-training outcome is {training_outcome}."
        )
    elif training_outcome in {"stable", "insufficient_evidence"}:
        reviews.append(
            f"The latest proxy-training outcome is {training_outcome}; it is not positive evidence."
        )

    if evidence.last_scored is not None:
        reference = now or datetime.now(timezone.utc)
        scored_at = evidence.last_scored
        if scored_at.tzinfo is None:
            scored_at = scored_at.replace(tzinfo=timezone.utc)
        if (reference - scored_at).days > 180:
            reviews.append("The cached assessment is older than 180 days and should be refreshed.")

    blockers = list(dict.fromkeys(blockers))
    reviews = list(dict.fromkeys(reviews))
    if blockers:
        return ReadinessAssessment(
            decision="DO NOT PROCEED",
            reasons=tuple(blockers + reviews),
            next_step="Remediate the blockers, refresh the assessment, and audit again before training.",
        )
    if reviews or not evidence.complete:
        return ReadinessAssessment(
            decision="REVIEW",
            reasons=tuple(reviews or ["Complete training-readiness evidence is not available."]),
            next_step="Resolve the review items and run the seven-stage audit before committing training compute.",
        )
    return ReadinessAssessment(
        decision="PROCEED",
        reasons=("Complete evidence is present and no readiness gate was triggered.",),
        next_step="Proceed to a bounded seven-stage audit and train only on the accepted output.",
    )


async def _evidence_from_session(
    session,
    dataset_id: str,
    *,
    source: EvidenceSource,
) -> DatasetEvidence | None:
    dataset = (
        await session.execute(select(DatasetRow).where(DatasetRow.hf_id == dataset_id))
    ).scalar_one_or_none()
    if dataset is None:
        return None

    leaderboard = (
        await session.execute(
            select(LeaderboardRow).where(LeaderboardRow.dataset_id == dataset.id)
        )
    ).scalar_one_or_none()
    score_rows = (
        await session.execute(
            select(ScoreRow)
            .where(ScoreRow.dataset_id == dataset.id)
            .order_by(ScoreRow.scored_at.desc(), ScoreRow.id.desc())
        )
    ).scalars().all()
    latest_scores: dict[str, ScoreRow] = {}
    for row in score_rows:
        latest_scores.setdefault(row.scorer_name, row)

    contamination_rows = (
        await session.execute(
            select(ContaminationRow)
            .where(ContaminationRow.dataset_id == dataset.id)
            .order_by(ContaminationRow.benchmark_name)
        )
    ).scalars().all()
    latest_training = (
        await session.execute(
            select(TrainingRunRow)
            .where(TrainingRunRow.dataset_id == dataset.id)
            .order_by(TrainingRunRow.trained_at.desc(), TrainingRunRow.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    missing: list[str] = []
    dimensions: dict[str, DimensionEvidence] = {}
    for name in SCORE_DIMENSIONS:
        row = latest_scores.get(name)
        value = getattr(leaderboard, name, None) if leaderboard else None
        if value is None or row is None:
            missing.append(name)
        dimensions[name] = DimensionEvidence(
            score=value,
            details=dict(row.details or {}) if row else {},
            warnings=tuple(row.warnings or []) if row else (),
            sample_size=row.sample_size if row else None,
            scored_at=row.scored_at if row else None,
        )

    auto_score = leaderboard.auto_score if leaderboard else None
    contamination_rate = leaderboard.contamination_score if leaderboard else None
    if auto_score is None:
        missing.append("auto_score")
    if contamination_rate is None:
        missing.append("contamination")
    if not contamination_rows:
        missing.append("contamination_checks")
    if dataset.last_scored is None:
        missing.append("scored_at")

    contamination_checks = tuple(
        {
            "benchmark": row.benchmark_name,
            "overlap_rate": row.overlap_rate,
            "overlap_count": row.overlap_count,
            "total_eval": row.total_eval,
            "method": row.method,
            "checked_at": _iso(row.checked_at),
        }
        for row in contamination_rows
    )
    evaluation = latest_training.eval_scores or {} if latest_training else {}
    return DatasetEvidence(
        dataset_id=dataset.hf_id,
        name=dataset.name,
        category=leaderboard.category if leaderboard else None,
        license=dataset.license,
        dimensions=dimensions,
        auto_score=auto_score,
        contamination_rate=contamination_rate,
        contamination_checks=contamination_checks,
        repetition_pct=leaderboard.repetition_pct if leaderboard else None,
        training_score=leaderboard.training_score if leaderboard else None,
        training_outcome=evaluation.get("run_outcome"),
        training_reason=evaluation.get("outcome_reason"),
        last_scored=dataset.last_scored,
        complete=not missing,
        missing_evidence=tuple(dict.fromkeys(missing)),
        source=source,
    )


async def get_cached_dataset(
    dataset_id: str,
    *,
    session_factory=None,
) -> DatasetEvidence | None:
    """Return exact-ID persisted evidence without making a network request."""
    factory = session_factory or async_session_factory
    async with factory() as session:
        return await _evidence_from_session(session, dataset_id, source="cache")


async def _persist_live_result(
    dataset_id: str,
    result: dict[str, Any],
    dimensions: dict[str, float],
    contamination_checks: list[dict[str, Any]],
    *,
    session_factory=None,
) -> None:
    factory = session_factory or async_session_factory
    scored_at = datetime.now(timezone.utc)
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    sample_size = result.get("samples_fetched")
    if not isinstance(sample_size, int):
        sample_size = None

    async with factory() as session:
        dataset = (
            await session.execute(select(DatasetRow).where(DatasetRow.hf_id == dataset_id))
        ).scalar_one_or_none()
        if dataset is None:
            dataset = DatasetRow(
                hf_id=dataset_id,
                name=str(result.get("name") or dataset_id),
            )
            session.add(dataset)
            await session.flush()

        dataset.name = str(result.get("name") or dataset.name or dataset_id)
        dataset.description = result.get("description") or dataset.description
        dataset.license = result.get("license") or metadata.get("license") or dataset.license
        dataset.tags = result.get("tags") or metadata.get("tags") or dataset.tags or []
        dataset.downloads = metadata.get("downloads", dataset.downloads)
        dataset.likes = metadata.get("likes", dataset.likes)
        dataset.created_at = metadata.get("created_at") or dataset.created_at
        dataset.source_url = dataset.source_url or f"https://huggingface.co/datasets/{dataset_id}"
        dataset.last_scored = scored_at

        leaderboard = (
            await session.execute(
                select(LeaderboardRow).where(LeaderboardRow.dataset_id == dataset.id)
            )
        ).scalar_one_or_none()
        if leaderboard is None:
            leaderboard = LeaderboardRow(dataset_id=dataset.id)
            session.add(leaderboard)

        for name, value in dimensions.items():
            setattr(leaderboard, name, value)
        leaderboard.category = result.get("category") or leaderboard.category
        leaderboard.contamination_score = float(result["contamination_rate"])
        repetition_pct = result.get("repetition_pct")
        leaderboard.repetition_pct = (
            float(repetition_pct)
            if isinstance(repetition_pct, (int, float)) and not isinstance(repetition_pct, bool)
            else None
        )
        leaderboard.auto_score = round(sum(dimensions.values()) / len(dimensions), 4)
        if leaderboard.training_score is None:
            leaderboard.combined_score = leaderboard.auto_score
        else:
            leaderboard.combined_score = round(
                (leaderboard.auto_score * 0.9) + (leaderboard.training_score * 0.1),
                4,
            )
        leaderboard.updated_at = scored_at

        await session.execute(delete(ScoreRow).where(ScoreRow.dataset_id == dataset.id))
        for name, value in dimensions.items():
            block = result[name]
            session.add(
                ScoreRow(
                    dataset_id=dataset.id,
                    scorer_name=name,
                    score=value,
                    details=block.get("details", {}),
                    warnings=block.get("warnings", []),
                    sample_size=sample_size,
                    scored_at=scored_at,
                )
            )

        await session.execute(
            delete(ContaminationRow).where(ContaminationRow.dataset_id == dataset.id)
        )
        for check in contamination_checks:
            details = check["details"]
            name = str(details.get("benchmark") or check.get("name") or "unknown")
            name = name.removeprefix("contamination_")
            session.add(
                ContaminationRow(
                    dataset_id=dataset.id,
                    benchmark_name=name,
                    overlap_rate=float(details["overlap_rate"]),
                    overlap_count=_optional_int(details.get("overlap_count")),
                    total_eval=_optional_int(
                        details.get("total_eval_ngrams", details.get("total_eval"))
                    ),
                    method=str(details.get("method") or "unknown"),
                    checked_at=scored_at,
                )
            )

        await session.flush()
        ranked = (
            await session.execute(
                select(LeaderboardRow).order_by(
                    LeaderboardRow.combined_score.desc().nullslast(),
                    LeaderboardRow.dataset_id,
                )
            )
        ).scalars().all()
        for rank, row in enumerate(ranked, start=1):
            row.rank = rank
        await session.commit()


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


async def get_or_score_dataset(
    dataset_id: str,
    *,
    sample_size: int = 10_000,
    refresh: bool = False,
    cached_only: bool = False,
    token: str | None = None,
    scorer: Scorer | None = None,
    session_factory=None,
) -> DatasetEvidence:
    """Reuse complete exact-ID evidence, or score and persist a new assessment."""
    if refresh and cached_only:
        raise DatasetIntelligenceError("--refresh and --cached-only cannot be used together.")

    cached = await get_cached_dataset(dataset_id, session_factory=session_factory)
    if cached is not None and cached.complete and not refresh:
        return cached
    if cached_only:
        if cached is None:
            raise CacheMissError(f"No cached assessment exists for {dataset_id}.")
        raise CacheMissError(
            f"Cached assessment for {dataset_id} is incomplete: "
            + ", ".join(cached.missing_evidence)
            + "."
        )

    if scorer is None:
        from socbench.runner import run_socbench_scoring

        scorer = run_socbench_scoring
    try:
        result = await scorer(dataset_id, sample_size=sample_size, token=token)
    except DatasetIntelligenceError:
        raise
    except Exception as exc:
        raise ScoringFailedError(
            f"Live scoring failed for {dataset_id}: {type(exc).__name__}: {exc}"
        ) from exc

    dimensions, contamination_checks = _validated_live_result(result)
    await _persist_live_result(
        dataset_id,
        result,
        dimensions,
        contamination_checks,
        session_factory=session_factory,
    )
    persisted = await get_cached_dataset(dataset_id, session_factory=session_factory)
    if persisted is None or not persisted.complete:
        missing = "unknown" if persisted is None else ", ".join(persisted.missing_evidence)
        raise ScoringFailedError(
            f"The assessment was not persisted as complete evidence ({missing})."
        )
    return DatasetEvidence(**{**persisted.__dict__, "source": "live"})


def rank_comparison(evidence: Iterable[DatasetEvidence]) -> list[DatasetEvidence]:
    """Order comparison rows by decision, then automated score, without hiding gaps."""
    decision_order = {"PROCEED": 0, "REVIEW": 1, "DO NOT PROCEED": 2}
    return sorted(
        evidence,
        key=lambda item: (
            decision_order[assess_readiness(item).decision],
            -(item.auto_score if item.auto_score is not None else -1.0),
            item.dataset_id.lower(),
        ),
    )
