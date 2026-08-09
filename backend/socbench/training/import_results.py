"""Import validated GPT-2 proxy-training result artifacts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from socbench.training.outcomes import classify_training_curve

DEFAULT_PATTERNS = (
    "training_results/validated/*_v*/result.json",
)


@dataclass(frozen=True)
class TrainingArtifact:
    dataset_id: str
    path: str
    campaign_version: int
    n_tokens: int
    n_samples: int | None
    parameters: int | None
    max_iters: int
    final_val_loss: float
    best_val_loss: float
    loss_curve: list[float]
    loss_steps: list[int]
    convergence_steps: int
    completed_steps: int
    eval_scores: dict
    model_config: dict
    trained_at: str | None


@dataclass(frozen=True)
class ImportPlan:
    selected: dict[str, TrainingArtifact]
    incomplete: list[tuple[str, str]]
    orphans: list[TrainingArtifact]
    scores: dict[str, float]


def discover_result_files(root: Path, patterns: Iterable[str] = DEFAULT_PATTERNS) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted({path.resolve() for path in files if path.name != "summary.json"})


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _campaign_version(path: Path) -> int:
    for part in path.parts:
        if "_v" not in part:
            continue
        suffix = part.rsplit("_v", 1)[-1]
        digits = []
        for char in suffix:
            if not char.isdigit():
                break
            digits.append(char)
        if digits:
            return int("".join(digits))
    if path.name.endswith(".json") and "_v" in path.stem:
        suffix = path.stem.rsplit("_v", 1)[-1]
        digits = "".join(char for char in suffix if char.isdigit())
        if digits:
            return int(digits)
    return 0


def _extract_marker_json(text: str) -> dict | None:
    marker = "SOCBENCH_RESULT_JSON="
    if marker not in text:
        return None
    for line in text.splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    try:
        events = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict):
            continue
        data = event.get("data")
        if not isinstance(data, str) or marker not in data:
            continue
        for line in data.splitlines():
            if marker not in line:
                continue
            payload = line.split(marker, 1)[1].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                continue
    return None


def _normalize_marker_summary(summary: dict) -> dict:
    loss_data = summary.get("loss_curve.json")
    eval_data = summary.get("eval_results.json")
    hardware = summary.get("hardware")
    if not isinstance(loss_data, dict):
        loss_data = {}
    if not isinstance(eval_data, dict):
        eval_data = {}
    if not isinstance(hardware, dict):
        hardware = {}
    gpu_names = hardware.get("gpu_names")
    if not isinstance(gpu_names, list):
        gpu_names = []

    return {
        "dataset_id": summary.get("dataset_id"),
        "n_tokens": loss_data.get("n_tokens") or loss_data.get("total_tokens") or summary.get("tokens_budget"),
        "n_samples": summary.get("n_samples"),
        "parameters": summary.get("parameters"),
        "max_iters": loss_data.get("max_iters") or loss_data.get("total_iters"),
        "final_val_loss": eval_data.get("final_val_loss") or loss_data.get("final_val_loss"),
        "best_val_loss": eval_data.get("best_val_loss") or loss_data.get("best_val_loss"),
        "gpu": summary.get("gpu") or ", ".join(str(name) for name in gpu_names) or None,
        "pytorch_version": summary.get("pytorch_version"),
        "batch_size": summary.get("batch_size"),
        "use_fp16": summary.get("use_fp16"),
        "num_gpus": summary.get("num_gpus") or hardware.get("cuda_device_count") or 1,
        "distributed_world_size": hardware.get("distributed_world_size"),
        "launcher": hardware.get("launcher"),
        "loss_curve": loss_data.get("loss_curve"),
    }


def load_training_artifact(path: Path, root: Path) -> tuple[TrainingArtifact | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable artifact: {exc}"

    if path.suffix.lower() == ".log":
        marker_summary = _extract_marker_json(text)
        if marker_summary is None:
            return None, "missing SOCBENCH_RESULT_JSON marker"
        data = _normalize_marker_summary(marker_summary)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"unreadable JSON: {exc}"
        if "loss_curve.json" in data or "eval_results.json" in data:
            data = _normalize_marker_summary(data)

    dataset_id = data.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return None, "missing dataset_id"

    raw_curve = data.get("loss_curve")
    if not isinstance(raw_curve, list) or len(raw_curve) < 1:
        return None, "missing comparable loss_curve"

    curve_values: list[float] = []
    loss_steps: list[int] = []
    for point in raw_curve:
        if not isinstance(point, dict):
            return None, "loss_curve entries must be objects"
        val_loss = _as_float(point.get("val_loss"))
        if val_loss is None:
            return None, "loss_curve entry missing finite val_loss"
        curve_values.append(val_loss)
        step = _as_int(point.get("step"))
        if step is None:
            return None, "loss_curve entry missing integer step"
        loss_steps.append(step)

    final_val_loss = _as_float(data.get("final_val_loss"))
    best_val_loss = _as_float(data.get("best_val_loss"))
    max_iters = _as_int(data.get("max_iters"))
    n_tokens = _as_int(data.get("n_tokens"))
    if final_val_loss is None:
        return None, "missing finite final_val_loss"
    if best_val_loss is None:
        best_val_loss = min(curve_values)
    if max_iters is None or max_iters <= 0:
        return None, "missing positive max_iters"
    if n_tokens is None or n_tokens <= 0:
        return None, "missing positive n_tokens"
    completed_steps = loss_steps[-1]
    if completed_steps + 1 < max_iters * 0.8:
        return None, "run did not reach at least 80% of max_iters"

    outcome = classify_training_curve(curve_values, loss_steps)
    if not math.isclose(final_val_loss, outcome.final_val_loss, rel_tol=1e-6, abs_tol=1e-6):
        return None, "final_val_loss does not match loss_curve"
    claimed_outcome = data.get("run_outcome")
    if claimed_outcome is not None and claimed_outcome != outcome.run_outcome:
        return None, "run_outcome does not match loss_curve"
    claimed_relative = _as_float(data.get("relative_improvement"))
    if claimed_relative is not None and not math.isclose(
        claimed_relative, outcome.relative_improvement, rel_tol=1e-6, abs_tol=1e-6
    ):
        return None, "relative_improvement does not match loss_curve"

    if data.get("run_type") == "real":
        evidence = data.get("evidence_sha256")
        if not isinstance(evidence, dict) or not evidence:
            return None, "real run is missing evidence_sha256"
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
            for digest in evidence.values()
        ):
            return None, "real run has an invalid evidence SHA-256 digest"
        proof_path = path.with_name("eval_results.json")
        if not proof_path.exists():
            return None, "real run is missing eval_results.json"
        try:
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"unreadable eval_results.json: {exc}"
        for key, expected in {
            "dataset_id": dataset_id,
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_loss,
        }.items():
            actual = proof.get(key)
            if isinstance(expected, float):
                if _as_float(actual) is None or not math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-6):
                    return None, f"eval_results.json has mismatched {key}"
            elif actual != expected:
                return None, f"eval_results.json has mismatched {key}"

    n_samples = _as_int(data.get("n_samples"))
    parameters = _as_int(data.get("parameters"))
    rel_path = str(path.resolve().relative_to(root.resolve()))
    model_config = {
        "parameters": parameters,
        "max_iters": max_iters,
        "gpu": data.get("gpu"),
        "pytorch_version": data.get("pytorch_version"),
        "batch_size": data.get("batch_size"),
        "gradient_accumulation_steps": data.get("gradient_accumulation_steps"),
        "effective_batch_size": data.get("effective_batch_size"),
        "use_fp16": data.get("use_fp16"),
        "num_gpus": data.get("num_gpus", 1),
        "distributed_world_size": data.get("distributed_world_size"),
        "launcher": data.get("launcher"),
        "run_type": data.get("run_type"),
        "evidence_sha256": data.get("evidence_sha256"),
        "completed_at": data.get("completed_at"),
        "n_samples": n_samples,
        "source_artifact": rel_path,
        "campaign_version": _campaign_version(path),
        "selection_metric": "latest_campaign_then_lowest_best_val_loss",
        "completed_steps": completed_steps,
    }

    return TrainingArtifact(
        dataset_id=dataset_id.strip(),
        path=rel_path,
        campaign_version=_campaign_version(path),
        n_tokens=n_tokens,
        n_samples=n_samples,
        parameters=parameters,
        max_iters=max_iters,
        final_val_loss=final_val_loss,
        best_val_loss=best_val_loss,
        loss_curve=curve_values,
        loss_steps=loss_steps,
        convergence_steps=outcome.best_step,
        completed_steps=completed_steps,
        eval_scores={
            **outcome.as_dict(),
            "best_val_loss": best_val_loss,
            "best_relative_improvement": (outcome.initial_val_loss - best_val_loss)
            / outcome.initial_val_loss,
            "source_artifact": rel_path,
            "campaign_version": _campaign_version(path),
        },
        model_config=model_config,
        trained_at=data.get("completed_at") or data.get("trained_at"),
    ), None


def score_selected(selected: dict[str, TrainingArtifact]) -> dict[str, float]:
    return {
        dataset_id: artifact.eval_scores["training_score"]
        for dataset_id, artifact in selected.items()
    }


def _recompute_all_training_scores(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT
            t.id AS training_run_id,
            t.dataset_id,
            t.eval_scores,
            t.loss_curve,
            t.model_config,
            l.*
        FROM training_runs t
        JOIN leaderboard l ON l.dataset_id = t.dataset_id
        """
    ).fetchall()
    for row in rows:
        try:
            eval_scores = json.loads(row["eval_scores"] or "{}")
        except (TypeError, json.JSONDecodeError):
            eval_scores = {}
        try:
            curve = json.loads(row["loss_curve"] or "[]")
        except (TypeError, json.JSONDecodeError):
            curve = []
        curve_values = [_as_float(value) for value in curve]
        finite_values = [value for value in curve_values if value is not None]
        if not finite_values:
            continue
        completed_steps = _as_int((json.loads(row["model_config"] or "{}") or {}).get("completed_steps"))
        steps = list(range(len(finite_values)))
        if completed_steps is not None and len(steps) > 1:
            steps = [round(i * completed_steps / (len(steps) - 1)) for i in range(len(steps))]
        outcome = classify_training_curve(finite_values, steps)
        eval_scores.update(outcome.as_dict())
        score = outcome.training_score
        conn.execute(
            "UPDATE training_runs SET eval_scores = ? WHERE id = ?",
            (_json(eval_scores), row["training_run_id"]),
        )
        conn.execute(
            """
            UPDATE leaderboard
            SET training_score = ?, combined_score = ?, updated_at = CURRENT_TIMESTAMP
            WHERE dataset_id = ?
            """,
            (score, _combined_score(row, score), row["dataset_id"]),
        )


def build_import_plan(root: Path, db_path: Path, include_orphans: bool = False) -> ImportPlan:
    loaded: dict[str, TrainingArtifact] = {}
    incomplete: list[tuple[str, str]] = []
    for path in discover_result_files(root):
        artifact, reason = load_training_artifact(path, root)
        if artifact is None:
            incomplete.append((str(path.resolve().relative_to(root.resolve())), reason or "unknown"))
            continue
        current = loaded.get(artifact.dataset_id)
        if (
            current is None
            or artifact.campaign_version > current.campaign_version
            or (
                artifact.campaign_version == current.campaign_version
                and artifact.best_val_loss < current.best_val_loss
            )
        ):
            loaded[artifact.dataset_id] = artifact

    conn = sqlite3.connect(db_path)
    try:
        db_ids = {row[0] for row in conn.execute("SELECT hf_id FROM datasets")}
    finally:
        conn.close()

    selected = {
        dataset_id: artifact
        for dataset_id, artifact in loaded.items()
        if include_orphans or dataset_id in db_ids
    }
    orphans = [
        artifact
        for dataset_id, artifact in sorted(loaded.items())
        if dataset_id not in db_ids
    ]
    return ImportPlan(
        selected=selected,
        incomplete=incomplete,
        orphans=orphans,
        scores=score_selected(selected),
    )


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _loss_stability(loss_curve: list[float]) -> float:
    if len(loss_curve) < 2:
        return 0.0
    tail = loss_curve[max(0, int(len(loss_curve) * 0.8)) :]
    if len(tail) < 2:
        return 0.0
    mean = sum(tail) / len(tail)
    return (sum((value - mean) ** 2 for value in tail) / len(tail)) ** 0.5


def _combined_score(row: sqlite3.Row, training_score: float) -> float:
    auto_score = row["auto_score"]
    if auto_score is None:
        dims = [
            row["quality"],
            row["diversity"],
            row["utility"],
            row["documentation"],
            row["popularity"],
            row["freshness"],
            row["pii_safety"],
        ]
        present = [value for value in dims if value is not None]
        auto_score = sum(present) / len(present) if present else 0.0
    return round((float(auto_score) * 0.9) + (training_score * 0.1), 4)


def apply_import_plan(plan: ImportPlan, db_path: Path, create_missing_datasets: bool = False) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        changed = 0
        for dataset_id, artifact in sorted(plan.selected.items()):
            score = plan.scores[dataset_id]
            ds = conn.execute(
                "SELECT id FROM datasets WHERE hf_id = ?",
                (dataset_id,),
            ).fetchone()
            if ds is None:
                if not create_missing_datasets:
                    continue
                conn.execute(
                    """
                    INSERT INTO datasets (hf_id, name, description, tags)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        dataset_id,
                        "Recovered from a validated external training artifact; automated Socbench scoring pending.",
                        _json(["recovered-training-artifact"]),
                    ),
                )
                ds = conn.execute(
                    "SELECT id FROM datasets WHERE hf_id = ?",
                    (dataset_id,),
                ).fetchone()
                if ds is None:
                    continue
            dataset_pk = ds["id"]
            eval_scores = artifact.eval_scores
            existing = conn.execute(
                "SELECT id FROM training_runs WHERE dataset_id = ? ORDER BY id",
                (dataset_pk,),
            ).fetchall()
            values = (
                artifact.final_val_loss,
                _json(artifact.loss_curve),
                artifact.convergence_steps,
                artifact.n_tokens,
                _json(artifact.model_config),
                _json(eval_scores),
                _loss_stability(artifact.loss_curve),
                artifact.trained_at,
                dataset_pk,
            )
            if existing:
                conn.execute(
                    """
                    UPDATE training_runs
                    SET final_val_loss = ?, loss_curve = ?, convergence_steps = ?,
                        tokens_seen = ?, model_config = ?, eval_scores = ?,
                        loss_stability = ?, trained_at = COALESCE(?, trained_at)
                    WHERE dataset_id = ?
                    """,
                    values,
                )
            else:
                conn.execute(
                    """
                    INSERT INTO training_runs (
                        final_val_loss, loss_curve, convergence_steps, tokens_seen,
                        model_config, eval_scores, loss_stability, trained_at, dataset_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?)
                    """,
                    values,
                )
            if len(existing) > 1:
                duplicate_ids = [row["id"] for row in existing[1:]]
                conn.executemany(
                    "DELETE FROM training_runs WHERE id = ?",
                    [(row_id,) for row_id in duplicate_ids],
                )

            lb = conn.execute(
                "SELECT * FROM leaderboard WHERE dataset_id = ?",
                (dataset_pk,),
            ).fetchone()
            if lb is not None:
                conn.execute(
                    """
                    UPDATE leaderboard
                    SET training_score = ?, combined_score = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE dataset_id = ?
                    """,
                    (score, _combined_score(lb, score), dataset_pk),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO leaderboard (dataset_id, category, training_score, combined_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (dataset_pk, "posttraining-sft", score, round(score * 0.1, 4)),
                )
            changed += 1

        _recompute_all_training_scores(conn)

        entries = conn.execute(
            "SELECT id FROM leaderboard ORDER BY combined_score DESC NULLS LAST"
        ).fetchall()
        for rank, row in enumerate(entries, start=1):
            conn.execute("UPDATE leaderboard SET rank = ? WHERE id = ?", (rank, row["id"]))
        conn.commit()
        return changed
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def backup_db(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_suffix(f".before-training-import-{stamp}.db")
    shutil.copy2(db_path, backup)
    return backup


def print_plan(plan: ImportPlan) -> None:
    print(f"selected complete runs: {len(plan.selected)}")
    for dataset_id, artifact in sorted(plan.selected.items()):
        score = plan.scores.get(dataset_id)
        print(
            f"  {dataset_id}: v{artifact.campaign_version} score={score:.4f} best={artifact.best_val_loss:.4f} "
            f"final={artifact.final_val_loss:.4f} artifact={artifact.path}"
        )
    print(f"orphaned complete runs: {len(plan.orphans)}")
    for artifact in plan.orphans:
        print(
            f"  {artifact.dataset_id}: v{artifact.campaign_version} "
            f"best={artifact.best_val_loss:.4f} artifact={artifact.path}"
        )
    print(f"incomplete/skipped artifacts: {len(plan.incomplete)}")
    for path, reason in plan.incomplete:
        print(f"  {path}: {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve().parents[2] / "socbench.db")
    parser.add_argument(
        "--include-orphans",
        action="store_true",
        help="include complete training artifacts whose dataset rows are missing",
    )
    parser.add_argument(
        "--create-missing-datasets",
        action="store_true",
        help="create minimal dataset rows for included orphaned training artifacts",
    )
    parser.add_argument("--apply", action="store_true", help="write recovered runs into the DB")
    args = parser.parse_args(argv)

    plan = build_import_plan(args.root, args.db, include_orphans=args.include_orphans)
    print_plan(plan)
    if not args.apply:
        print("dry run only; pass --apply to write the DB")
        return 0
    backup = backup_db(args.db)
    changed = apply_import_plan(
        plan,
        args.db,
        create_missing_datasets=args.create_missing_datasets,
    )
    print(f"backed up DB to {backup}")
    print(f"imported/updated {changed} training runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
