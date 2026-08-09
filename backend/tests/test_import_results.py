import json
import sqlite3
from pathlib import Path

from socbench.training.import_results import (
    apply_import_plan,
    build_import_plan,
    load_training_artifact,
)


def _write_result(path: Path, dataset_id: str, best: float, final: float, max_iters: int = 50):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "n_tokens": 123456,
                "n_samples": 1000,
                "parameters": 124439808,
                "max_iters": max_iters,
                "final_val_loss": final,
                "best_val_loss": best,
                "gpu": "Tesla T4",
                "pytorch_version": "2.4.0+cu118",
                "batch_size": 8,
                "use_fp16": False,
                "loss_curve": [
                    {"step": 0, "train_loss": 10.0, "val_loss": 10.0},
                    {"step": max_iters - 1, "train_loss": 1.0, "val_loss": final},
                ],
            }
        ),
        encoding="utf-8",
    )


def _make_db(path: Path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hf_id VARCHAR(512) NOT NULL,
            name VARCHAR(512) NOT NULL,
            description TEXT,
            tags JSON
        );
        CREATE TABLE leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            category VARCHAR(64),
            auto_score FLOAT,
            quality FLOAT,
            diversity FLOAT,
            utility FLOAT,
            documentation FLOAT,
            popularity FLOAT,
            freshness FLOAT,
            pii_safety FLOAT,
            training_score FLOAT,
            combined_score FLOAT,
            rank INTEGER,
            updated_at DATETIME
        );
        CREATE TABLE training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            model_config JSON,
            tokens_seen INTEGER,
            final_val_loss FLOAT,
            loss_curve JSON,
            convergence_steps INTEGER,
            loss_stability FLOAT,
            eval_scores JSON,
            gpu_hours FLOAT,
            trained_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute("INSERT INTO datasets (hf_id, name) VALUES ('dataset/a', 'dataset/a')")
    conn.execute("INSERT INTO datasets (hf_id, name) VALUES ('dataset/b', 'dataset/b')")
    conn.execute(
        "INSERT INTO leaderboard (dataset_id, auto_score, combined_score) VALUES (1, 0.8, 0.8)"
    )
    conn.execute(
        "INSERT INTO leaderboard (dataset_id, auto_score, combined_score) VALUES (2, 0.4, 0.4)"
    )
    conn.commit()
    conn.close()


def test_load_training_artifact_rejects_smoke_schema(tmp_path):
    root = tmp_path
    path = root / "backend" / "training_outputs" / "x" / "loss_curve.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"losses": [10.0], "best_val_loss": 10.0, "total_iters": 1}),
        encoding="utf-8",
    )

    artifact, reason = load_training_artifact(path, root)

    assert artifact is None
    assert reason == "missing dataset_id"


def test_load_training_artifact_accepts_log_marker(tmp_path):
    root = tmp_path
    path = root / "backend" / "training_outputs" / "dataset-a" / "runner.log"
    path.parent.mkdir(parents=True)
    summary = {
        "dataset_id": "dataset/a",
        "tokens_budget": 123456,
        "loss_curve.json": {
            "n_tokens": 123456,
            "max_iters": 10,
            "best_val_loss": 4.0,
            "final_val_loss": 4.2,
            "loss_curve": [
                {"step": 0, "train_loss": None, "val_loss": 5.0},
                {"step": 9, "train_loss": None, "val_loss": 4.2},
            ],
        },
        "eval_results.json": {
            "best_val_loss": 4.0,
            "final_val_loss": 4.2,
        },
    }
    path.write_text(
        json.dumps([{"stream_name": "stderr", "data": "SOCBENCH_RESULT_JSON=" + json.dumps(summary) + "\n"}]),
        encoding="utf-8",
    )

    artifact, reason = load_training_artifact(path, root)

    assert reason is None
    assert artifact is not None
    assert artifact.dataset_id == "dataset/a"
    assert artifact.n_tokens == 123456
    assert artifact.final_val_loss == 4.2


def test_build_plan_accepts_result_summary(tmp_path):
    root = tmp_path
    db = tmp_path / "socbench.db"
    _make_db(db)
    path = root / "backend" / "training_outputs" / "dataset-a" / "socbench_result.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": "dataset/a",
                "tokens_budget": 123456,
                "loss_curve.json": {
                    "n_tokens": 123456,
                    "max_iters": 10,
                    "best_val_loss": 4.0,
                    "final_val_loss": 4.2,
                    "loss_curve": [
                        {"step": 0, "train_loss": None, "val_loss": 5.0},
                        {"step": 9, "train_loss": None, "val_loss": 4.2},
                    ],
                },
                "eval_results.json": {
                    "best_val_loss": 4.0,
                    "final_val_loss": 4.2,
                },
            }
        ),
        encoding="utf-8",
    )

    plan = build_import_plan(root, db)

    assert set(plan.selected) == {"dataset/a"}
    assert plan.selected["dataset/a"].path.endswith("socbench_result.json")
    assert plan.selected["dataset/a"].n_tokens == 123456


def test_load_training_artifact_accepts_single_step_comparable_curve(tmp_path):
    root = tmp_path
    path = root / "backend" / "training_outputs" / "dataset-a" / "loss_curve.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": "dataset/a",
                "n_tokens": 1000000,
                "max_iters": 1,
                "final_val_loss": 10.9,
                "best_val_loss": 10.9,
                "loss_curve": [{"step": 0, "train_loss": None, "val_loss": 10.9}],
            }
        ),
        encoding="utf-8",
    )

    artifact, reason = load_training_artifact(path, root)

    assert reason is None
    assert artifact is not None
    assert artifact.convergence_steps == 0


def test_load_training_artifact_preserves_distributed_evidence(tmp_path):
    root = tmp_path
    path = root / "training_results" / "validated" / "dataset-a_v23" / "result.json"
    _write_result(path, "dataset/a", 4.0, 4.2)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(
        {
            "run_type": "real",
            "completed_at": "2026-08-09T05:51:15+00:00",
            "num_gpus": 2,
            "distributed_world_size": 2,
            "launcher": "torch.distributed.run",
            "gradient_accumulation_steps": 64,
            "effective_batch_size": 512,
            "evidence_sha256": {"training_log": "abc123"},
        }
    )
    path.write_text(json.dumps(data), encoding="utf-8")

    artifact, reason = load_training_artifact(path, root)

    assert reason is None
    assert artifact is not None
    assert artifact.model_config["run_type"] == "real"
    assert artifact.trained_at == "2026-08-09T05:51:15+00:00"
    assert artifact.model_config["num_gpus"] == 2
    assert artifact.model_config["distributed_world_size"] == 2
    assert artifact.model_config["launcher"] == "torch.distributed.run"
    assert artifact.model_config["gradient_accumulation_steps"] == 64
    assert artifact.model_config["effective_batch_size"] == 512
    assert artifact.model_config["evidence_sha256"]["training_log"] == "abc123"


def test_build_plan_selects_best_complete_run_and_reports_orphans(tmp_path):
    db = tmp_path / "socbench.db"
    _make_db(db)
    _write_result(tmp_path / "training_results" / "validated" / "a_v21" / "result.json", "dataset/a", 5.0, 5.5)
    _write_result(tmp_path / "training_results" / "validated" / "a_v22" / "result.json", "dataset/a", 4.0, 4.2)
    _write_result(tmp_path / "training_results" / "validated" / "orphan_v22" / "result.json", "dataset/missing", 3.0, 3.2)

    plan = build_import_plan(tmp_path, db)

    assert set(plan.selected) == {"dataset/a"}
    assert plan.selected["dataset/a"].best_val_loss == 4.0
    assert [artifact.dataset_id for artifact in plan.orphans] == ["dataset/missing"]


def test_build_plan_prefers_latest_campaign_before_lower_loss(tmp_path):
    db = tmp_path / "socbench.db"
    _make_db(db)
    _write_result(tmp_path / "training_results" / "validated" / "a_v21" / "result.json", "dataset/a", 3.0, 3.1)
    _write_result(tmp_path / "training_results" / "validated" / "a_v22" / "result.json", "dataset/a", 4.0, 4.2)

    plan = build_import_plan(tmp_path, db)

    assert plan.selected["dataset/a"].campaign_version == 22
    assert plan.selected["dataset/a"].best_val_loss == 4.0


def test_apply_import_plan_is_idempotent_and_preserves_auto_score(tmp_path):
    db = tmp_path / "socbench.db"
    _make_db(db)
    _write_result(tmp_path / "training_results" / "validated" / "a_v22" / "result.json", "dataset/a", 4.0, 4.2)
    _write_result(tmp_path / "training_results" / "validated" / "b_v22" / "result.json", "dataset/b", 8.0, 8.1)
    plan = build_import_plan(tmp_path, db)

    assert apply_import_plan(plan, db) == 2
    assert apply_import_plan(plan, db) == 2

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT dataset_id, final_val_loss FROM training_runs ORDER BY dataset_id").fetchall()
    leaderboard = conn.execute("SELECT dataset_id, training_score, combined_score FROM leaderboard ORDER BY dataset_id").fetchall()
    conn.close()

    assert rows == [(1, 4.2), (2, 8.1)]
    assert len(leaderboard) == 2
    assert leaderboard[0][1] == 1.0
    assert leaderboard[0][2] == 0.82
    assert leaderboard[1][1] == 0.0
    assert leaderboard[1][2] == 0.36


def test_apply_import_plan_can_create_training_only_dataset_rows(tmp_path):
    db = tmp_path / "socbench.db"
    _make_db(db)
    _write_result(tmp_path / "training_results" / "validated" / "orphan_v22" / "result.json", "dataset/missing", 4.0, 4.2)
    plan = build_import_plan(tmp_path, db, include_orphans=True)

    assert apply_import_plan(plan, db, create_missing_datasets=True) == 1

    conn = sqlite3.connect(db)
    row = conn.execute(
        """
        SELECT d.hf_id, l.category, t.final_val_loss
        FROM datasets d
        JOIN leaderboard l ON l.dataset_id = d.id
        JOIN training_runs t ON t.dataset_id = d.id
        WHERE d.hf_id = 'dataset/missing'
        """
    ).fetchone()
    conn.close()

    assert row == ("dataset/missing", "posttraining-sft", 4.2)


def test_apply_import_plan_recomputes_scores_for_existing_training_rows(tmp_path):
    db = tmp_path / "socbench.db"
    _make_db(db)
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO training_runs (dataset_id, final_val_loss, loss_curve, eval_scores)
        VALUES (2, 8.1, '[8.0, 8.1]', '{"best_val_loss": 8.0}')
        """
    )
    conn.commit()
    conn.close()
    _write_result(
        tmp_path / "training_results" / "validated" / "a_v23" / "result.json",
        "dataset/a",
        4.0,
        4.2,
    )

    plan = build_import_plan(tmp_path, db)
    assert apply_import_plan(plan, db) == 1

    conn = sqlite3.connect(db)
    leaderboard = conn.execute(
        "SELECT dataset_id, training_score FROM leaderboard ORDER BY dataset_id"
    ).fetchall()
    existing_eval = json.loads(
        conn.execute("SELECT eval_scores FROM training_runs WHERE dataset_id = 2").fetchone()[0]
    )
    conn.close()

    assert leaderboard == [(1, 1.0), (2, 0.0)]
    assert existing_eval["training_score"] == 0.0
    assert "all current complete training runs" in existing_eval["normalization"]
