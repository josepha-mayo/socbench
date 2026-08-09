import pytest

from socbench.training.outcomes import classify_training_curve


@pytest.mark.parametrize(
    ("curve", "expected", "score"),
    [
        ([10.0, 8.0], "improved", 0.2),
        ([10.0, 9.95], "stable", 0.0),
        ([10.0, 10.2], "regressed", 0.0),
        ([10.0, 11.0], "diverged", 0.0),
        ([10.0], "insufficient_evidence", 0.0),
    ],
)
def test_classify_training_curve(curve, expected, score):
    result = classify_training_curve(curve)

    assert result.run_outcome == expected
    assert result.training_score == pytest.approx(score)


def test_classify_training_curve_records_best_step_without_hiding_final_regression():
    result = classify_training_curve([10.0, 8.0, 11.0], [0, 50, 100])

    assert result.best_step == 50
    assert result.best_relative_improvement == pytest.approx(0.2)
    assert result.run_outcome == "diverged"
    assert result.training_score == 0.0
