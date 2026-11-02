import numpy as np
import pytest

from trusttab.evaluation import evaluate_classifier, expected_calibration_error


def test_evaluate_classifier_returns_expected_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.7, 0.9])

    result = evaluate_classifier(y_true, y_pred, y_prob)

    assert set(result) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "brier_score",
        "ece",
    }
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["balanced_accuracy"] == pytest.approx(0.75)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["brier_score"] == pytest.approx(0.0675)
    assert all(0.0 <= value <= 1.0 for value in result.values())


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    y_true = np.array([0, 1])
    y_prob = np.array([0.0, 1.0])

    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0)


def test_ece_includes_confidence_one_in_last_bin():
    y_true = np.array([1])
    y_prob = np.array([1.0])

    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0)


def test_ece_rejects_invalid_bin_count():
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.2, 0.8], n_bins=0)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "y_prob"),
    [
        ([0, 1], [0], [0.2, 0.8]),
        ([0, 1], [0, 1], [0.2]),
        ([0, 1], [0, 1], [-0.1, 0.8]),
        ([0, 1], [0, 1], [0.2, 1.1]),
        ([0, 1], [0, 1], [0.2, float("nan")]),
        ([], [], []),
    ],
)
def test_evaluate_classifier_rejects_invalid_inputs(y_true, y_pred, y_prob):
    with pytest.raises(ValueError):
        evaluate_classifier(y_true, y_pred, y_prob)


def test_roc_auc_requires_both_classes():
    with pytest.raises(ValueError):
        evaluate_classifier([1, 1], [1, 1], [0.8, 0.9])
