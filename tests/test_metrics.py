import numpy as np
import pytest

from trusttab.evaluation import (
    adaptive_expected_calibration_error,
    evaluate_classifier,
    expected_calibration_error,
    log_loss_score,
    maximum_calibration_error,
)


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
        "maximum_calibration_error",
        "adaptive_ece",
        "log_loss",
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


def test_mce_calculates_maximum_equal_width_bin_error():
    assert maximum_calibration_error(
        [0, 1, 1, 0], [0.1, 0.4, 0.8, 0.9], n_bins=2
    ) == pytest.approx(0.3)


def test_adaptive_ece_uses_weighted_frequency_bins():
    assert adaptive_expected_calibration_error(
        [0, 1, 1, 0], [0.1, 0.4, 0.8, 0.9], n_bins=2
    ) == pytest.approx(0.3)


def test_adaptive_ece_keeps_tied_confidences_together():
    # All four confidences are 0.8, so ties produce one non-empty bin.
    assert adaptive_expected_calibration_error(
        [0, 0, 0, 0], [0.8, 0.2, 0.8, 0.2], n_bins=3
    ) == pytest.approx(0.3)


def test_log_loss_matches_binary_cross_entropy():
    assert log_loss_score([0, 1], [0.2, 0.8]) == pytest.approx(
        -np.log(0.8)
    )


def test_log_loss_handles_zero_and_one_probabilities():
    assert log_loss_score([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    assert np.isfinite(log_loss_score([0, 1], [1.0, 0.0]))


def test_reliability_metrics_for_well_and_poorly_calibrated_predictions():
    assert maximum_calibration_error([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    assert adaptive_expected_calibration_error(
        [0, 1], [0.0, 1.0]
    ) == pytest.approx(0.0)
    assert maximum_calibration_error([0, 0], [0.9, 0.9]) == pytest.approx(0.9)
    assert adaptive_expected_calibration_error(
        [0, 0], [0.9, 0.9]
    ) == pytest.approx(0.9)


@pytest.mark.parametrize(
    "metric",
    [maximum_calibration_error, adaptive_expected_calibration_error],
)
@pytest.mark.parametrize("n_bins", [0, -1, 1.5, True, np.bool_(True)])
def test_calibration_metrics_reject_invalid_bin_counts(metric, n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        metric([0, 1], [0.2, 0.8], n_bins=n_bins)


@pytest.mark.parametrize(
    "metric",
    [
        maximum_calibration_error,
        adaptive_expected_calibration_error,
        log_loss_score,
    ],
)
@pytest.mark.parametrize(
    ("y_true", "y_prob"),
    [
        ([0, 2], [0.2, 0.8]),
        ([0, 1], [0.2, float("nan")]),
        ([0, 1], [0.2, float("inf")]),
        ([0, 1], [-0.1, 0.8]),
        ([0, 1], [0.2, 1.1]),
        ([], []),
        ([0, 1], [0.2]),
        ([[0], [1]], [0.2, 0.8]),
        ([0, 1], [[0.2], [0.8]]),
    ],
)
def test_new_metrics_reject_invalid_inputs(metric, y_true, y_prob):
    with pytest.raises(ValueError):
        metric(y_true, y_prob)


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
