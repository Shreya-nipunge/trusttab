"""Tests for probability calibration utilities."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trusttab.evaluation.calibration import (
    calibrate_classifier,
    calibrate_fitted_classifier,
    get_calibration_curve,
)


def make_data():
    X, y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        random_state=42,
    )
    return X, y


def test_calibrate_classifier_predicts_probabilities():
    X, y = make_data()
    model = calibrate_classifier(
        LogisticRegression(max_iter=1000),
        method="sigmoid",
        cv=3,
    )
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(y), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(probabilities).all()


def test_invalid_calibration_method_raises():
    with pytest.raises(ValueError, match="method"):
        calibrate_classifier(LogisticRegression(), method="invalid")


def test_invalid_cv_raises():
    with pytest.raises(ValueError, match="cv"):
        calibrate_classifier(LogisticRegression(), cv=1)


def test_calibrate_fitted_classifier():
    X, y = make_data()
    base = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    base.fit(X[:80], y[:80])

    calibrated = calibrate_fitted_classifier(
        base,
        X[80:],
        y[80:],
        method="sigmoid",
    )
    probabilities = calibrated.predict_proba(X[80:])

    assert probabilities.shape == (40, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_fitted_calibration_does_not_refit_base_estimator():
    X, y = make_data()
    base = LogisticRegression(max_iter=1000).fit(X[:80], y[:80])
    original_coefficients = base.coef_.copy()

    calibrated = calibrate_fitted_classifier(base, X[80:], y[80:])

    np.testing.assert_array_equal(base.coef_, original_coefficients)
    assert calibrated.estimator.estimator is base
    probabilities = calibrated.predict_proba(X[80:])
    assert probabilities.shape == (40, 2)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


class _DecisionOnlyClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_ = True
        return self

    def decision_function(self, X):
        return np.asarray(X)[:, 0]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def test_fitted_calibration_uses_decision_function_when_no_probabilities_exist():
    X, y = make_data()
    base = _DecisionOnlyClassifier().fit(X[:80], y[:80])

    calibrated = calibrate_fitted_classifier(base, X[80:], y[80:])
    probabilities = calibrated.predict_proba(X[80:])

    assert probabilities.shape == (40, 2)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_calibrate_unfitted_classifier_raises():
    X, y = make_data()

    with pytest.raises(NotFittedError):
        calibrate_fitted_classifier(
            LogisticRegression(),
            X,
            y,
        )


def test_calibration_curve_returns_valid_values():
    y = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.7])

    predicted, observed = get_calibration_curve(
        y,
        probabilities,
        n_bins=3,
    )

    assert len(predicted) == len(observed)
    assert np.all((predicted >= 0) & (predicted <= 1))
    assert np.all((observed >= 0) & (observed <= 1))


@pytest.mark.parametrize(
    "y, probabilities, message",
    [
        ([0, 1], [[0.2, 0.8], [0.7, 0.3]], "one-dimensional"),
        ([0], [0.2, 0.3], "same length"),
        ([0, 1], [0.2, 1.5], "between 0 and 1"),
        ([0, 2], [0.2, 0.8], "binary labels"),
        ([0, 1], [0.2, float("nan")], "finite"),
    ],
)
def test_invalid_calibration_curve_inputs_raise(y, probabilities, message):
    with pytest.raises(ValueError, match=message):
        get_calibration_curve(y, probabilities)


def test_invalid_bin_count_raises():
    with pytest.raises(ValueError, match="n_bins"):
        get_calibration_curve([0, 1], [0.2, 0.8], n_bins=0)
