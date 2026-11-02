"""Optional TabPFN adapter for TrustTab."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class TabPFNAdapter(ClassifierMixin, BaseEstimator):
    """Scikit-learn-compatible adapter for the optional TabPFN model."""

    def __init__(self, model_params: dict[str, Any] | None = None) -> None:
        self.model_params = model_params

    def fit(self, X: Any, y: Any) -> "TabPFNAdapter":
        """Create and fit TabPFN, reporting a helpful error if unavailable."""
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise ImportError(
                "TabPFN is optional. Install it with "
                "'pip install trusttab[tabpfn]'. "
                "Check TabPFN's installation and model-weight requirements."
            ) from exc

        self.model_ = TabPFNClassifier(**(self.model_params or {}))
        self.model_.fit(X, y)
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels."""
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities in the model's class order."""
        check_is_fitted(self, "model_")
        probabilities = np.asarray(self.model_.predict_proba(X))

        if probabilities.ndim != 2:
            raise ValueError("TabPFN predict_proba must return a 2D array.")

        return probabilities
