"""Tests for the optional TabPFN adapter."""

import sys
from types import ModuleType

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from trusttab.models.tabpfn_adapter import TabPFNAdapter


def test_import_does_not_require_tabpfn():
    """Importing the adapter should not require the optional package."""
    assert TabPFNAdapter is not None


def test_fit_predict_with_mocked_tabpfn(monkeypatch):
    """Test predictions using a fake TabPFN classifier."""
    class FakeTabPFNClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([0.8, 0.2], (len(X), 1))

    fake_module = ModuleType("tabpfn")
    fake_module.TabPFNClassifier = FakeTabPFNClassifier
    monkeypatch.setitem(sys.modules, "tabpfn", fake_module)

    model = TabPFNAdapter(model_params={"device": "cpu"})
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    model.fit(X, y)

    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_array_equal(model.predict(X), [0, 0, 0])
    np.testing.assert_allclose(
        model.predict_proba(X),
        [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
    )
    assert model.model_.kwargs == {"device": "cpu"}


def test_predict_before_fit_raises():
    """Predicting before fitting should raise NotFittedError."""
    model = TabPFNAdapter()

    with pytest.raises(NotFittedError):
        model.predict([[1]])


def test_missing_tabpfn_gives_install_instructions(monkeypatch):
    """Missing TabPFN should produce a helpful installation message."""
    original_import = __import__

    def missing_import(name, *args, **kwargs):
        if name == "tabpfn":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_import)

    model = TabPFNAdapter()

    with pytest.raises(ImportError, match=r"trusttab\[tabpfn\]"):
        model.fit([[1], [2]], [0, 1])