"""Tests for disjoint, stratified calibration experiment partitions."""

import numpy as np
import pandas as pd
import pytest

from trusttab.experiments.seeding import derive_seed
from trusttab.experiments.splitting import split_train_calibration_test


def _fixture_data():
    X = pd.DataFrame({"row_id": np.arange(200), "feature": np.arange(200) / 3})
    y = pd.Series(np.tile([0, 1], 100), index=X.index, name="target")
    return X, y


def _split(X, y, seed):
    return split_train_calibration_test(
        X,
        y,
        split_seed=derive_seed(seed, "split", "fixture"),
        calibration_split_seed=derive_seed(seed, "calibration_split", "fixture"),
    )


def test_three_way_split_is_disjoint_complete_and_stratified():
    X, y = _fixture_data()
    X_train, X_cal, X_test, y_train, y_cal, y_test = _split(X, y, 4)

    train_ids = set(X_train["row_id"])
    calibration_ids = set(X_cal["row_id"])
    test_ids = set(X_test["row_id"])
    assert train_ids.isdisjoint(calibration_ids)
    assert train_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)
    assert train_ids | calibration_ids | test_ids == set(X["row_id"])
    assert (len(X_train), len(X_cal), len(X_test)) == (140, 30, 30)
    assert (
        set(y_train.unique()) == set(y_cal.unique()) == set(y_test.unique()) == {0, 1}
    )
    assert X_train.index.equals(y_train.index)
    assert X_cal.index.equals(y_cal.index)
    assert X_test.index.equals(y_test.index)


def test_three_way_split_replays_for_same_seeds_and_varies_for_new_seed():
    X, y = _fixture_data()
    first = _split(X, y, 11)
    replay = _split(X, y, 11)
    different = _split(X, y, 12)

    for first_part, replay_part in zip(first, replay):
        assert first_part.equals(replay_part)
    assert not first[0]["row_id"].equals(different[0]["row_id"])


@pytest.mark.parametrize(
    "y, message",
    [
        (pd.Series([0] * 12), "both binary classes"),
        (pd.Series([0, 1, 2] * 4), "binary classes 0 and 1"),
        (pd.Series([0, 1, 0, np.nan, 1, 0]), "without NaNs"),
    ],
)
def test_three_way_split_rejects_unsupported_or_insufficient_labels(y, message):
    X = pd.DataFrame({"row_id": np.arange(len(y))})
    with pytest.raises(ValueError, match=message):
        _split(X, y, 0)


def test_three_way_split_rejects_feature_target_length_mismatch():
    X = pd.DataFrame({"row_id": np.arange(12)})
    y = pd.Series([0, 1] * 5)

    with pytest.raises(ValueError, match="same number"):
        _split(X, y, 0)
