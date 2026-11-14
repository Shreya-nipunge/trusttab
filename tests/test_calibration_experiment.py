"""Leakage and end-to-end tests for the Platt calibration runner path."""

import numpy as np
import pandas as pd

from trusttab.experiments import benchmark
from trusttab.experiments.seeding import derive_seed
from trusttab.experiments.splitting import split_train_calibration_test


def _calibration_fixture():
    rng = np.random.default_rng(708)
    y = pd.Series(np.tile([0, 1], 100), name="target")
    X = pd.DataFrame(
        {"signal": y.to_numpy() + rng.normal(0, 0.8, len(y))},
        index=pd.Index(np.arange(len(y)), name="row_id"),
    )
    return X, y


def test_calibration_runner_keeps_fit_calibration_and_test_data_separate(
    monkeypatch, tmp_path
):
    X, y = _calibration_fixture()
    monkeypatch.setitem(benchmark.DATASET_LOADERS, "german_credit", lambda: (X, y))

    seed = 3
    split_seed = derive_seed(seed, "split", "german_credit")
    calibration_split_seed = derive_seed(seed, "calibration_split", "german_credit")
    X_train, X_cal, X_test, y_train, y_cal, y_test = split_train_calibration_test(
        X,
        y,
        split_seed=split_seed,
        calibration_split_seed=calibration_split_seed,
    )
    train_ids, calibration_ids, test_ids = (
        set(X_train.index),
        set(X_cal.index),
        set(X_test.index),
    )

    built_models = []
    original_build_model = benchmark.build_model

    def build_recording_model(*args, **kwargs):
        model = original_build_model(*args, **kwargs)
        original_fit = model.fit
        fit_rows = []

        def recording_fit(features, labels, **fit_params):
            fit_rows.append((set(features.index), np.asarray(labels).copy()))
            return original_fit(features, labels, **fit_params)

        model.fit = recording_fit
        built_models.append((model, fit_rows))
        return model

    monkeypatch.setattr(benchmark, "build_model", build_recording_model)

    original_noise = benchmark.inject_symmetric_label_noise
    noise_inputs = []

    def recording_noise(labels, noise_rate, seed):
        noise_inputs.append((labels.copy(), noise_rate, seed))
        return original_noise(labels, noise_rate, seed)

    monkeypatch.setattr(benchmark, "inject_symmetric_label_noise", recording_noise)

    original_calibrate = benchmark.calibrate_fitted_classifier
    calibrator_inputs = []
    fitted_calibrators = []

    def recording_calibrator(estimator, X_calibration, y_calibration, method):
        calibrator_inputs.append(
            (
                estimator,
                set(X_calibration.index),
                np.asarray(y_calibration).copy(),
                method,
            )
        )
        calibrated = original_calibrate(estimator, X_calibration, y_calibration, method)
        fitted_calibrators.append(calibrated)
        return calibrated

    monkeypatch.setattr(benchmark, "calibrate_fitted_classifier", recording_calibrator)

    original_evaluate = benchmark._evaluate
    evaluation_inputs = []

    def recording_evaluate(model, X_eval, y_eval, metric_evaluator):
        probabilities = np.asarray(model.predict_proba(X_eval))
        assert probabilities.shape == (len(X_eval), 2)
        assert np.isfinite(probabilities).all()
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        evaluation_inputs.append((set(X_eval.index), np.asarray(y_eval).copy()))
        return original_evaluate(model, X_eval, y_eval, metric_evaluator)

    monkeypatch.setattr(benchmark, "_evaluate", recording_evaluate)
    output_path = tmp_path / "calibrated.csv"
    results = benchmark.run_robustness_benchmark(
        dataset_name="german_credit",
        label_noise_levels=[0.0, 0.2],
        seeds=[seed],
        model_names=["logistic_regression"],
        test_size=0.2,
        model_params={"logistic_regression": {"max_iter": 500}},
        calibration_settings={"enabled": True},
        output_path=output_path,
    )

    assert train_ids.isdisjoint(calibration_ids)
    assert train_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)
    assert train_ids | calibration_ids | test_ids == set(X.index)
    assert (len(X_train), len(X_cal), len(X_test)) == (140, 30, 30)
    assert len(built_models) == 2
    assert len(noise_inputs) == 1
    passed_noise_labels, passed_noise_rate, passed_noise_seed = noise_inputs[0]
    pd.testing.assert_series_equal(passed_noise_labels, y_train)
    assert set(passed_noise_labels.index).issubset(train_ids)
    assert passed_noise_rate == 0.2
    assert passed_noise_seed == derive_seed(
        seed, "label_noise", "german_credit", float(0.2).hex()
    )
    assert all(len(fit_rows) == 1 for _, fit_rows in built_models)
    assert all(fit_rows[0][0] == train_ids for _, fit_rows in built_models)
    np.testing.assert_array_equal(built_models[0][1][0][1], y_train.to_numpy())
    assert not np.array_equal(built_models[1][1][0][1], y_train.to_numpy())
    assert all(rows == calibration_ids for _, rows, _, _ in calibrator_inputs)
    assert all(
        np.array_equal(labels, y_cal.to_numpy())
        for _, _, labels, _ in calibrator_inputs
    )
    assert all(method == "sigmoid" for _, _, _, method in calibrator_inputs)
    assert all(rows == test_ids for rows, _ in evaluation_inputs)
    assert all(
        np.array_equal(labels, y_test.to_numpy()) for _, labels in evaluation_inputs
    )
    pd.testing.assert_series_equal(y, pd.Series(np.tile([0, 1], 100), name="target"))

    label_rows = results.loc[results["experiment_type"] == "label_noise"]
    assert len(label_rows) == 2
    assert label_rows["condition_id"].is_unique
    assert label_rows["calibration_status"].eq("succeeded").all()
    assert label_rows["n_train"].eq(140).all()
    assert label_rows["n_calibration"].eq(30).all()
    assert label_rows["n_test"].eq(30).all()
    assert label_rows["calibration_method"].eq("sigmoid").all()
    for metric in benchmark.BENCHMARK_METRIC_COLUMNS:
        assert label_rows[metric].notna().all()
        assert label_rows[f"calibrated_{metric}"].notna().all()
    assert label_rows["model_params"].str.contains('"max_iter": 500').all()
    noisy_result = label_rows.loc[label_rows["corruption_level"] == 0.2].iloc[0]
    assert noisy_result["n_corrupted"] == np.sum(
        built_models[1][1][0][1] != y_train.to_numpy()
    )

    saved = pd.read_csv(output_path)
    assert len(saved) == len(results)
    assert set(
        f"calibrated_{metric}" for metric in benchmark.BENCHMARK_METRIC_COLUMNS
    ).issubset(saved.columns)

    first_calibrated_predictions = [
        calibrated.predict_proba(X_cal) for calibrated in fitted_calibrators
    ]
    changed_test_X = X.copy(deep=True)
    changed_test_X.loc[list(test_ids), "signal"] += 10_000
    monkeypatch.setitem(
        benchmark.DATASET_LOADERS,
        "german_credit",
        lambda: (changed_test_X, y),
    )
    second_fit_start = len(built_models)
    second_calibrator_start = len(fitted_calibrators)
    benchmark.run_robustness_benchmark(
        dataset_name="german_credit",
        label_noise_levels=[0.0, 0.2],
        seeds=[seed],
        model_names=["logistic_regression"],
        model_params={"logistic_regression": {"max_iter": 500}},
        calibration_settings={"enabled": True},
    )
    second_models = built_models[second_fit_start:]
    second_calibrators = fitted_calibrators[second_calibrator_start:]
    assert all(fit_rows[0][0] == train_ids for _, fit_rows in second_models)
    for changed_test_calibrator, expected_probabilities in zip(
        second_calibrators,
        first_calibrated_predictions,
    ):
        np.testing.assert_allclose(
            changed_test_calibrator.predict_proba(X_cal),
            expected_probabilities,
        )


def test_calibration_failure_is_reported_without_fake_metrics(monkeypatch):
    X, y = _calibration_fixture()
    monkeypatch.setitem(benchmark.DATASET_LOADERS, "german_credit", lambda: (X, y))

    def fail_calibration(*args, **kwargs):
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(benchmark, "calibrate_fitted_classifier", fail_calibration)

    result = benchmark.run_robustness_benchmark(
        dataset_name="german_credit",
        label_noise_levels=[0.2],
        seeds=[1],
        model_names=["logistic_regression"],
        calibration_settings={"enabled": True},
    )
    calibrated = result.loc[result["experiment_type"] == "label_noise"].iloc[0]

    assert calibrated["calibration_status"] == "failed"
    assert "RuntimeError: calibration failed" == calibrated["calibration_error"]
    assert np.isnan(calibrated["calibrated_ece"])
    assert np.isnan(calibrated["calibrated_brier_score"])
    assert np.isfinite(calibrated["ece"])
    assert np.isfinite(calibrated["brier_score"])


def test_evaluator_uses_positive_class_probability_when_class_order_is_reversed():
    class ReversedClassModel:
        classes_ = np.array([1, 0])

        def predict(self, X):
            return np.array([1, 0])

        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    observed_positive_probabilities = []

    def capture_metrics(y_true, y_pred, y_prob):
        observed_positive_probabilities.extend(y_prob.tolist())
        return {"accuracy": 1.0}

    benchmark._evaluate(
        ReversedClassModel(),
        pd.DataFrame({"signal": [0.0, 1.0]}),
        pd.Series([1, 0]),
        capture_metrics,
    )

    assert observed_positive_probabilities == [0.8, 0.1]
