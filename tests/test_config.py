"""Tests for experiment configuration loading."""

from textwrap import dedent

import pytest

from trusttab.experiments.config import load_experiment_config


def test_load_valid_experiment_config():
    config = load_experiment_config("configs/a6_smoke_test.yaml")

    assert config["dataset"]["name"] == "german_credit"
    assert config["experiments"]["noise_levels"] == [0.0, 0.2]
    assert config["experiments"]["seeds"] == [42]
    assert config["models"] == [
        "logistic_regression",
        "random_forest",
    ]


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_experiment_config("configs/does_not_exist.yaml")


def test_unsupported_dataset(tmp_path):
    config_file = tmp_path / "invalid.yaml"

    config_file.write_text(
        """
project:
  name: TrustTab
dataset:
  name: unsupported_dataset
experiments:
  noise_levels: [0.0]
  seeds: [42]
models:
  - logistic_regression
output:
  directory: results
  filename: test.csv
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_experiment_config(config_file)


def test_unsupported_model(tmp_path):
    config_file = tmp_path / "invalid_model.yaml"

    config_file.write_text(
        """
project:
  name: TrustTab
dataset:
  name: german_credit
experiments:
  noise_levels: [0.0]
  seeds: [42]
models:
  - unsupported_model
output:
  directory: results
  filename: test.csv
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported models"):
        load_experiment_config(config_file)


def test_load_adult_experiment_config(tmp_path):
    config_file = tmp_path / "adult.yaml"

    config_file.write_text(
        """
project:
  name: TrustTab
  version: "0.1.0"
dataset:
  name: adult
experiments:
  test_size: 0.2
  noise_levels: [0.0, 0.2, 0.4]
  seeds: [42]
models:
  - logistic_regression
output:
  directory: results
  filename: adult.csv
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_file)

    assert config["dataset"]["name"] == "adult"
    assert config["experiments"]["noise_levels"] == [
        0.0,
        0.2,
        0.4,
    ]


def _write_feature_shift_config(tmp_path, shifts):
    config_file = tmp_path / "feature_shift.yaml"
    config_file.write_text(
        """
project:
  name: TrustTab
dataset:
  name: german_credit
experiments:
  noise_levels: [0.0]
  seeds: [42]
models:
  - logistic_regression
output:
  directory: results
  filename: shifted.csv
feature_shifts:
"""
        + shifts,
        encoding="utf-8",
    )
    return config_file


def test_feature_shift_configuration_is_loaded(tmp_path):
    path = _write_feature_shift_config(
        tmp_path,
        """  - type: gaussian_noise
    features: [age]
    magnitude: 0.2
  - type: mean_shift
    features: [age]
    magnitude: 0.5
""",
    )
    config = load_experiment_config(path)
    assert len(config["feature_shifts"]) == 2
    assert config["feature_shifts"][0]["type"] == "gaussian_noise"

    path.write_text(
        path.read_text(encoding="utf-8").replace("  noise_levels: [0.0]\n", ""),
        encoding="utf-8",
    )
    config = load_experiment_config(path)
    assert len(config["feature_shifts"]) == 2


@pytest.mark.parametrize(
    "shift_yaml",
    [
        "  - type: unsupported\n    features: [age]\n    magnitude: 1.0\n",
        "  - type: gaussian_noise\n    features: [age]\n    magnitude: -1.0\n",
        "  - type: mean_shift\n    features: []\n    magnitude: 1.0\n",
        "  - type: category_probability_redistribution\n    features: [job]\n    magnitude: 1.0\n    probabilities: {a: 0.0, b: 0.0}\n",
    ],
)
def test_invalid_feature_shift_configuration_is_rejected(tmp_path, shift_yaml):
    path = _write_feature_shift_config(tmp_path, shift_yaml)
    with pytest.raises(ValueError, match="feature_shifts"):
        load_experiment_config(path)


def test_load_combined_unified_benchmark_config(tmp_path):
    path = tmp_path / "unified.yaml"
    path.write_text(
        dedent(
            """
            project:
              name: TrustTab
            dataset:
              names: [german_credit, adult]
            experiments:
              seeds: [0, 1]
              label_noise:
                enabled: true
                levels: [0.0, 0.2]
              feature_shift:
                enabled: true
                types: [mean_shift, variance_scaling]
                magnitudes: [0.5, 1.0]
                features: [age]
            models:
              - logistic_regression
            output:
              directory: results
              filename: unified.csv
            """
        ),
        encoding="utf-8",
    )
    config = load_experiment_config(path)
    assert config["benchmark_mode"] == "unified"
    assert config["benchmark_settings"]["label_noise_levels"] == [0.0, 0.2]
    assert config["benchmark_settings"]["feature_shifts"]["types"] == [
        "mean_shift",
        "variance_scaling",
    ]
    assert config["dataset"]["names"] == ["german_credit", "adult"]


def test_unified_configuration_rejects_invalid_feature_magnitude(tmp_path):
    path = tmp_path / "invalid-unified.yaml"
    path.write_text(
        dedent(
            """
            project:
              name: TrustTab
            dataset:
              name: german_credit
            experiments:
              seeds: [1]
              label_noise:
                enabled: true
                levels: [0.0]
              feature_shift:
                enabled: true
                types: [variance_scaling]
                magnitudes: [-1.0]
                features: [age]
            models:
              - logistic_regression
            output:
              directory: results
              filename: invalid.csv
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-negative"):
        load_experiment_config(path)


def test_tracking_settings_are_normalized_and_validated(tmp_path):
    path = tmp_path / "tracked-config.yaml"
    path.write_text(
        dedent(
            """\
            project:
              name: TrustTab
            dataset:
              name: german_credit
            experiments:
              seeds: [1]
              noise_levels: [0.0]
            models: [logistic_regression]
            output:
              directory: results
              filename: result.csv
            tracking:
              enabled: true
              directory: local_runs
              experiment_name: smoke
            """
        ),
        encoding="utf-8",
    )
    config = load_experiment_config(path)
    assert config["tracking"] == {
        "enabled": True,
        "directory": "local_runs",
        "experiment_name": "smoke",
    }

    path.write_text(
        path.read_text(encoding="utf-8").replace("enabled: true", "enabled: yesplease"),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="tracking.enabled"):
        load_experiment_config(path)
