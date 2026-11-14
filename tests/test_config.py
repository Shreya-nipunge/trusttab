"""Tests for experiment configuration loading."""

from textwrap import dedent
from itertools import product

import yaml

import pytest

from trusttab.experiments.config import (
    expand_experiment_conditions,
    load_experiment_config,
)


def test_load_valid_experiment_config():
    config = load_experiment_config("configs/a6_smoke_test.yaml")

    assert config["dataset"]["name"] == "german_credit"
    assert config["experiments"]["noise_levels"] == [0.0, 0.2]
    assert config["experiments"]["seeds"] == [42]
    assert config["models"] == [
        "logistic_regression",
        "random_forest",
    ]


def test_core_config_expands_to_exactly_200_unique_conditions():
    config = load_experiment_config("configs/experiments.yaml")
    conditions = expand_experiment_conditions(config)
    actual = {
        (
            condition["dataset"],
            condition["model"],
            condition["noise_level"],
            condition["seed"],
        )
        for condition in conditions
    }
    expected = set(
        product(
            ["german_credit", "heart_disease"],
            ["logistic_regression", "random_forest", "xgboost", "tabpfn"],
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0, 1, 2, 3, 4],
        )
    )

    assert config["model_params"]["random_forest"]["n_estimators"] == 300
    assert len(conditions) == 200
    assert len(actual) == 200
    assert actual == expected


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("dataset", "names"),
        ("models", None),
        ("noise", None),
        ("seeds", None),
    ],
)
def test_core_config_rejects_duplicate_condition_axes(tmp_path, section, key):
    with open("configs/experiments.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if section == "dataset":
        config["dataset"][key].append(config["dataset"][key][0])
    elif section == "models":
        config["models"].append(config["models"][0])
    elif section == "noise":
        levels = config["experiments"]["label_noise"]["levels"]
        levels.append(levels[0])
    else:
        config["experiments"]["seeds"].append(config["experiments"]["seeds"][0])
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_experiment_config(path)


def test_phase3_smoke_config_is_bounded():
    config = load_experiment_config("configs/phase3_smoke_test.yaml")

    assert config["dataset"]["name"] == "german_credit"
    assert config["models"] == ["logistic_regression", "random_forest"]
    assert config["experiments"]["seeds"] == [0]
    assert config["benchmark_settings"]["label_noise_levels"] == [0.0, 0.2]
    assert len(expand_experiment_conditions(config)) == 4


def test_calibration_smoke_config_is_enabled_and_uses_blueprint_split():
    config = load_experiment_config("configs/calibration_rescue_smoke.yaml")

    assert config["benchmark_mode"] == "unified"
    assert config["calibration_settings"] == {
        "enabled": True,
        "method": "sigmoid",
        "train_size": 0.7,
        "calibration_size": 0.15,
        "test_size": 0.15,
    }
    assert len(expand_experiment_conditions(config)) == 2


def test_calibration_is_disabled_by_default_to_preserve_existing_configs():
    config = load_experiment_config("configs/phase3_smoke_test.yaml")

    assert config["calibration_settings"]["enabled"] is False
    assert config["benchmark_mode"] == "label_noise_legacy"


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": "yes"},
        {"method": "isotonic"},
        {"train_size": 0.8},
        {"calibration_size": 0.0},
        {"unknown": True},
    ],
)
def test_invalid_calibration_settings_are_rejected(tmp_path, settings):
    config = {
        "project": {"name": "TrustTab"},
        "dataset": {"name": "german_credit"},
        "experiments": {
            "noise_levels": [0.0],
            "seeds": [0],
            "calibration": settings,
        },
        "models": ["logistic_regression"],
        "output": {"directory": "results", "filename": "test.csv"},
    }
    path = tmp_path / "invalid-calibration.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="calibration"):
        load_experiment_config(path)


def test_model_params_reject_random_state_override(tmp_path):
    path = tmp_path / "random-state.yaml"
    path.write_text(
        """
project: {name: TrustTab}
dataset: {name: german_credit}
experiments: {noise_levels: [0.0], seeds: [0]}
models: [random_forest]
model_params:
  random_forest: {random_state: 99}
output: {directory: results, filename: test.csv}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="managed by the experiment seed streams"):
        load_experiment_config(path)


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


def test_load_heart_disease_experiment_config(tmp_path):
    config_file = tmp_path / "heart_disease.yaml"
    config_file.write_text(
        """
project:
  name: TrustTab
dataset:
  name: heart_disease
experiments:
  noise_levels: [0.0]
  seeds: [0]
models:
  - logistic_regression
output:
  directory: results
  filename: heart_disease.csv
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_file)

    assert config["dataset"]["name"] == "heart_disease"


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
