"""Tests for experiment configuration loading."""

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