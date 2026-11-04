
"""Load and validate TrustTab experiment configuration."""

from pathlib import Path
from typing import Any

import yaml

from trusttab.models import SUPPORTED_MODELS
from trusttab.noise import validate_feature_shifts

SUPPORTED_DATASETS = ("german_credit", "adult")


def load_experiment_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """Load a YAML experiment configuration and validate its values."""

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    required_sections = (
        "project",
        "dataset",
        "experiments",
        "models",
        "output",
    )

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    if not isinstance(config["dataset"], dict):
        raise ValueError("dataset must be a YAML mapping.")

    dataset_name = config["dataset"].get("name")

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Currently supported: {', '.join(SUPPORTED_DATASETS)}."
        )

    experiments = config["experiments"]

    if not isinstance(experiments, dict):
        raise ValueError("experiments must be a YAML mapping.")

    feature_shifts = None
    if "feature_shifts" in config:
        try:
            feature_shifts = validate_feature_shifts(config["feature_shifts"])
            config["feature_shifts"] = feature_shifts
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid feature_shifts configuration: {exc}") from exc

    noise_levels = experiments.get("noise_levels")
    seeds = experiments.get("seeds")
    test_size = experiments.get("test_size", 0.2)
    models = config["models"]

    if not isinstance(noise_levels, list) and (
        noise_levels is not None or not feature_shifts
    ):
        raise ValueError("noise_levels must be a non-empty list.")
    if isinstance(noise_levels, list) and not noise_levels and not feature_shifts:
        raise ValueError("noise_levels must be a non-empty list.")

    if isinstance(noise_levels, list) and any(
        not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not 0 <= rate <= 1
        for rate in noise_levels
    ):
        raise ValueError(
            "Every noise level must be a number between 0 and 1."
        )

    if isinstance(noise_levels, list) and len(set(noise_levels)) != len(noise_levels):
        raise ValueError("Noise levels must be unique.")

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list.")

    if any(
        not isinstance(seed, int) or isinstance(seed, bool)
        for seed in seeds
    ):
        raise ValueError("Every seed must be an integer.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    if (
        not isinstance(test_size, (int, float))
        or isinstance(test_size, bool)
        or not 0 < test_size < 1
    ):
        raise ValueError("test_size must be between 0 and 1.")

    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list.")

    if len(set(models)) != len(models):
        raise ValueError("Model names must be unique.")

    unsupported_models = set(models) - set(SUPPORTED_MODELS)

    if unsupported_models:
        raise ValueError(
            f"Unsupported models: {sorted(unsupported_models)}. "
            f"Supported models: {list(SUPPORTED_MODELS)}"
        )

    if not isinstance(config["output"], dict):
        raise ValueError("output must be a YAML mapping.")

    output = config["output"]

    if not isinstance(output.get("directory"), str):
        raise ValueError("output.directory must be a string.")

    if not isinstance(output.get("filename"), str):
        raise ValueError("output.filename must be a string.")

    if not output["filename"].strip():
        raise ValueError("output.filename cannot be empty.")

    return config
