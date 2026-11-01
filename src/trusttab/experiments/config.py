
"""Load and validate TrustTab experiment configuration."""

from pathlib import Path
from typing import Any

import yaml

from trusttab.models import SUPPORTED_MODELS


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

    dataset_name = config["dataset"].get("name")

    if dataset_name != "german_credit":
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            "Currently supported: german_credit."
        )

    experiments = config["experiments"]

    noise_levels = experiments.get("noise_levels")
    seeds = experiments.get("seeds")
    test_size = experiments.get("test_size", 0.2)
    models = config["models"]

    if not isinstance(noise_levels, list) or not noise_levels:
        raise ValueError("noise_levels must be a non-empty list.")

    if any(
        not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not 0 <= rate <= 1
        for rate in noise_levels
    ):
        raise ValueError(
            "Every noise level must be a number between 0 and 1."
        )

    if len(set(noise_levels)) != len(noise_levels):
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

    output = config["output"]

    if not isinstance(output.get("directory"), str):
        raise ValueError("output.directory must be a string.")

    if not isinstance(output.get("filename"), str):
        raise ValueError("output.filename must be a string.")

    if not output["filename"].strip():
        raise ValueError("output.filename cannot be empty.")

    return config
