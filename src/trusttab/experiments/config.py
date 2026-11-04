"""Load and validate TrustTab experiment configuration."""

from pathlib import Path
from typing import Any

import yaml

from trusttab.models import SUPPORTED_MODELS
from trusttab.noise import validate_feature_shifts

SUPPORTED_DATASETS = ("german_credit", "adult")


def _validate_noise_levels(levels: Any) -> list[float]:
    if not isinstance(levels, list) or not levels:
        raise ValueError("noise levels must be a non-empty list.")
    if any(
        isinstance(level, bool)
        or not isinstance(level, (int, float))
        or not 0 <= level <= 1
        for level in levels
    ):
        raise ValueError("Every noise level must be a number between 0 and 1.")
    if len(set(levels)) != len(levels):
        raise ValueError("Noise levels must be unique.")
    return [float(level) for level in levels]


def _validate_feature_shift_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise TypeError("experiments.feature_shift must be a YAML mapping.")
    enabled = settings.get("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError("experiments.feature_shift.enabled must be a boolean.")
    types = settings.get("types", [])
    magnitudes = settings.get("magnitudes", [])
    if not isinstance(types, list) or not isinstance(magnitudes, list):
        raise TypeError("Feature-shift types and magnitudes must be lists.")
    if enabled and (not types or not magnitudes):
        raise ValueError(
            "Enabled feature shifts require non-empty types and magnitudes."
        )
    if any(not isinstance(shift_type, str) for shift_type in types):
        raise ValueError("Every feature-shift type must be a string.")
    if len(set(types)) != len(types):
        raise ValueError("Feature-shift types must be unique.")
    if any(
        isinstance(magnitude, (dict, list, set, bool))
        or not isinstance(magnitude, (int, float))
        for magnitude in magnitudes
    ):
        raise ValueError("Feature-shift magnitudes must be numbers.")
    if len(set(magnitudes)) != len(magnitudes):
        raise ValueError("Feature-shift magnitudes must be unique.")
    features = settings.get("features")
    if features is not None:
        if isinstance(features, dict):
            if set(features).difference(types):
                raise ValueError(
                    "Feature selections may only use configured shift types."
                )
            for feature_names in features.values():
                if (
                    not isinstance(feature_names, list)
                    or not feature_names
                    or any(
                        not isinstance(name, str) or not name for name in feature_names
                    )
                    or len(set(feature_names)) != len(feature_names)
                ):
                    raise ValueError(
                        "Each feature selection must be a non-empty unique list."
                    )
        elif (
            not isinstance(features, list)
            or not features
            or any(not isinstance(name, str) or not name for name in features)
            or len(set(features)) != len(features)
        ):
            raise ValueError(
                "features must be a non-empty list or a type-to-list mapping."
            )

    if types and magnitudes:
        for shift_type in types:
            selected = (
                features.get(shift_type) if isinstance(features, dict) else features
            ) or ["__feature__"]
            for magnitude in magnitudes:
                try:
                    validate_feature_shifts(
                        [
                            {
                                "type": shift_type,
                                "features": selected,
                                "magnitude": magnitude,
                            }
                        ]
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid feature_shift setting: {exc}") from exc
    return {
        "enabled": enabled,
        "types": list(types),
        "magnitudes": list(magnitudes),
        **({"features": features} if features is not None else {}),
    }


def load_experiment_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """Load a YAML experiment configuration and validate its values."""

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

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
            raise ValueError(f"Missing configuration section: {section}")

    if not isinstance(config["dataset"], dict):
        raise ValueError("dataset must be a YAML mapping.")

    dataset_names = config["dataset"].get("names")
    if dataset_names is None:
        dataset_name = config["dataset"].get("name")
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Currently supported: {', '.join(SUPPORTED_DATASETS)}."
            )
        dataset_names = [dataset_name]
    if (
        not isinstance(dataset_names, list)
        or not dataset_names
        or any(name not in SUPPORTED_DATASETS for name in dataset_names)
        or len(set(dataset_names)) != len(dataset_names)
    ):
        raise ValueError(
            f"dataset.name or dataset.names must contain supported unique names: "
            f"{', '.join(SUPPORTED_DATASETS)}."
        )

    experiments = config["experiments"]

    if not isinstance(experiments, dict):
        raise ValueError("experiments must be a YAML mapping.")

    feature_shifts = None
    if "feature_shifts" in config and "feature_shift" in experiments:
        raise ValueError(
            "Use experiments.feature_shift or legacy top-level feature_shifts, not both."
        )
    if "feature_shifts" in config:
        if (
            not isinstance(config["feature_shifts"], list)
            or not config["feature_shifts"]
        ):
            raise ValueError("feature_shifts must be a non-empty list.")
        try:
            feature_shifts = validate_feature_shifts(config["feature_shifts"])
            config["feature_shifts"] = feature_shifts
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid feature_shifts configuration: {exc}") from exc

    has_nested_label_noise = "label_noise" in experiments
    nested_label_noise = experiments.get("label_noise")
    has_legacy_noise_levels = False
    if has_nested_label_noise:
        if not isinstance(nested_label_noise, dict):
            raise ValueError("experiments.label_noise must be a YAML mapping.")
        if "noise_levels" in experiments:
            raise ValueError(
                "Use experiments.label_noise.levels or legacy noise_levels, not both."
            )
        enabled = nested_label_noise.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("experiments.label_noise.enabled must be a boolean.")
        raw_levels = nested_label_noise.get("levels", [])
        if enabled or raw_levels:
            levels = _validate_noise_levels(raw_levels)
        else:
            levels = []
        experiments["label_noise"] = {"enabled": enabled, "levels": levels}
    else:
        noise_levels = experiments.get("noise_levels")
        has_legacy_noise_levels = noise_levels is not None
        if noise_levels is None:
            noise_levels = (
                [] if feature_shifts or "feature_shift" in experiments else None
            )
        if noise_levels is None or (
            not noise_levels
            and not feature_shifts
            and "feature_shift" not in experiments
        ):
            raise ValueError("noise_levels must be a non-empty list.")
        levels = _validate_noise_levels(noise_levels) if noise_levels else []
        # Preserve the legacy field, and expose a normalized view for unified runs.
        experiments["label_noise"] = {
            "enabled": bool(levels),
            "levels": levels,
        }

    feature_shift_settings = None
    if "feature_shift" in experiments:
        feature_shift_settings = _validate_feature_shift_settings(
            experiments["feature_shift"]
        )
        experiments["feature_shift"] = feature_shift_settings
    elif feature_shifts is not None:
        feature_shift_settings = feature_shifts

    if (
        has_nested_label_noise
        or "feature_shift" in experiments
        or len(dataset_names) > 1
        or (feature_shifts is not None and has_legacy_noise_levels)
    ):
        config["benchmark_mode"] = "unified"
    elif feature_shifts is not None:
        config["benchmark_mode"] = "feature_shift_legacy"
    else:
        config["benchmark_mode"] = "label_noise_legacy"

    config["benchmark_settings"] = {
        "label_noise_levels": levels,
        "feature_shifts": feature_shift_settings,
    }

    test_size = experiments.get("test_size", 0.2)
    seeds = experiments.get("seeds")
    models = config["models"]

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list.")

    if any(
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= 2**32 - 1
        for seed in seeds
    ):
        raise ValueError("Every seed must be an integer between 0 and 2**32 - 1.")

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

    tracking = config.get("tracking", {})
    if not isinstance(tracking, dict):
        raise TypeError("tracking must be a YAML mapping.")
    tracking_enabled = tracking.get("enabled", False)
    if not isinstance(tracking_enabled, bool):
        raise TypeError("tracking.enabled must be a boolean.")
    tracking_directory = tracking.get("directory", "experiments/runs")
    if not isinstance(tracking_directory, str):
        raise TypeError("tracking.directory must be a string.")
    if not tracking_directory.strip():
        raise ValueError("tracking.directory must be a non-empty string.")
    experiment_name = tracking.get("experiment_name")
    if experiment_name is not None and (
        not isinstance(experiment_name, str) or not experiment_name.strip()
    ):
        raise ValueError("tracking.experiment_name must be a non-empty string.")
    config["tracking"] = {
        "enabled": tracking_enabled,
        "directory": tracking_directory,
        "experiment_name": experiment_name,
    }

    return config
