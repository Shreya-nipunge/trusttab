"""Reproducible feature-distribution shift utilities for tabular data."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

_SHIFT_ALIASES = {
    "gaussian_noise": "gaussian_noise",
    "mean_shift": "mean_shift",
    "variance_scaling": "variance_scaling",
    "category_probability_redistribution": "category_probability_redistribution",
    "categorical_probability_redistribution": "category_probability_redistribution",
    "category_replacement": "category_replacement",
}


def _validate_frame(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.columns.has_duplicates:
        raise ValueError("DataFrame column names must be unique.")
    return data.copy(deep=True)


def _validate_magnitude(value: Any, shift_type: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError("magnitude must be a finite number.")
    magnitude = float(value)
    if not np.isfinite(magnitude):
        raise ValueError("magnitude must be finite.")
    if shift_type in {"gaussian_noise", "variance_scaling"} and magnitude < 0:
        raise ValueError(f"magnitude for {shift_type} must be non-negative.")
    if shift_type in {
        "category_probability_redistribution",
        "category_replacement",
    } and not 0 <= magnitude <= 1:
        raise ValueError(f"magnitude for {shift_type} must be between 0 and 1.")
    return magnitude


def _validate_rng(seed: int | None) -> np.random.Generator:
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
    ):
        raise TypeError("seed must be an integer or None.")
    return np.random.default_rng(seed)


def _numeric_shift(
    data: pd.DataFrame,
    feature: str,
    shift_type: str,
    magnitude: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if feature not in data.columns:
        raise ValueError(f"Unknown feature: {feature}")
    series = data[feature]
    if not is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"Feature '{feature}' must have a numeric dtype for {shift_type}.")
    observed = series.notna().to_numpy()
    values = series.to_numpy(dtype=float, na_value=np.nan, copy=True)
    if not observed.any():
        raise ValueError(f"Feature '{feature}' contains no observed values.")
    current = values[observed]

    if shift_type == "gaussian_noise":
        scale = float(np.std(current)) * magnitude
        values[observed] = current + rng.normal(0.0, scale, size=len(current))
    elif shift_type == "mean_shift":
        values[observed] = current + magnitude
    elif shift_type == "variance_scaling":
        mean = float(np.mean(current))
        # magnitude is a variance multiplier, so deviations scale by sqrt(magnitude).
        values[observed] = mean + (current - mean) * np.sqrt(magnitude)
    else:
        raise ValueError(f"Unsupported numerical shift type: {shift_type}")

    data[feature] = pd.Series(values, index=series.index, name=series.name)
    return data


def _probability_mapping(
    probabilities: Mapping[Any, Any] | None,
    categories: np.ndarray,
    feature: str,
) -> np.ndarray:
    if probabilities is None:
        return np.full(len(categories), 1.0 / len(categories))
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("probabilities must be a non-empty category-to-weight mapping.")
    unknown = set(probabilities).difference(categories.tolist())
    if unknown:
        raise ValueError(
            f"Unknown category for feature '{feature}': {sorted(unknown, key=str)}"
        )
    weights = np.asarray([probabilities.get(category, 0.0) for category in categories], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Category probabilities must be finite, non-negative, and sum to > 0.")
    return weights / weights.sum()


def _categorical_shift(
    data: pd.DataFrame,
    feature: str,
    shift_type: str,
    magnitude: float,
    rng: np.random.Generator,
    *,
    probabilities: Mapping[Any, Any] | None = None,
    replacement: Any = None,
    has_replacement: bool = False,
) -> pd.DataFrame:
    if feature not in data.columns:
        raise ValueError(f"Unknown feature: {feature}")
    series = data[feature]
    observed = series.notna().to_numpy()
    if not observed.any():
        raise ValueError(f"Feature '{feature}' contains no observed values.")
    values = series.to_numpy(dtype=object, copy=True)
    categories = pd.unique(values[observed])
    if (
        magnitude > 0
        and len(categories) < 2
        and shift_type == "category_replacement"
        and not has_replacement
    ):
        raise ValueError(
            f"Feature '{feature}' needs at least two observed categories for replacement."
        )
    if has_replacement and replacement not in categories:
        raise ValueError(
            f"Replacement category {replacement!r} is not available in feature '{feature}'."
        )

    if shift_type == "category_probability_redistribution":
        if magnitude == 0:
            return data
        observed_values = values[observed]
        empirical = np.asarray(
            [np.mean(observed_values == category) for category in categories],
            dtype=float,
        )
        target = _probability_mapping(probabilities, categories, feature)
        distribution = (1.0 - magnitude) * empirical + magnitude * target
        values[observed] = rng.choice(categories, size=observed.sum(), p=distribution)
    elif shift_type == "category_replacement":
        current = values[observed]
        n_replace = round(magnitude * len(current))
        candidates = np.flatnonzero(current != replacement) if has_replacement else np.arange(len(current))
        n_replace = min(n_replace, len(candidates))
        if n_replace:
            selected = rng.choice(candidates, size=n_replace, replace=False)
            if has_replacement:
                current[selected] = replacement
            else:
                for index in selected:
                    choices = categories[categories != current[index]]
                    current[index] = rng.choice(choices)
            values[observed] = current
    else:
        raise ValueError(f"Unsupported categorical shift type: {shift_type}")

    # Rebuild categorical columns with the same category set and ordering.
    if isinstance(series.dtype, pd.CategoricalDtype):
        data[feature] = pd.Categorical(
            values,
            categories=series.cat.categories,
            ordered=series.cat.ordered,
        )
    else:
        data[feature] = pd.Series(values, index=series.index, name=series.name)
    return data


def apply_numerical_feature_shift(
    data: pd.DataFrame,
    feature: str,
    shift_type: str,
    magnitude: float,
    seed: int | None = None,
) -> pd.DataFrame:
    """Apply Gaussian noise, a mean offset, or variance scaling to one column.

    Gaussian-noise magnitude is a multiplier of the observed feature standard
    deviation. Mean-shift magnitude is an additive offset in feature units.
    Variance-scaling magnitude multiplies variance (deviations are scaled by
    its square root). Missing values are left unchanged.
    """
    canonical = _SHIFT_ALIASES.get(shift_type)
    if canonical not in {"gaussian_noise", "mean_shift", "variance_scaling"}:
        raise ValueError(f"Unsupported numerical shift type: {shift_type}")
    output = _validate_frame(data)
    amount = _validate_magnitude(magnitude, canonical)
    return _numeric_shift(output, feature, canonical, amount, _validate_rng(seed))


def apply_categorical_feature_shift(
    data: pd.DataFrame,
    feature: str,
    shift_type: str,
    magnitude: float,
    seed: int | None = None,
    *,
    probabilities: Mapping[Any, Any] | None = None,
    replacement: Any = None,
) -> pd.DataFrame:
    """Redistribute categorical probabilities or replace a controlled fraction.

    For probability redistribution, magnitude is the mixing fraction toward
    the supplied category probabilities (or a uniform target if omitted).
    For replacement, magnitude is the fraction of observed rows to replace.
    When replacement is omitted, each selected value changes to another
    category observed in that feature.
    """
    canonical = _SHIFT_ALIASES.get(shift_type)
    if canonical not in {
        "category_probability_redistribution",
        "category_replacement",
    }:
        raise ValueError(f"Unsupported categorical shift type: {shift_type}")
    output = _validate_frame(data)
    amount = _validate_magnitude(magnitude, canonical)
    return _categorical_shift(
        output,
        feature,
        canonical,
        amount,
        _validate_rng(seed),
        probabilities=probabilities,
        replacement=replacement,
        has_replacement=replacement is not None,
    )


def validate_feature_shifts(shifts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize shift specifications without applying them."""
    if not isinstance(shifts, Sequence) or isinstance(shifts, (str, bytes)):
        raise TypeError("shifts must be a sequence of mappings.")
    normalized = []
    for index, spec in enumerate(shifts):
        if not isinstance(spec, Mapping):
            raise TypeError(f"feature_shifts[{index}] must be a mapping.")
        shift_type = spec.get("type")
        if not isinstance(shift_type, str) or shift_type not in _SHIFT_ALIASES:
            raise ValueError(f"feature_shifts[{index}] has unsupported type: {shift_type!r}.")
        canonical = _SHIFT_ALIASES[shift_type]
        features = spec.get("features")
        if (
            not isinstance(features, Sequence)
            or isinstance(features, (str, bytes))
            or not features
            or any(not isinstance(feature, str) or not feature for feature in features)
        ):
            raise ValueError(f"feature_shifts[{index}].features must be a non-empty list of names.")
        if len(set(features)) != len(features):
            raise ValueError(f"feature_shifts[{index}].features must be unique.")
        if "magnitude" not in spec:
            raise ValueError(f"feature_shifts[{index}] requires magnitude.")
        amount = _validate_magnitude(spec["magnitude"], canonical)
        item = dict(spec)
        item["type"] = canonical
        item["features"] = list(features)
        item["magnitude"] = amount
        if canonical == "category_probability_redistribution" and "probabilities" in item:
            probability_spec = item["probabilities"]
            if not isinstance(probability_spec, Mapping) or not probability_spec:
                raise ValueError(
                    f"feature_shifts[{index}].probabilities must be a non-empty mapping."
                )
            nested_targets = all(
                isinstance(value, Mapping) for value in probability_spec.values()
            )
            if nested_targets:
                if set(probability_spec) != set(features):
                    raise ValueError(
                        "Per-feature probabilities must specify every affected feature."
                    )
                targets = [probability_spec[feature] for feature in features]
            else:
                targets = [probability_spec]
            for target in targets:
                if not isinstance(target, Mapping) or not target:
                    raise ValueError("Each category probability target must be a non-empty mapping.")
                try:
                    weights = np.asarray(list(target.values()), dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Category probability weights must be numeric.") from exc
                if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                    raise ValueError(
                        "Category probabilities must be finite, non-negative, and sum to > 0."
                    )
        if "replacements" in item:
            replacements = item["replacements"]
            if not isinstance(replacements, Mapping):
                raise ValueError(f"feature_shifts[{index}].replacements must be a mapping.")
            unknown_features = set(replacements).difference(features)
            if unknown_features:
                raise ValueError(
                    f"feature_shifts[{index}].replacements has features not listed in features: "
                    f"{sorted(unknown_features)}"
                )
        normalized.append(item)
    return normalized


def apply_feature_shifts(
    data: pd.DataFrame,
    shifts: Sequence[Mapping[str, Any]],
    seed: int | None = None,
) -> pd.DataFrame:
    """Apply a sequence of feature-level transformations to a copied frame.

    Column names, row index, and shape are preserved. Missing cells are
    excluded from perturbation and remain missing. Every random operation uses
    the seeded generator in specification order.
    """
    output = _validate_frame(data)
    normalized = validate_feature_shifts(shifts)
    rng = _validate_rng(seed)
    for spec in normalized:
        shift_type = spec["type"]
        for feature in spec["features"]:
            if feature not in output.columns:
                raise ValueError(f"Unknown feature in feature shift: {feature}")
            if shift_type in {"gaussian_noise", "mean_shift", "variance_scaling"}:
                output = _numeric_shift(
                    output, feature, shift_type, spec["magnitude"], rng
                )
            else:
                probabilities = spec.get("probabilities")
                if isinstance(probabilities, Mapping) and feature in probabilities and isinstance(
                    probabilities[feature], Mapping
                ):
                    probabilities = probabilities[feature]
                replacements = spec.get("replacements")
                replacement = (
                    replacements.get(feature)
                    if isinstance(replacements, Mapping) and feature in replacements
                    else spec.get("replacement")
                )
                has_replacement = (
                    (isinstance(replacements, Mapping) and feature in replacements)
                    or "replacement" in spec
                )
                output = _categorical_shift(
                    output,
                    feature,
                    shift_type,
                    spec["magnitude"],
                    rng,
                    probabilities=probabilities,
                    replacement=replacement,
                    has_replacement=has_replacement,
                )
    return output
