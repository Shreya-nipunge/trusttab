"""Reproducible label-noise injection utilities."""

import numpy as np
import pandas as pd


def inject_symmetric_label_noise(
    y: pd.Series | np.ndarray,
    noise_rate: float,
    seed: int,
) -> pd.Series | np.ndarray:
    """
    Independently flip each label with probability ``noise_rate``.

    For binary labels, a selected label changes to the other class. For
    multiclass labels, it changes uniformly to one of the other observed
    classes. The realized number of flips is random (Binomial for binary
    labels), with expected value ``noise_rate * len(y)``.

    Parameters
    ----------
    y:
        Original class labels.
    noise_rate:
        Independent per-label flip probability, between 0 and 1.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Noisy labels in the same container type as the input.
    """

    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("Labels must be a pandas Series or NumPy array.")

    if len(y) == 0:
        raise ValueError("Labels must not be empty.")

    if np.asarray(y).ndim != 1:
        raise ValueError("Labels must be one-dimensional.")

    if not isinstance(noise_rate, (int, float)) or isinstance(
        noise_rate, bool
    ):
        raise TypeError("Noise rate must be a number.")

    if not 0 <= noise_rate <= 1:
        raise ValueError("Noise rate must be between 0 and 1.")

    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError("Seed must be an integer.")

    if pd.isna(y).any():
        raise ValueError("Labels must not contain missing values.")

    original = np.asarray(y).copy()
    classes = np.unique(original)

    if len(classes) < 2 and noise_rate > 0:
        raise ValueError(
            "At least two distinct classes are required to inject noise."
        )

    rng = np.random.default_rng(seed)
    noisy = original.copy()
    if noise_rate == 0:
        flip_mask = np.zeros(len(original), dtype=bool)
    elif noise_rate == 1:
        flip_mask = np.ones(len(original), dtype=bool)
    else:
        flip_mask = rng.random(len(original)) < noise_rate

    for index in np.flatnonzero(flip_mask):
        alternative_classes = classes[classes != original[index]]
        noisy[index] = rng.choice(alternative_classes)

    if isinstance(y, pd.Series):
        return pd.Series(
            noisy,
            index=y.index,
            name=y.name,
        )

    return noisy
