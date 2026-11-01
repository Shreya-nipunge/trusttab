"""Reproducible label-noise injection utilities."""

import numpy as np
import pandas as pd


def inject_symmetric_label_noise(
    y: pd.Series | np.ndarray,
    noise_rate: float,
    seed: int,
) -> pd.Series | np.ndarray:
    """
    Randomly replace labels with a different class label.

    Each sample has the same probability of being selected for
    corruption. A selected sample is assigned a different class.

    Parameters
    ----------
    y:
        Original class labels.
    noise_rate:
        Fraction of labels to corrupt, between 0 and 1.
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

    n_samples = len(original)
    n_corrupt = int(round(noise_rate * n_samples))

    if n_corrupt > 0:
        selected_indices = rng.choice(
            n_samples,
            size=n_corrupt,
            replace=False,
        )

        for index in selected_indices:
            alternative_classes = classes[classes != original[index]]

            noisy[index] = rng.choice(alternative_classes)

    if isinstance(y, pd.Series):
        return pd.Series(
            noisy,
            index=y.index,
            name=y.name,
        )

    return noisy