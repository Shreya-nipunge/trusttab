"""Stable independent random-state derivation for experiment components."""

import hashlib

import numpy as np


def derive_seed(base_seed: int, component: str, *keys: object) -> int:
    """Derive a stable uint32 seed for one named stochastic component.

    A cryptographic digest, rather than Python's process-randomized ``hash``,
    makes results stable across processes and Python runs. Components such as
    splitting, label noise, model initialization, and feature shifts receive
    separate streams even when they share a configured repetition seed.
    """

    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, (int, np.integer))
        or not 0 <= int(base_seed) <= np.iinfo(np.uint32).max
    ):
        raise ValueError("base_seed must be an integer between 0 and 2**32 - 1.")
    if not isinstance(component, str) or not component:
        raise ValueError("component must be a non-empty string.")

    payload = "\0".join((component, *(str(key) for key in keys))).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    digest_words = [
        int.from_bytes(digest[offset : offset + 4], "little")
        for offset in range(0, 16, 4)
    ]
    seed_sequence = np.random.SeedSequence([int(base_seed), *digest_words])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
