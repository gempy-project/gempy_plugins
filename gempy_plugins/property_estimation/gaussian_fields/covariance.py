"""Stationary covariance conventions in geological coordinate space."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianCovariance:
    """C(h) = variance * exp(-h**2 / (2 * length_scale**2)).

    Local physical scales are length_scale * continuity_scale perpendicular to
    bedding, and additionally multiplied by anisotropy_ratio parallel to bedding.
    These describe the intended local metric; embedding distortion is separate.
    """
    length_scale: float = 1.0
    variance: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError("length_scale must be finite and positive")
        if not np.isfinite(self.variance) or self.variance <= 0:
            raise ValueError("variance must be finite and positive")

    def __call__(self, distances):
        return self.variance * np.exp(-.5 * (np.asarray(distances) / self.length_scale) ** 2)

    def sample_frequencies(self, count, dimensions, seed):
        return np.random.default_rng(seed).normal(size=(count, dimensions)) / self.length_scale
