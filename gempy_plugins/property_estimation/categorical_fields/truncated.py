"""Ordered, unconditional truncated-Gaussian facies using a fixed spectral basis."""
import numpy as np
from scipy.special import ndtri

from ..gaussian_fields import GaussianFieldSimulator
from ..geological_coordinates._utils import positive_integer
from .results import FaciesRealizations


class TruncatedGaussianFacies:
    """Threshold one Gaussian field to obtain ordered facies.

    Proportions are marginal ensemble targets, not exact per-realization quotas.
    A continuous path between nonadjacent categories must pass through intermediate
    categories; voxel sampling can miss those intermediate intervals. Components
    retain the Gaussian simulator's independent coefficient draws.
    """

    def __init__(self, simulator, proportions, *, names=None):
        if not isinstance(simulator, GaussianFieldSimulator):
            raise TypeError("simulator must be a GaussianFieldSimulator")
        probabilities = np.asarray(proportions, dtype=float)
        if (probabilities.ndim != 1 or not len(probabilities) or not np.isfinite(probabilities).all()
                or np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1., atol=1e-6, rtol=0)):
            raise ValueError("proportions must be a positive finite vector summing to one")
        probabilities = probabilities / probabilities.sum()
        names = tuple(f'facies_{i}' for i in range(len(probabilities))) if names is None else tuple(names)
        if (len(names) != len(probabilities) or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("names must contain one unique nonempty string per facies")
        self.simulator = simulator
        self.proportions = tuple(probabilities.tolist())
        self.names = names
        self.thresholds = ndtri(np.cumsum(probabilities)[:-1])
        if not np.isfinite(self.thresholds).all() or np.any(np.diff(self.thresholds) <= 0):
            raise ValueError("proportions produce indistinguishable or infinite thresholds")

    def iter_simulate(self, **kwargs):
        """Yield integer batches; accepts the Gaussian simulator's chunk/seed options.

        Latent mean is fixed to zero. Thresholds are scaled by the configured
        standard deviation, so changing latent variance does not change proportions.
        """
        if 'mean' in kwargs:
            raise ValueError("Facies simulation fixes latent mean to zero")
        torch = self.simulator.torch
        for latent in self.simulator.iter_simulate(mean=0., **kwargs):
            thresholds = torch.as_tensor(self.thresholds * np.sqrt(self.simulator.covariance.variance),
                                         dtype=latent.values.dtype, device=latent.values.device)
            codes = torch.bucketize(latent.values.contiguous(), thresholds, right=True)
            metadata = dict(latent.metadata, method='unconditional_truncated_gaussian',
                            standardized_thresholds=self.thresholds.tolist())
            yield FaciesRealizations(codes, latent.embedding, self.names, self.proportions,
                                     latent.realization_start, metadata)

    def simulate(self, *, realizations=1, output_device='cpu', **kwargs):
        """Materialize an ensemble; iter_simulate keeps output memory bounded."""
        realizations = positive_integer(realizations, 'realizations')
        torch = self.simulator.torch
        values = torch.empty((realizations, len(self.simulator.embedding.coordinates)),
                             dtype=torch.long, device=output_device)
        metadata = {}
        for batch in self.iter_simulate(realizations=realizations, output_device=output_device, **kwargs):
            first = batch.realization_start
            values[first:first + len(batch.values)] = batch.values
            metadata = batch.metadata
        return FaciesRealizations(values, self.simulator.embedding, self.names, self.proportions, metadata=metadata)
