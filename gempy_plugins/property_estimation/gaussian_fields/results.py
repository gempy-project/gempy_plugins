"""Batched simulation results and existing property-estimation interoperability."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geological_coordinates.embedding import LandmarkGeologicalEmbedding


@dataclass
class GaussianRealizations:
    """``values`` is a Torch tensor shaped (realizations, active_nodes)."""
    values: Any
    embedding: LandmarkGeologicalEmbedding
    realization_start: int = 0
    metadata: dict = field(default_factory=dict)

    def to_grid(self, realization=0):
        if not isinstance(realization, (int, np.integer)) or not 0 <= realization < len(self.values):
            raise IndexError("realization is outside this batch")
        selected = self.values[realization].detach().cpu().numpy()
        values = np.full(np.prod(self.embedding.grid_shape), np.nan, dtype=selected.dtype)
        values[self.embedding.node_indices] = selected
        return values.reshape(self.embedding.grid_shape)

    def to_cells(self, realization=0):
        """Full tetrahedron-indexed values, NaN outside the active mesh cells."""
        if self.embedding.metadata.get('sampling') != 'tetrahedron_centroids':
            raise ValueError("Cell export requires a tetrahedral embedding")
        return self.to_grid(realization)

    def to_property_field(self, realization=0):
        """Select one realization for existing property-volume plotting functions."""
        from ..kriging import PropertyField
        domains = self.embedding.metadata.get('domain_keys')
        if not domains:
            raise ValueError("PropertyField conversion requires GemPy domain_keys metadata")
        values = self.to_grid(realization)
        return PropertyField(values, np.full(values.shape, np.nan), [tuple(key) for key in domains])


@dataclass
class ConditionalGaussianRealizations(GaussianRealizations):
    """Conditional grid draws plus latent values at the actual observation locations.

    Observation values have shape (realizations, observations), in input order.
    They exclude measurement noise; exact observations are honored there even
    when their physical positions do not coincide with voxel centers.
    """
    observation_values: Any = None


@dataclass
class GaussianMoments:
    """Analytical posterior mean and marginal latent variance on active nodes."""
    mean: Any
    variance: Any
    embedding: LandmarkGeologicalEmbedding
    metadata: dict = field(default_factory=dict)

    def to_grids(self):
        arrays = []
        for tensor in (self.mean, self.variance):
            values = np.full(np.prod(self.embedding.grid_shape), np.nan)
            values[self.embedding.node_indices] = tensor.detach().cpu().numpy()
            arrays.append(values.reshape(self.embedding.grid_shape))
        return tuple(arrays)

    def to_property_field(self):
        from ..kriging import PropertyField
        domains = self.embedding.metadata.get('domain_keys')
        if not domains:
            raise ValueError("PropertyField conversion requires GemPy domain_keys metadata")
        mean, variance = self.to_grids()
        return PropertyField(mean, variance, [tuple(key) for key in domains])
