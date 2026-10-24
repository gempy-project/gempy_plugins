"""Voxel-sampled geological anisotropy, in physical coordinates."""
from dataclasses import dataclass, field

import numpy as np

from ._utils import fingerprint


@dataclass
class GeologicalMetricField:
    """A regular lattice with a spatially varying, transversely isotropic metric.

    All flat arrays use NumPy C order. ``positions`` and ``gradients`` passed to
    :meth:`from_gradients` are in the same physical coordinate system. The metric
    has unit perpendicular scale; covariance range is applied by the simulator.
    Integer domain labels define impermeable boundaries. Invalid gradients are
    excluded, never replaced with an arbitrary orientation.
    """

    positions: np.ndarray
    normals: np.ndarray
    active_mask: np.ndarray
    domain_labels: np.ndarray
    grid_shape: tuple
    anisotropy_ratio: float
    invalid_gradient_count: int = 0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_gradients(cls, positions, gradients, mask, *, anisotropy_ratio=20.0,
                       domain_labels=None, gradient_tolerance=1e-12, metadata=None):
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 3 or any(size == 0 for size in mask.shape):
            raise ValueError("mask must be a nonempty three-dimensional voxel array")
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        gradients = np.asarray(gradients, dtype=np.float64).reshape(-1, 3)
        if positions.shape != (mask.size, 3) or gradients.shape != positions.shape:
            raise ValueError("positions and gradients must contain one XYZ vector per voxel")
        if not np.isfinite(positions).all():
            raise ValueError("positions must be finite")
        if not np.isfinite(anisotropy_ratio) or anisotropy_ratio < 1:
            raise ValueError("anisotropy_ratio must be finite and at least one")
        if not np.isfinite(gradient_tolerance) or gradient_tolerance < 0:
            raise ValueError("gradient_tolerance must be finite and nonnegative")
        if domain_labels is None:
            domain_labels = np.zeros(mask.size, dtype=np.int64)
        labels = np.asarray(domain_labels)
        if labels.size != mask.size or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("domain_labels must contain one integer per voxel")
        magnitude = np.linalg.norm(gradients, axis=1)
        finite = np.isfinite(gradients).all(axis=1) & np.isfinite(magnitude)
        selected = mask.ravel()
        scale = np.max(magnitude[finite & selected], initial=0.0)
        valid = finite & (magnitude > gradient_tolerance * scale) & (magnitude > 0)
        active = selected & valid
        if not active.any():
            raise ValueError("No active voxels have valid, nonzero gradients")
        normals = np.zeros_like(gradients)
        normals[active] = gradients[active] / magnitude[active, None]
        return cls(positions.copy(), normals, active, labels.ravel().astype(np.int64),
                   mask.shape, float(anisotropy_ratio), int(np.sum(selected & ~valid)),
                   dict(metadata or {}))

    @classmethod
    def from_gempy(cls, model, unit, **kwargs):
        from .gempy_adapter import metric_from_gempy
        return metric_from_gempy(model, unit, **kwargs)

    def tensors(self, flat_indices=None):
        """Materialize tensors for active voxels, or specified active flat indices."""
        if flat_indices is None:
            flat_indices = np.flatnonzero(self.active_mask)
        if not np.all(self.active_mask[flat_indices]):
            raise ValueError("Metric requested at inactive voxels")
        normal = self.normals[flat_indices]
        lateral = self.anisotropy_ratio ** -2
        return lateral * np.eye(3) + (1 - lateral) * normal[..., :, None] * normal[..., None, :]

    def edge_costs(self, start, end):
        """Integrate the endpoint-average tensor along straight lattice edges."""
        delta = self.positions[end] - self.positions[start]
        lateral = self.anisotropy_ratio ** -2
        normal_start = np.einsum('ij,ij->i', delta, self.normals[start])
        normal_end = np.einsum('ij,ij->i', delta, self.normals[end])
        squared = lateral * np.einsum('ij,ij->i', delta, delta)
        squared += (1 - lateral) * (normal_start ** 2 + normal_end ** 2) / 2
        return np.sqrt(squared)

    @property
    def fingerprint(self):
        return fingerprint(
            dict(shape=self.grid_shape, ratio=self.anisotropy_ratio, source=self.metadata),
            self.positions, self.normals, self.active_mask, self.domain_labels,
        )
