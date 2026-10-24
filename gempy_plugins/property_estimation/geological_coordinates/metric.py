"""Sampled geological anisotropy, in physical coordinates."""
from dataclasses import dataclass, field

import numpy as np

from ._utils import fingerprint


@dataclass
class GeologicalMetricField:
    """A spatially varying, transversely isotropic metric on voxels or points.

    All flat arrays use NumPy C order. ``positions`` and ``gradients`` passed to
    :meth:`from_gradients` are in the same physical coordinate system. The metric
    has perpendicular scale ``continuity_scale`` and parallel scale
    ``continuity_scale * anisotropy_ratio``. Both accept scalars or full-grid
    arrays. The simulator's covariance length scale multiplies these local scales.
    Integer domain labels define impermeable boundaries. Invalid gradients are
    excluded, never replaced with an arbitrary orientation.
    ``from_points`` uses one-dimensional storage, e.g. one entry per tetrahedron;
    it does not imply a regular lattice or enable voxel-coordinate queries.
    """

    positions: np.ndarray
    normals: np.ndarray
    active_mask: np.ndarray
    domain_labels: np.ndarray
    grid_shape: tuple
    anisotropy_ratio: float | np.ndarray
    invalid_gradient_count: int = 0
    metadata: dict = field(default_factory=dict)
    continuity_scale: float | np.ndarray = 1.0

    @classmethod
    def from_gradients(cls, positions, gradients, mask, *, anisotropy_ratio=20.0,
                        domain_labels=None, gradient_tolerance=1e-12, metadata=None, continuity_scale=1.0):
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 3 or any(size == 0 for size in mask.shape):
            raise ValueError("mask must be a nonempty three-dimensional voxel array")
        return cls._from_samples(positions, gradients, mask, anisotropy_ratio=anisotropy_ratio,
                                 domain_labels=domain_labels, gradient_tolerance=gradient_tolerance,
                                 metadata=metadata, continuity_scale=continuity_scale)

    @classmethod
    def from_points(cls, positions, gradients, *, mask=None, anisotropy_ratio=20.0,
                    domain_labels=None, gradient_tolerance=1e-12, metadata=None, continuity_scale=1.0):
        """Metric at arbitrary physical points, with scalar or per-point scales."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
            raise ValueError("positions must be a nonempty (points, 3) array")
        mask = np.ones(len(positions), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if mask.shape != (len(positions),):
            raise ValueError("mask must contain one entry per point")
        return cls._from_samples(positions, gradients, mask, anisotropy_ratio=anisotropy_ratio,
                                 domain_labels=domain_labels, gradient_tolerance=gradient_tolerance,
                                 metadata=metadata, continuity_scale=continuity_scale)

    @classmethod
    def _from_samples(cls, positions, gradients, mask, *, anisotropy_ratio, domain_labels,
                      gradient_tolerance, metadata, continuity_scale):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        gradients = np.asarray(gradients, dtype=np.float64).reshape(-1, 3)
        if positions.shape != (mask.size, 3) or gradients.shape != positions.shape:
            raise ValueError("positions and gradients must contain one XYZ vector per sample")
        if not np.isfinite(positions).all():
            raise ValueError("positions must be finite")
        anisotropy_ratio = _voxel_parameter(anisotropy_ratio, mask.shape, 'anisotropy_ratio', minimum=1.)
        continuity_scale = _voxel_parameter(continuity_scale, mask.shape, 'continuity_scale', minimum=0.)
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
                   mask.shape, anisotropy_ratio, int(np.sum(selected & ~valid)),
                   dict(metadata or {}), continuity_scale)

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
        parallel, perpendicular = self.local_scales(flat_indices)
        lateral, vertical = parallel ** -2, perpendicular ** -2
        return (lateral[..., None, None] * np.eye(3)
                + (vertical - lateral)[..., None, None] * normal[..., :, None] * normal[..., None, :])

    def local_scales(self, flat_indices=None):
        """Parallel/perpendicular range multipliers at full-grid flat indices."""
        if flat_indices is None:
            flat_indices = np.flatnonzero(self.active_mask)
        ratio = np.broadcast_to(self.anisotropy_ratio, (self.active_mask.size,))[flat_indices]
        scale = np.broadcast_to(self.continuity_scale, (self.active_mask.size,))[flat_indices]
        return ratio * scale, scale

    @property
    def scale_metadata(self):
        """JSON-safe summaries; complete maps are included in the metric fingerprint."""
        result = {}
        for name in ('anisotropy_ratio', 'continuity_scale'):
            parameter = np.asarray(getattr(self, name))
            selected = np.broadcast_to(parameter, (self.active_mask.size,))[self.active_mask]
            result[name] = (float(parameter) if parameter.ndim == 0 else
                            dict(kind='voxel_map' if len(self.grid_shape) == 3 else 'sample_map',
                                 minimum=float(selected.min()), maximum=float(selected.max())))
        return result

    def edge_costs(self, start, end):
        """Integrate the endpoint-average tensor along straight lattice edges."""
        delta = self.positions[end] - self.positions[start]
        normal_start = np.einsum('ij,ij->i', delta, self.normals[start])
        normal_end = np.einsum('ij,ij->i', delta, self.normals[end])
        parallel_a, perpendicular_a = self.local_scales(start)
        parallel_b, perpendicular_b = self.local_scales(end)
        a, b = parallel_a ** -2, parallel_b ** -2
        squared = .5 * (a + b) * np.einsum('ij,ij->i', delta, delta)
        squared += .5 * ((perpendicular_a ** -2 - a) * normal_start ** 2
                         + (perpendicular_b ** -2 - b) * normal_end ** 2)
        return np.sqrt(squared)

    @property
    def fingerprint(self):
        return fingerprint(
            dict(shape=self.grid_shape, scales=self.scale_metadata, source=self.metadata),
            self.positions, self.normals, self.active_mask, self.domain_labels,
            np.asarray(self.anisotropy_ratio), np.asarray(self.continuity_scale),
        )


def _voxel_parameter(value, shape, name, *, minimum):
    value = np.asarray(value, dtype=float)
    if value.ndim and value.shape not in (shape, (int(np.prod(shape)),)):
        raise ValueError(f"{name} must be scalar or contain one value per full-grid voxel")
    invalid = value <= 0 if minimum == 0 else value < minimum
    if not np.isfinite(value).all() or np.any(invalid):
        requirement = 'positive' if minimum == 0 else 'at least one'
        raise ValueError(f"{name} must be finite and {requirement}")
    return float(value) if value.ndim == 0 else value.ravel().copy()
