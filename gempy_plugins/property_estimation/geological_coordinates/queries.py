"""Conservative physical-coordinate queries of a cached voxel embedding."""
from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class GeologicalPoints:
    """Embedded sample locations, tied to one immutable geological embedding.

    ``positions`` retains the requested physical locations when supplied by query.
    Coordinates are interpolated, not simulated values. Only compatible active
    voxel centers contribute; boundary extrapolation and snapping are rejected.
    """
    coordinates: np.ndarray
    component_ids: np.ndarray
    embedding_fingerprint: str
    positions: np.ndarray | None = None

    @classmethod
    def from_nodes(cls, embedding, indices):
        indices = np.atleast_1d(indices)
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(embedding.coordinates))):
            raise ValueError("Expected nonempty valid integer active-node indices")
        return cls(embedding.coordinates[indices].copy(), embedding.component_ids[indices].copy(),
                   embedding.fingerprint)


def query_coordinates(embedding, graph, positions):
    """Trilinear coordinate interpolation inside compatible voxel-center cells.

    Every positive-weight corner must be active and in the same component. This
    deliberately rejects queries straddling holes, unit boundaries, or faults.
    It is an interpolation of the existing embedding, not a new graph-distance
    solve. No gradients are sampled and no nearest-node snapping is performed.
    """
    if graph.fingerprint != embedding.graph_fingerprint:
        raise ValueError("Embedding and query graph do not match")
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or not np.isfinite(positions).all():
        raise ValueError("positions must be a nonempty finite (points, 3) array")
    shape = graph.metric.grid_shape
    if len(shape) != 3:
        raise ValueError("Physical coordinate queries currently require a voxel embedding; use cell indices for tetrahedra")
    lattice = graph.metric.positions.reshape((*shape, 3))
    axes = [lattice[:, 0, 0, 0], lattice[0, :, 0, 1], lattice[0, 0, :, 2]]
    for axis, centers in enumerate(axes):
        view_shape = [1, 1, 1]
        view_shape[axis] = len(centers)
        if np.any(np.diff(centers) <= 0) or not np.allclose(
                lattice[..., axis], centers.reshape(view_shape), rtol=0, atol=1e-9):
            raise ValueError("Physical queries require an axis-aligned increasing rectilinear grid")
    lower, upper, fractions = [], [], []
    for axis, centers in enumerate(axes):
        x = positions[:, axis]
        if np.any(x < centers[0]) or np.any(x > centers[-1]):
            raise ValueError("Query lies outside the voxel-center bounds; extrapolation is not supported")
        if len(centers) == 1:
            lo = hi = np.zeros(len(x), dtype=int)
            fraction = np.zeros(len(x))
        else:
            hi = np.clip(np.searchsorted(centers, x, side='right'), 1, len(centers) - 1)
            lo = hi - 1
            fraction = (x - centers[lo]) / (centers[hi] - centers[lo])
        lower.append(lo)
        upper.append(hi)
        fractions.append(fraction)
    mapping = np.full(np.prod(shape), -1, dtype=int)
    mapping[graph.node_indices] = np.arange(len(graph.node_indices))
    coordinates = np.zeros((len(positions), embedding.coordinates.shape[1]), dtype=float)
    components = np.full(len(positions), -1, dtype=int)
    for corner in product((0, 1), repeat=3):
        ijk = [upper[a] if bit else lower[a] for a, bit in enumerate(corner)]
        weights = np.prod([fractions[a] if bit else 1 - fractions[a] for a, bit in enumerate(corner)], axis=0)
        selected = np.flatnonzero(weights > 0)
        nodes = mapping[np.ravel_multi_index(tuple(ijk), shape)[selected]]
        if np.any(nodes < 0):
            raise ValueError("Query requires inactive voxel centers; choose a location inside the selected unit")
        labels = embedding.component_ids[nodes]
        if np.any((components[selected] >= 0) & (components[selected] != labels)):
            raise ValueError("Query crosses geological components; interpolation across faults is not supported")
        components[selected] = labels
        coordinates[selected] += weights[selected, None] * embedding.coordinates[nodes]
    return GeologicalPoints(coordinates, components, embedding.fingerprint, positions.copy())
