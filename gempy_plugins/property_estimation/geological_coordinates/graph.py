"""Domain-safe sparse graphs and CPU reference shortest paths."""
from dataclasses import dataclass
from itertools import product
from math import gcd

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from ._utils import fingerprint, positive_integer
from .metric import GeologicalMetricField


@dataclass
class AnisotropicGraph:
    metric: GeologicalMetricField
    adjacency: csr_matrix
    node_indices: np.ndarray
    component_ids: np.ndarray
    connectivity: int | str
    stencil_radius: int = 1

    @classmethod
    def from_metric(cls, metric, *, connectivity=26, stencil_radius=1, dtype=np.float32):
        """Build 6/18/26-neighbor edges without diagonal corner cutting.

        Every voxel touched by a segment must be active and have the same
        domain label, including cells touched only at corners/edges. Radius > 1
        adds primitive lattice directions to measure/reduce directional bias.
        """
        if len(metric.grid_shape) != 3:
            raise ValueError("Voxel graph construction requires a three-dimensional lattice")
        if connectivity not in (6, 18, 26):
            raise ValueError("connectivity must be 6, 18, or 26")
        stencil_radius = positive_integer(stencil_radius, 'stencil_radius')
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError("Graph weights must use float32 or float64")
        nodes = np.flatnonzero(metric.active_mask)
        index_map = np.full(metric.active_mask.size, -1, dtype=np.int64)
        index_map[nodes] = np.arange(len(nodes))
        shape = metric.grid_shape
        lattice = np.arange(metric.active_mask.size).reshape(shape)
        starts, ends, weights = [], [], []
        max_axes = {6: 1, 18: 2, 26: 3}[connectivity]
        for offset in product(range(-stencil_radius, stencil_radius + 1), repeat=3):
            if offset <= (0, 0, 0) or np.count_nonzero(offset) > max_axes:
                continue
            if gcd(*offset) != 1 or any(abs(o) >= n for o, n in zip(offset, shape)):
                continue
            lower = tuple(max(0, -o) for o in offset)
            upper = tuple(min(n, n - o) for n, o in zip(shape, offset))
            a = lattice[tuple(slice(lo, hi) for lo, hi in zip(lower, upper))].ravel()
            b = lattice[tuple(slice(lo + o, hi + o) for lo, hi, o in zip(lower, upper, offset))].ravel()
            valid = metric.active_mask[a] & metric.active_mask[b]
            valid &= metric.domain_labels[a] == metric.domain_labels[b]
            for corner in _touched_voxels(offset):
                if corner in ((0, 0, 0), offset):
                    continue
                middle = lattice[tuple(slice(lo + o, hi + o)
                                       for lo, hi, o in zip(lower, upper, corner))].ravel()
                valid &= metric.active_mask[middle] & (metric.domain_labels[middle] == metric.domain_labels[a])
            a, b = a[valid], b[valid]
            cost = metric.edge_costs(a, b).astype(dtype)
            if not np.all(np.isfinite(cost) & (cost > 0)):
                raise ValueError("Edges must have finite, strictly positive costs; check voxel positions and precision")
            starts.append(index_map[a])
            ends.append(index_map[b])
            weights.append(cost)
        a = np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)
        b = np.concatenate(ends) if ends else np.empty(0, dtype=np.int64)
        costs = np.concatenate(weights) if weights else np.empty(0, dtype=dtype)
        graph = coo_matrix((np.concatenate((costs, costs)),
                            (np.concatenate((a, b)), np.concatenate((b, a)))),
                           shape=(len(nodes), len(nodes))).tocsr()
        _, components = connected_components(graph, directed=False)
        return cls(metric, graph, nodes, components, connectivity, stencil_radius)

    @classmethod
    def from_tetrahedral_mesh(cls, mesh, gradients, **kwargs):
        """Cell-centered graph through shared faces, blocked by units and faults.

        Gradients and optional scale maps have one entry per mesh tetrahedron.
        See ``tetrahedral.graph_from_tetrahedral_mesh`` for the edge-cost rule.
        """
        from .tetrahedral import graph_from_tetrahedral_mesh
        return graph_from_tetrahedral_mesh(mesh, gradients, **kwargs)

    @property
    def positions(self):
        return self.metric.positions[self.node_indices]

    @property
    def fingerprint(self):
        return fingerprint(dict(metric=self.metric.fingerprint, connectivity=self.connectivity, stencil_radius=self.stencil_radius),
                           self.adjacency.indptr, self.adjacency.indices, self.adjacency.data)

    def shortest_paths(self, sources):
        """Distances from active-node indices; unreachable nodes have distance inf.

        SciPy computes float64 distances even for float32 edge storage. Callers
        should batch sources to bound this temporary allocation.
        """
        sources = np.atleast_1d(sources)
        if not np.issubdtype(sources.dtype, np.integer) or np.any(sources < 0) or np.any(sources >= len(self.node_indices)):
            raise ValueError("sources must be valid integer active-node indices")
        return dijkstra(self.adjacency, directed=False, indices=sources)


def _touched_voxels(offset):
    """Supercover of a center-to-center lattice segment, including boundary ties."""
    times = {0., 1.}
    for step in offset:
        times.update((i + .5) / abs(step) for i in range(abs(step)))
    result = set()
    for t in sorted(times):
        point = t * np.asarray(offset)
        choices = []
        for value in point:
            shifted = value + .5
            if np.isclose(shifted, round(shifted), rtol=0., atol=1e-12):
                choices.append((int(round(shifted)) - 1, int(round(shifted))))
            else:
                choices.append((int(np.floor(shifted)),))
        result.update(product(*choices))
    return sorted(result)
