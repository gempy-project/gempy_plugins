"""Landmark MDS with a reusable, component-aware coordinate representation."""
from dataclasses import dataclass, field
import json

import numpy as np
from scipy.sparse.csgraph import dijkstra

from ._utils import fingerprint, positive_integer
from .graph import AnisotropicGraph
from .metric import GeologicalMetricField


@dataclass
class LandmarkProjection:
    landmark_indices: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    squared_row_means: np.ndarray
    squared_grand_mean: float

    @property
    def dimensions(self):
        return self.eigenvectors.shape[1]

    def transform(self, distances):
        """Project (..., landmarks) distances from this component's landmarks."""
        distances = np.asarray(distances, dtype=np.float64)
        if distances.shape[-1] != len(self.landmark_indices) or not np.isfinite(distances).all() or np.any(distances < 0):
            raise ValueError("Expected finite, nonnegative distances to this component's landmarks")
        squared = distances ** 2
        centered = -.5 * (squared - squared.mean(axis=-1, keepdims=True)
                          - self.squared_row_means + self.squared_grand_mean)
        return (centered @ self.eigenvectors) / np.sqrt(self.eigenvalues[:self.dimensions])

    @property
    def diagnostics(self):
        positive = self.eigenvalues[self.eigenvalues > 0].sum()
        negative = -self.eigenvalues[self.eigenvalues < 0].sum()
        return dict(effective_dimensions=self.dimensions,
                    retained_positive_fraction=float(self.eigenvalues[:self.dimensions].sum() / positive) if positive else 1.0,
                    negative_spectrum_fraction=float(negative / (positive + negative)) if positive + negative else 0.0)


@dataclass
class LandmarkGeologicalEmbedding:
    """Coordinates indexed by active node, with independent connected components.

    Coordinates have the requested width. Components with lower positive rank
    have trailing zero columns; ``projections`` records their effective rank.
    Distances between different components are undefined, even if their stored
    coordinates happen to coincide. Treat arrays as immutable cache artifacts.
    ``grid_shape`` is the storage shape: a voxel lattice, or ``(tetrahedron_count,)``
    for a cell-centered mesh embedding. ``node_indices`` maps to that storage.
    """
    coordinates: np.ndarray
    component_ids: np.ndarray
    node_indices: np.ndarray
    grid_shape: tuple
    projections: list
    graph_fingerprint: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_graph(cls, graph, *, landmarks=32, dimensions=6, seed=0,
                   placement='farthest', distance_dtype=np.float32, chunk_size=65536):
        landmarks = positive_integer(landmarks, 'landmarks')
        dimensions = positive_integer(dimensions, 'dimensions')
        chunk_size = positive_integer(chunk_size, 'chunk_size')
        if placement not in ('farthest', 'random'):
            raise ValueError("placement must be 'farthest' or 'random'")
        distance_dtype = np.dtype(distance_dtype)
        if distance_dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError("distance_dtype must be float32 or float64")
        rng = np.random.default_rng(seed)
        coordinates = np.zeros((len(graph.node_indices), dimensions), dtype=np.float32)
        projections = []
        for component in range(int(graph.component_ids.max()) + 1):
            nodes = np.flatnonzero(graph.component_ids == component)
            adjacency = graph.adjacency[nodes][:, nodes]
            count = min(landmarks, len(nodes))
            if count == 1 and len(nodes) > 1:
                raise ValueError("At least two landmarks are required for a non-singleton component")
            distances = np.empty((count, len(nodes)), dtype=distance_dtype)
            selected = np.empty(count, dtype=np.int64)
            nearest = np.full(len(nodes), np.inf)
            random_nodes = rng.choice(len(nodes), count, replace=False)
            next_node = int(random_nodes[0])
            for i in range(count):
                selected[i] = next_node
                row = dijkstra(adjacency, directed=False, indices=next_node)
                distances[i] = row
                if not np.isfinite(distances[i]).all():
                    raise ValueError("Nonfinite landmark distances; check graph connectivity and precision")
                nearest = np.minimum(nearest, row)
                nearest[selected[:i + 1]] = -np.inf
                if i + 1 < count:
                    next_node = int(np.argmax(nearest)) if placement == 'farthest' else int(random_nodes[i + 1])
            squared = distances[:, selected].astype(np.float64) ** 2
            squared = (squared + squared.T) / 2
            means = squared.mean(axis=1)
            grand = float(means.mean())
            gram = -.5 * (squared - means[:, None] - means[None, :] + grand)
            values, vectors = np.linalg.eigh(gram)
            values, vectors = values[::-1], vectors[:, ::-1]
            # Squaring float32 distances can create small spurious positive
            # eigenvalues even for an exactly low-dimensional Euclidean metric.
            relative_tolerance = max(1e-10, count * np.finfo(distance_dtype).eps)
            tolerance = max(float(values[0]), 0.0) * relative_tolerance
            rank = min(dimensions, int(np.sum(values > tolerance)))
            projection = LandmarkProjection(nodes[selected], values, vectors[:, :rank], means, grand)
            for start in range(0, len(nodes), chunk_size):
                stop = min(start + chunk_size, len(nodes))
                coordinates[nodes[start:stop], :rank] = projection.transform(distances[:, start:stop].T)
            projections.append(projection)
        metadata = dict(graph.metric.metadata, landmarks=landmarks, dimensions=dimensions,
                        seed=int(seed), placement=placement, distance_dtype=distance_dtype.name,
                        connectivity=graph.connectivity, stencil_radius=graph.stencil_radius,
                        **graph.metric.scale_metadata,
                        invalid_gradient_count=graph.metric.invalid_gradient_count)
        return cls(coordinates, graph.component_ids.copy(), graph.node_indices.copy(),
                   graph.metric.grid_shape, projections, graph.fingerprint, metadata)

    @classmethod
    def from_constant_metric(cls, graph, *, normal_tolerance=1e-6):
        """Analytical reference for constant tensors, with independent components.

        Within a component these coordinates ignore holes and nonconvex boundaries.
        Use them to diagnose graph/embedding error, not to bypass geological barriers.
        """
        from .analytical import constant_metric_coordinates
        coordinates = constant_metric_coordinates(graph.metric, normal_tolerance=normal_tolerance)
        metadata = dict(graph.metric.metadata, coordinate_method='constant_metric',
                        **graph.metric.scale_metadata,
                        reference_distance='unconstrained_straight_line')
        return cls(coordinates.astype(np.float32), graph.component_ids.copy(), graph.node_indices.copy(),
                   graph.metric.grid_shape, [], graph.fingerprint, metadata)

    @classmethod
    def from_gempy(cls, model, unit, *, anisotropy_ratio=20.0, landmarks=32,
                   dimensions=6, connectivity=26, stencil_radius=1, seed=0, structural_group=None, fault_block=None,
                   continuity_scale=1.0):
        metric = GeologicalMetricField.from_gempy(
            model, unit, anisotropy_ratio=anisotropy_ratio,
            structural_group=structural_group, fault_block=fault_block,
            continuity_scale=continuity_scale,
        )
        graph = AnisotropicGraph.from_metric(metric, connectivity=connectivity, stencil_radius=stencil_radius)
        return cls.from_graph(graph, landmarks=landmarks, dimensions=dimensions, seed=seed)

    @property
    def diagnostics(self):
        return [projection.diagnostics for projection in self.projections]

    def query(self, graph, positions):
        """Embed physical locations using domain-safe voxel-center interpolation."""
        from .queries import query_coordinates
        return query_coordinates(self, graph, positions)

    @property
    def fingerprint(self):
        arrays = [self.coordinates, self.component_ids, self.node_indices]
        for p in self.projections:
            arrays.extend((p.landmark_indices, p.eigenvalues, p.eigenvectors, p.squared_row_means,
                           np.array(p.squared_grand_mean)))
        return fingerprint(dict(graph=self.graph_fingerprint, shape=self.grid_shape, metadata=self.metadata), *arrays)

    def save(self, path):
        """Save a versioned, pickle-free embedding, including projection factors."""
        header = dict(version=1, grid_shape=self.grid_shape, graph_fingerprint=self.graph_fingerprint,
                      metadata=self.metadata, components=len(self.projections), fingerprint=self.fingerprint)
        arrays = dict(coordinates=self.coordinates, component_ids=self.component_ids, node_indices=self.node_indices)
        for i, p in enumerate(self.projections):
            for name in ('landmark_indices', 'eigenvalues', 'eigenvectors', 'squared_row_means', 'squared_grand_mean'):
                arrays[f'{i}_{name}'] = np.asarray(getattr(p, name))
        with open(path, 'wb') as stream:
            np.savez_compressed(stream, header=json.dumps(header, sort_keys=True), **arrays)

    @classmethod
    def load(cls, path, *, expected_graph_fingerprint=None):
        """Load a standalone artifact; supply a graph fingerprint to check reuse."""
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data['header']))
            if header['version'] != 1:
                raise ValueError("Unsupported embedding cache version")
            if expected_graph_fingerprint is not None and header['graph_fingerprint'] != expected_graph_fingerprint:
                raise ValueError("Embedding cache does not match the requested geological graph")
            projections = [LandmarkProjection(
                data[f'{i}_landmark_indices'], data[f'{i}_eigenvalues'], data[f'{i}_eigenvectors'],
                data[f'{i}_squared_row_means'], float(data[f'{i}_squared_grand_mean']),
            ) for i in range(header['components'])]
            result = cls(data['coordinates'], data['component_ids'], data['node_indices'],
                         tuple(header['grid_shape']), projections, header['graph_fingerprint'], header['metadata'])
        if result.fingerprint != header['fingerprint']:
            raise ValueError("Embedding cache checksum mismatch")
        return result


GeologicalEmbedding = LandmarkGeologicalEmbedding
