import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from gempy_plugins.property_estimation.geological_coordinates import (
    GeologicalMetricField, AnisotropicGraph, LandmarkGeologicalEmbedding,
)


def test_landmark_centering_reconstructs_euclidean_geometry_and_cache(tmp_path):
    rng = np.random.default_rng(12)
    xyz = rng.normal(size=(30, 3))
    metric = GeologicalMetricField.from_gradients(xyz, np.tile([0, 0, 1], (30, 1)), np.ones((30, 1, 1), bool), anisotropy_ratio=1)
    distances = cdist(xyz, xyz)
    graph = AnisotropicGraph(metric, csr_matrix(distances), np.arange(30), np.zeros(30, int), 26)
    embedding = LandmarkGeologicalEmbedding.from_graph(graph, landmarks=12, dimensions=6, distance_dtype=np.float64, chunk_size=7)
    assert embedding.projections[0].dimensions == 3
    single_precision = LandmarkGeologicalEmbedding.from_graph(graph, landmarks=12, dimensions=6)
    assert single_precision.projections[0].dimensions == 3
    np.testing.assert_allclose(cdist(single_precision.coordinates, single_precision.coordinates), distances, atol=2e-6)
    np.testing.assert_allclose(cdist(embedding.coordinates, embedding.coordinates), distances, atol=3e-7)
    projection = embedding.projections[0]
    np.testing.assert_allclose(projection.transform(distances[:, projection.landmark_indices]), embedding.coordinates[:, :3], atol=2e-7)
    path = tmp_path / 'embedding.npz'
    embedding.save(path)
    loaded = LandmarkGeologicalEmbedding.load(path, expected_graph_fingerprint=graph.fingerprint)
    assert loaded.fingerprint == embedding.fingerprint
    np.testing.assert_array_equal(loaded.coordinates, embedding.coordinates)
    metric.anisotropy_ratio = 2
    with pytest.raises(ValueError, match='does not match'):
        LandmarkGeologicalEmbedding.load(path, expected_graph_fingerprint=graph.fingerprint)


def test_disconnected_components_and_singletons_have_finite_independent_embeddings():
    shape = (7, 1, 1)
    xyz = np.indices(shape).reshape(3, -1).T.astype(float)
    mask = np.ones(shape, bool)
    mask[[1, 5]] = False
    metric = GeologicalMetricField.from_gradients(xyz, np.tile([0, 0, 1], (7, 1)), mask)
    graph = AnisotropicGraph.from_metric(metric)
    embedding = LandmarkGeologicalEmbedding.from_graph(graph)
    assert len(embedding.projections) == 3
    assert [p.dimensions for p in embedding.projections] == [0, 1, 0]
    assert np.isfinite(embedding.coordinates).all()
    np.testing.assert_allclose(cdist(embedding.coordinates[1:4], embedding.coordinates[1:4]),
                               cdist(xyz[2:5] / 20, xyz[2:5] / 20), atol=1e-7)
