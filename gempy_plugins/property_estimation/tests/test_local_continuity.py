import numpy as np
import pytest
from scipy.spatial.distance import cdist

from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding
from gempy_plugins.property_estimation.geological_coordinates.analytical import constant_metric_coordinates
from gempy_plugins.property_estimation.geological_coordinates.diagnostics import embedding_quality, graph_refinement_quality


def test_variable_scales_tensors_and_endpoint_integration():
    shape = (5, 3, 2)
    xyz = np.indices(shape).reshape(3, -1).T.astype(float)
    gradients = np.column_stack((xyz[:, 0] / 5, np.zeros(len(xyz)), np.ones(len(xyz))))
    ratio, scale = 2 + xyz[:, 0], .5 + xyz[:, 1] / 2
    metric = GeologicalMetricField.from_gradients(xyz, gradients, np.ones(shape, bool),
                                                 anisotropy_ratio=ratio.reshape(shape), continuity_scale=scale)
    tensors = metric.tensors()
    expected_eigenvalues = np.sort(np.column_stack(((ratio * scale) ** -2, (ratio * scale) ** -2, scale ** -2)))
    np.testing.assert_allclose(np.linalg.eigvalsh(tensors), expected_eigenvalues, atol=1e-14)
    start, end = np.array([0, 3, 8]), np.array([5, 6, 12])
    delta = xyz[end] - xyz[start]
    expected = np.sqrt(np.einsum('ni,nij,nj->n', delta, (tensors[start] + tensors[end]) / 2, delta))
    np.testing.assert_allclose(metric.edge_costs(start, end), expected)
    np.testing.assert_allclose(metric.edge_costs(end, start), expected)
    with pytest.raises(ValueError, match='constant metric'):
        constant_metric_coordinates(metric)


def test_short_to_long_continuity_is_encoded_by_isomap_and_covariance(tmp_path):
    torch = pytest.importorskip('torch')
    from gempy_plugins.property_estimation.gaussian_fields import GaussianCovariance, GaussianFieldSimulator
    shape = (41, 1, 1)
    xyz = np.indices(shape).reshape(3, -1).T.astype(float)
    scale = .5 + xyz[:, 0] / 40
    metric = GeologicalMetricField.from_gradients(xyz, np.tile([0, 0, 1], (len(xyz), 1)), np.ones(shape, bool),
                                                 anisotropy_ratio=3, continuity_scale=scale)
    graph = AnisotropicGraph.from_metric(metric)
    embedding = GeologicalEmbedding.from_graph(graph, landmarks=12, dimensions=3, distance_dtype='float64')
    # On a line, geodesics are exactly cumulative endpoint-weighted edge lengths.
    increments = np.sqrt(.5 * ((3 * scale[:-1]) ** -2 + (3 * scale[1:]) ** -2))
    q = np.r_[0, np.cumsum(increments)]
    np.testing.assert_allclose(cdist(embedding.coordinates, embedding.coordinates), np.abs(q[:, None] - q), atol=1e-6)
    quality = embedding_quality(graph, embedding, length_scale=1)
    assert quality['passes_provisional_targets']
    simulator = GaussianFieldSimulator(embedding, GaussianCovariance(1), n_frequencies=4096, dtype='float64')
    covariance = simulator.finite_covariance([1, 4, 36, 39]).numpy()
    assert covariance[2, 3] > covariance[0, 1] + .4
    # Fixed amplitude: changing continuity does not change point variance.
    np.testing.assert_allclose(covariance.diagonal(), 1, atol=1e-12)
    filename = tmp_path / 'local_geometry.npz'
    embedding.save(filename)
    restored = GeologicalEmbedding.load(filename, expected_graph_fingerprint=graph.fingerprint)
    np.testing.assert_array_equal(restored.coordinates, embedding.coordinates)
    assert restored.metadata['continuity_scale']['maximum'] == 1.5
    original = metric.fingerprint
    metric.continuity_scale[20] += .1
    assert metric.fingerprint != original
    with pytest.raises(ValueError, match='does not match'):
        GeologicalEmbedding.load(filename, expected_graph_fingerprint=graph.fingerprint)


def test_uniform_maps_equal_scalar_metrics_and_analytical_coordinates():
    shape = (4, 3, 2)
    xyz = np.indices(shape).reshape(3, -1).T.astype(float)
    arguments = (xyz, np.tile([0, 0, 1], (len(xyz), 1)), np.ones(shape, bool))
    scalar = GeologicalMetricField.from_gradients(*arguments, anisotropy_ratio=5, continuity_scale=2)
    mapped = GeologicalMetricField.from_gradients(*arguments, anisotropy_ratio=np.full(shape, 5), continuity_scale=np.full(shape, 2))
    np.testing.assert_array_equal(mapped.tensors(), scalar.tensors())
    np.testing.assert_allclose(constant_metric_coordinates(mapped), (xyz - xyz.mean(axis=0)) / [10, 10, 2])
    np.testing.assert_array_equal(AnisotropicGraph.from_metric(mapped).adjacency.toarray(),
                                  AnisotropicGraph.from_metric(scalar).adjacency.toarray())


def test_local_scales_must_match_in_refinement_comparison():
    def make(factor, offset=0):
        shape = (5 * factor, 1, 1)
        xyz = np.indices(shape).reshape(3, -1).T.astype(float)
        xyz[:, 0] = (xyz[:, 0] + .5) / factor
        metric = GeologicalMetricField.from_gradients(xyz, np.tile([0, 0, 1], (len(xyz), 1)), np.ones(shape, bool),
                                                     anisotropy_ratio=2, continuity_scale=1 + xyz[:, 0] / 5 + offset)
        return AnisotropicGraph.from_metric(metric)
    report = graph_refinement_quality(make(1), make(3), length_scale=2)
    assert report['matched_nodes'] == 5
    with pytest.raises(ValueError, match='continuity scales'):
        graph_refinement_quality(make(1), make(3, .1), length_scale=2)


@pytest.mark.parametrize('kwargs', [dict(continuity_scale=0), dict(continuity_scale=-1),
                                    dict(continuity_scale=[1, 2]), dict(continuity_scale=np.nan),
                                    dict(anisotropy_ratio=np.zeros((2, 2, 2))), dict(anisotropy_ratio=[2, 3])])
def test_invalid_scale_maps(kwargs):
    xyz = np.indices((2, 2, 2)).reshape(3, -1).T.astype(float)
    with pytest.raises(ValueError):
        GeologicalMetricField.from_gradients(xyz, np.tile([0, 0, 1], (len(xyz), 1)), np.ones((2, 2, 2), bool), **kwargs)
