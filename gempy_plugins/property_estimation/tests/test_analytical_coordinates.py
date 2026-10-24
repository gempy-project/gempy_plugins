import numpy as np
import pytest
from scipy.spatial.distance import cdist

from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding
from gempy_plugins.property_estimation.geological_coordinates.analytical import bedding_orientation, constant_metric_coordinates
from gempy_plugins.property_estimation.geological_coordinates.diagnostics import constant_metric_quality, embedding_quality, graph_refinement_quality


@pytest.mark.parametrize('normal', [[0., 0., 1.], [-.3, 0., 1.]])
def test_analytical_reference_recovers_constant_metric_and_cache(normal, tmp_path):
    xyz = np.indices((9, 4, 7)).reshape(3, -1).T.astype(float)
    gradients = np.tile(normal, (len(xyz), 1))
    gradients[::2] *= -1  # Bedding normals are unoriented, including the comparator.
    metric = GeologicalMetricField.from_gradients(xyz, gradients, np.ones((9, 4, 7), bool), anisotropy_ratio=5)
    graph = AnisotropicGraph.from_metric(metric)
    analytical = GeologicalEmbedding.from_constant_metric(graph)
    delta = xyz[:, None] - xyz[None, :]
    exact = np.sqrt(np.einsum('...i,ij,...j->...', delta, metric.tensors([0])[0], delta))
    np.testing.assert_allclose(cdist(analytical.coordinates, analytical.coordinates), exact, atol=5e-7)
    quality = constant_metric_quality(graph, analytical, length_scale=2)
    assert quality['embedding_vs_analytical']['covariance_absolute_p90'] < 1e-6
    assert quality['graph_vs_analytical']['covariance_absolute_p90'] > .02
    comparator = embedding_quality(graph, analytical, length_scale=2)
    np.testing.assert_allclose(abs(np.dot(comparator['cartesian_normal'], np.asarray(normal) / np.linalg.norm(normal))), 1)
    path = tmp_path / 'analytical.npz'
    analytical.save(path)
    restored = GeologicalEmbedding.load(path, expected_graph_fingerprint=graph.fingerprint)
    np.testing.assert_array_equal(restored.coordinates, analytical.coordinates)


def test_varying_metrics_cannot_be_mislabelled_as_analytical():
    xyz = np.indices((3, 2, 1)).reshape(3, -1).T.astype(float)
    gradients = np.column_stack((xyz[:, 0], np.zeros(len(xyz)), np.ones(len(xyz))))
    metric = GeologicalMetricField.from_gradients(xyz, gradients, np.ones((3, 2, 1), bool), anisotropy_ratio=5)
    assert bedding_orientation(metric)['angular_deviation_max_degrees'] > 20
    with pytest.raises(ValueError, match='constant metric'):
        constant_metric_coordinates(metric)
    metric.anisotropy_ratio = 1  # Orientation does not affect an isotropic tensor.
    np.testing.assert_allclose(cdist(constant_metric_coordinates(metric), constant_metric_coordinates(metric)), cdist(xyz, xyz))


def test_refinement_preserves_endpoints_and_reports_lost_connectivity():
    def line_graph(factor, barrier=False):
        shape = (7 * factor, 1, 1)
        xyz = np.indices(shape).reshape(3, -1).T.astype(float)
        xyz[:, 0] = (xyz[:, 0] + .5) / factor
        mask = np.ones(shape, bool)
        if barrier:
            mask[9] = False  # Between coarse centers; all seven centers remain active.
        metric = GeologicalMetricField.from_gradients(xyz, np.tile([0, 0, 1], (len(xyz), 1)), mask, anisotropy_ratio=1)
        return AnisotropicGraph.from_metric(metric)
    coarse = line_graph(1)
    fine = line_graph(3)
    quality = graph_refinement_quality(coarse, fine, length_scale=5)
    assert quality['matched_nodes'] == 7
    assert quality['unmatched_active_nodes'] == 0
    assert quality['distance_relative_p90'] < 1e-6
    assert quality['reachability_lost_pairs'] == 0
    split = graph_refinement_quality(coarse, line_graph(3, barrier=True), length_scale=5)
    assert split['matched_nodes'] == 7
    assert split['reachability_lost_pairs'] > 0
    with pytest.raises(ValueError, match='odd-integer'):
        graph_refinement_quality(coarse, line_graph(2), length_scale=5)
    fine.metric.positions[:, 0] += .01
    with pytest.raises(ValueError, match='coincide'):
        graph_refinement_quality(coarse, fine, length_scale=5)
