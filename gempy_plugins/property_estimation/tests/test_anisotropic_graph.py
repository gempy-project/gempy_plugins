import numpy as np
import pytest

from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField, AnisotropicGraph


@pytest.mark.parametrize('normal', [[0, 0, 1], [1, 2, 3]])
def test_constant_metric_graph_bounds_analytical_distance(normal):
    shape = (7, 7, 7)
    xyz = np.indices(shape).reshape(3, -1).T.astype(float)
    metric = GeologicalMetricField.from_gradients(xyz, np.tile(normal, (len(xyz), 1)), np.ones(shape, bool), anisotropy_ratio=4)
    graph = AnisotropicGraph.from_metric(metric, dtype=np.float64)
    source = np.ravel_multi_index((3, 3, 3), shape)
    delta = xyz - xyz[source]
    analytical = np.sqrt(np.einsum('ni,ij,nj->n', delta, metric.tensors([source])[0], delta))
    distance = graph.shortest_paths([source])[0]
    assert np.all(distance >= analytical - 1e-12)
    # Along every one of the 26 available directions, one step is exact.
    neighbors = graph.adjacency[source].indices
    np.testing.assert_allclose(distance[neighbors], analytical[neighbors])
    assert (graph.adjacency != graph.adjacency.T).nnz == 0
    assert graph.adjacency.data.min() > 0


def test_domain_boundary_and_diagonal_corner_cutting():
    shape = (3, 3, 1)
    xyz = np.indices(shape).reshape(3, -1).T.astype(float)
    mask = np.zeros(shape, bool)
    mask[0, 0, 0] = mask[1, 1, 0] = True
    metric = GeologicalMetricField.from_gradients(xyz, np.tile([0, 0, 1], (9, 1)), mask)
    graph = AnisotropicGraph.from_metric(metric)
    assert len(np.unique(graph.component_ids)) == 2
    assert np.isinf(graph.shortest_paths([0])[0, 1])
    labels = np.zeros(shape, dtype=int)
    labels[1:] = 1
    metric = GeologicalMetricField.from_gradients(xyz, np.tile([0, 0, 1], (9, 1)), np.ones(shape, bool), domain_labels=labels)
    graph = AnisotropicGraph.from_metric(metric)
    rows, cols = graph.adjacency.nonzero()
    assert np.all(labels.ravel()[rows] == labels.ravel()[cols])
    assert len(np.unique(graph.component_ids)) == 2


def test_wider_stencil_improves_directional_error_without_bridging_barriers():
    shape = (9, 5, 5)
    xyz = np.indices(shape).reshape(3, -1).T.astype(float)
    normals = np.tile([-.5, 0, 1], (len(xyz), 1))
    metric = GeologicalMetricField.from_gradients(xyz, normals, np.ones(shape, bool), anisotropy_ratio=5)
    narrow = AnisotropicGraph.from_metric(metric)
    wide = AnisotropicGraph.from_metric(metric, stencil_radius=2)
    target = np.ravel_multi_index((8, 0, 4), shape)
    delta = xyz[target] - xyz[0]
    exact = np.sqrt(delta @ metric.tensors([0])[0] @ delta)
    assert wide.shortest_paths([0])[0, target] < narrow.shortest_paths([0])[0, target]
    np.testing.assert_allclose(wide.shortest_paths([0])[0, target], exact, rtol=1e-6)
    mask = np.ones(shape, bool)
    mask[4] = False
    split = AnisotropicGraph.from_metric(GeologicalMetricField.from_gradients(xyz, normals, mask), stencil_radius=3)
    assert len(np.unique(split.component_ids)) == 2
