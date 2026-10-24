import numpy as np
import pytest

from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField


def test_metric_eigenstructure_sign_invariance_and_invalid_gradients():
    xyz = np.indices((3, 2, 2)).reshape(3, -1).T.astype(float)
    gradients = np.tile([1., 2., 3.], (len(xyz), 1))
    gradients[0] = 0
    gradients[1] = np.nan
    metric = GeologicalMetricField.from_gradients(xyz, gradients, np.ones((3, 2, 2), bool), anisotropy_ratio=10)
    assert metric.invalid_gradient_count == 2
    assert not metric.active_mask[:2].any()
    np.testing.assert_allclose(np.linalg.eigvalsh(metric.tensors()), np.tile([.01, .01, 1], (10, 1)))
    flipped = GeologicalMetricField.from_gradients(xyz, -gradients, np.ones((3, 2, 2), bool), anisotropy_ratio=10)
    np.testing.assert_allclose(metric.tensors(), flipped.tensors())
    np.testing.assert_allclose(metric.edge_costs(np.array([2]), np.array([5])),
                               flipped.edge_costs(np.array([2]), np.array([5])))
    with pytest.raises(ValueError, match='inactive'):
        metric.tensors([0])


def test_endpoint_average_cost_matches_tensor_quadratic_form():
    xyz = np.array([[0, 0, 0], [2, 3, 4]])
    metric = GeologicalMetricField.from_gradients(xyz, [[0, 0, 1], [1, 0, 1]], np.ones((2, 1, 1), bool))
    delta = xyz[1] - xyz[0]
    expected = np.sqrt(delta @ metric.tensors().mean(axis=0) @ delta)
    np.testing.assert_allclose(metric.edge_costs(np.array([0]), np.array([1])), expected)
