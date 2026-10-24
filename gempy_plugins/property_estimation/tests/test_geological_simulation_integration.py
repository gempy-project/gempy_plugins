import numpy as np
import pytest
import gempy as gp

from gempy_plugins.property_estimation.examples.geological_models import make_geological_model
from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField, GeologicalEmbedding
from gempy_plugins.property_estimation.geological_coordinates import AnisotropicGraph
from gempy_plugins.property_estimation.geological_coordinates.diagnostics import embedding_quality


@pytest.mark.parametrize('geometry', ['horizontal', 'dipping', 'folded', 'syncline', 'gentle_fold', 'tight_fold', 'thin_layer'])
def test_gempy_voxel_gradients_match_world_coordinate_derivatives(geometry):
    model = make_geological_model(geometry, resolution=(10, 5, 10))
    metric = GeologicalMetricField.from_gempy(model, 'lower')
    indices = np.flatnonzero(metric.active_mask)[::7]
    points = metric.positions[indices]
    evaluator = gp.prepare_field_evaluator(model)
    finite_difference = []
    for axis in np.eye(3):
        finite_difference.append((evaluator(points + .001 * axis)[0] - evaluator(points - .001 * axis)[0]) / .002)
    finite_difference = np.column_stack(finite_difference)
    finite_difference /= np.linalg.norm(finite_difference, axis=1, keepdims=True)
    # Compare unoriented normals; the anisotropic tensor is sign-invariant.
    agreement = np.abs(np.einsum('ij,ij->i', finite_difference, metric.normals[indices]))
    np.testing.assert_allclose(agreement, 1., atol=1e-6)
    if geometry == 'folded':
        assert np.ptp(metric.normals[indices, 0]) > .2
    embedding = GeologicalEmbedding.from_gempy(model, 'lower', landmarks=16, dimensions=6)
    assert np.isfinite(embedding.coordinates).all()
    assert embedding.metadata['domain_keys']


def test_missing_gradients_and_basement_orientation_are_explicit():
    model = make_geological_model('horizontal', resolution=(6, 3, 6), compute_gradients=False)
    with pytest.raises(ValueError, match='compute_scalar_gradient'):
        GeologicalMetricField.from_gempy(model, 'lower')
    with pytest.raises(ValueError, match='structural_group'):
        GeologicalMetricField.from_gempy(model, 'basement')


def test_folded_embedding_meets_local_covariance_gate():
    model = make_geological_model('folded')
    metric = GeologicalMetricField.from_gempy(model, 'lower', anisotropy_ratio=5)
    graph = AnisotropicGraph.from_metric(metric, stencil_radius=3)
    embedding = GeologicalEmbedding.from_graph(graph, landmarks=32, dimensions=6)
    quality = embedding_quality(graph, embedding, length_scale=5)
    assert quality['held_out_sources']
    assert quality['passes_provisional_targets'], quality
    assert quality['covariance_absolute_p90'] < quality['cartesian_covariance_absolute_p90']
    from gempy_plugins.property_estimation.gaussian_fields import GaussianCovariance, GaussianFieldSimulator
    from gempy_plugins.property_estimation.geological_coordinates.diagnostics import directional_variograms
    simulator = GaussianFieldSimulator(embedding, GaussianCovariance(length_scale=5), n_frequencies=512)
    fields = simulator.simulate(realizations=64, seed=13)
    variograms = directional_variograms(graph, fields.values.numpy(), max_separation=5)
    assert variograms['parallel']['semivariance'] < variograms['perpendicular']['semivariance'] / 2
    assert fields.to_property_field().values.shape == tuple(model.grid.regular_grid.resolution)


def test_faulted_gempy_domains_are_independent():
    model = make_geological_model('faulted', resolution=(12, 5, 12))
    metric = GeologicalMetricField.from_gempy(model, 'lower')
    assert len(metric.metadata['domain_keys']) == 2
    graph = AnisotropicGraph.from_metric(metric, stencil_radius=2)
    start, end = graph.adjacency.nonzero()
    labels = metric.domain_labels[graph.node_indices]
    assert np.all(labels[start] == labels[end])
    assert len(np.unique(graph.component_ids)) >= 2


def test_nonuniform_input_scaling_uses_gradient_chain_rule():
    model = make_geological_model('dipping', resolution=(8, 4, 8))
    model.input_transform.scale *= np.array([1.5, .7, 1.])
    gp.compute_model(model, engine_config=gp.data.GemPyEngineConfig(backend=gp.data.AvailableBackends.numpy, dtype='float64'))
    metric = GeologicalMetricField.from_gempy(model, 'lower')
    indices = np.flatnonzero(metric.active_mask)[::5]
    evaluator = gp.prepare_field_evaluator(model)
    points = metric.positions[indices]
    gradient = np.column_stack([(evaluator(points + .001 * axis)[0] - evaluator(points - .001 * axis)[0]) / .002
                                for axis in np.eye(3)])
    gradient /= np.linalg.norm(gradient, axis=1, keepdims=True)
    np.testing.assert_allclose(metric.normals[indices], gradient, atol=1e-6)
