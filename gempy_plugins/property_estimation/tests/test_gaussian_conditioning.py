import numpy as np
import pytest

torch = pytest.importorskip('torch')

from gempy_plugins.property_estimation.gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding
from .test_spectral_fields import line_embedding


@pytest.mark.parametrize('noise', [[0., 0.], [0., .2], [.1, .4]])
def test_conditional_ensemble_matches_dense_gaussian_posterior(noise):
    simulator = GaussianFieldSimulator(line_embedding(), GaussianCovariance(2, 4),
                                       n_frequencies=64, dtype='float64')
    nodes, data, mean = np.array([2, 6]), np.array([4., 1.]), 3.
    prior = simulator.finite_covariance(np.arange(9)).numpy()
    cross = prior[:, nodes]
    gain = np.linalg.solve(prior[np.ix_(nodes, nodes)] + np.diag(noise), cross.T).T
    expected_mean = mean + gain @ (data - mean)
    expected_covariance = prior - gain @ cross.T
    conditioned = simulator.condition(nodes, data, mean=mean, noise_variance=noise)
    moments = conditioned.moments(point_chunk_size=2)
    np.testing.assert_allclose(moments.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(moments.variance, expected_covariance.diagonal(), atol=1e-12)
    draws = conditioned.simulate(realizations=8000, seed=32, realization_batch_size=1000)
    samples = draws.values.numpy()
    variance = expected_covariance.diagonal().clip(0)
    assert np.all(np.abs(samples.mean(axis=0) - expected_mean) < 6 * np.sqrt(variance / len(samples)) + 1e-10)
    covariance_se = np.sqrt((variance[:, None] * variance[None, :] + expected_covariance ** 2) / (len(samples) - 1))
    assert np.all(np.abs(np.cov(samples.T) - expected_covariance) < 6 * covariance_se + 1e-10)
    np.testing.assert_allclose(draws.observation_values, samples[:, nodes], atol=1e-12)
    exact = np.asarray(noise) == 0
    np.testing.assert_allclose(samples[:, nodes[exact]], np.broadcast_to(data[exact], (len(samples), exact.sum())), atol=1e-11)
    if not exact.all():
        assert np.all(np.var(draws.observation_values.numpy()[:, ~exact], axis=0) > .02)


def test_matched_draw_update_and_fault_isolation_chunking_and_caching():
    geometry = line_embedding(split=True)
    simulator = GaussianFieldSimulator(geometry, GaussianCovariance(2), n_frequencies=32)
    condition = simulator.condition([1], [2.], mean=.5)
    prior = simulator.simulate(realizations=17, seed=12, mean=.5, point_chunk_size=2, realization_batch_size=3)
    draws = condition.simulate(realizations=17, seed=12, point_chunk_size=2, realization_batch_size=3)
    covariance = simulator.finite_covariance(np.arange(8)).numpy()
    expected = prior.values.numpy() + (2 - prior.values[:, 1].numpy())[:, None] * covariance[:, 1] / covariance[1, 1]
    np.testing.assert_allclose(draws.values, expected, atol=2e-6)
    np.testing.assert_array_equal(draws.values[:, 4:], prior.values[:, 4:])
    moments = condition.moments()
    np.testing.assert_allclose(moments.mean[4:], .5)
    np.testing.assert_allclose(moments.variance[4:], 1, atol=2e-7)
    assert moments.variance[1] < 1e-12
    field = moments.to_property_field()
    assert np.isnan(field.values[4, 0, 0])
    assert np.isnan(field.variance[4, 0, 0])
    simulator.cache_basis(max_bytes=simulator.basis_cache_bytes, point_chunk_size=3)
    batches = list(condition.iter_simulate(realizations=17, seed=12, point_chunk_size=1, realization_batch_size=5))
    assert [batch.realization_start for batch in batches] == [0, 5, 10, 15]
    np.testing.assert_allclose(torch.cat([batch.values for batch in batches]), draws.values, atol=2e-6)
    np.testing.assert_allclose(torch.cat([batch.observation_values for batch in batches]), 2, atol=1e-6)
    assert draws.metadata['basis_fingerprint'] == simulator.basis_fingerprint


def test_redundant_conflicting_and_rank_limited_observations():
    simulator = GaussianFieldSimulator(line_embedding(), n_frequencies=16, dtype='float64')
    single = simulator.condition([3], [2.]).simulate(realizations=4, seed=7)
    duplicate = simulator.condition([3, 3], [2., 2.])
    assert duplicate.diagnostics[0]['retained_rank'] == 1
    np.testing.assert_allclose(duplicate.simulate(realizations=4, seed=7).values, single.values, atol=1e-12)
    with pytest.raises(ValueError, match='inconsistent'):
        simulator.condition([3, 3], [2., 3.])
    # Independent repeated noisy measurements combine through their precision.
    repeated = simulator.condition([3, 3], [2., 4.], noise_variance=.2).moments()
    combined = simulator.condition([3], [3.], noise_variance=.1).moments()
    np.testing.assert_allclose(repeated.mean, combined.mean, atol=1e-12)
    np.testing.assert_allclose(repeated.variance, combined.variance, atol=1e-12)
    small = GaussianFieldSimulator(line_embedding(), n_frequencies=1, dtype='float64')
    values = small.simulate(seed=77).values[0].numpy()
    constrained = small.condition(np.arange(9), values)
    assert constrained.diagnostics[0]['retained_rank'] == 2
    np.testing.assert_allclose(constrained.simulate(realizations=5).values, np.broadcast_to(values, (5, 9)), atol=1e-12)
    assert np.max(constrained.moments().variance.numpy()) < 1e-12
    values[0] += .1
    with pytest.raises(ValueError, match='inconsistent'):
        small.condition(np.arange(9), values)


def test_noisy_streaming_is_reproducible():
    simulator = GaussianFieldSimulator(line_embedding(split=True), n_frequencies=64, dtype='float64')
    condition = simulator.condition([1, 6], [1., -1.], noise_variance=[.2, .5])
    first = condition.simulate(realizations=11, seed=12, point_chunk_size=1, realization_batch_size=2)
    second = condition.simulate(realizations=11, seed=12, point_chunk_size=10, realization_batch_size=11)
    np.testing.assert_allclose(first.values, second.values, atol=1e-12)
    np.testing.assert_allclose(first.observation_values, second.observation_values, atol=1e-12)


@pytest.mark.parametrize('geometry', ['folded', 'faulted'])
def test_actual_gempy_conditioning_demo(geometry):
    from gempy_plugins.property_estimation.examples.conditioned_geological_simulation import run_demo
    case, = run_demo([geometry], realizations=8)
    assert case.report['exact_observation_max_error'] < 1e-10
    assert case.report['noisy_observation_latent_std'] > 0
    assert case.report['quality']['passes_provisional_targets']
    assert np.all(case.moments.variance.numpy() <= .01 + 1e-12)
    if geometry == 'faulted':
        assert case.report['untouched_component_nodes'] > 0
        assert case.report['untouched_component_max_change'] == 0


def query_geometry():
    shape = (6, 3, 4)
    xyz = np.indices(shape).reshape(3, -1).T.astype(float)
    mask = np.ones(shape, bool)
    mask[1, 1, 1] = False
    labels = np.zeros(shape, dtype=int)
    labels[3:] = 1
    metric = GeologicalMetricField.from_gradients(xyz, np.tile([-.3, 0, 1], (len(xyz), 1)), mask,
                                                 domain_labels=labels, anisotropy_ratio=5)
    graph = AnisotropicGraph.from_metric(metric)
    return graph, GeologicalEmbedding.from_constant_metric(graph)


def test_physical_queries_preserve_coordinates_and_reject_barriers():
    graph, embedding = query_geometry()
    nodes = embedding.query(graph, graph.positions)
    np.testing.assert_array_equal(nodes.coordinates, embedding.coordinates)
    np.testing.assert_array_equal(nodes.component_ids, embedding.component_ids)
    positions = np.array([[4.2, 1.3, 2.4], [3.6, .2, .3]])
    points = embedding.query(graph, positions)
    normal = graph.metric.normals[graph.node_indices[0]]
    delta = positions - graph.positions.mean(axis=0)
    expected = delta / 5 + .8 * (delta @ normal)[:, None] * normal
    np.testing.assert_allclose(points.coordinates, expected, atol=1e-7)
    np.testing.assert_array_equal(points.positions, positions)
    simulator = GaussianFieldSimulator(embedding, n_frequencies=64, dtype='float64')
    result = simulator.condition(points, [1.2, -.3]).simulate(realizations=5)
    np.testing.assert_allclose(result.observation_values, np.tile([1.2, -.3], (5, 1)), atol=1e-12)
    assert np.min(np.abs(result.values[0].numpy() - 1.2)) > 1e-4  # No voxel was snapped to the off-grid datum.
    with pytest.raises(ValueError, match='inactive'):
        embedding.query(graph, [[.5, .5, .5]])
    with pytest.raises(ValueError, match='components'):
        embedding.query(graph, [[2.5, 1.5, 2.5]])
    with pytest.raises(ValueError, match='bounds'):
        embedding.query(graph, [[-.01, 0, 0]])
    changed = GeologicalEmbedding.from_graph(graph, landmarks=8, dimensions=3)
    with pytest.raises(ValueError, match='different embedding'):
        GaussianFieldSimulator(changed).condition(points, [1.2, -.3])


@pytest.mark.parametrize('nodes, values, kwargs', [
    ([0], [np.nan], {}), ([0, 1], [1.], {}), ([-1], [0.], {}), ([.5], [0.], {}),
    ([0], [1.], {'noise_variance': -.1}), ([0], [1.], {'noise_variance': [0., 1.]}),
    ([0], [1.], {'noise_variance': np.inf}), ([0], [1.], {'mean': np.nan}),
    ([0], [1.], {'rank_tolerance': 0}), ([], [], {}),
])
def test_invalid_conditioning_inputs(nodes, values, kwargs):
    with pytest.raises(ValueError):
        GaussianFieldSimulator(line_embedding()).condition(nodes, values, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_conditional_cpu_cuda_parity():
    embedding = line_embedding(split=True)
    cpu = GaussianFieldSimulator(embedding, n_frequencies=64, dtype='float64')
    gpu = GaussianFieldSimulator(embedding, n_frequencies=64, dtype='float64', device='cuda')
    kwargs = dict(observations=[1, 6], values=[1., -1.], noise_variance=[0., .2])
    cpu, gpu = cpu.condition(**kwargs), gpu.condition(**kwargs)
    actual = gpu.simulate(realizations=8, seed=7, output_device='cuda')
    assert actual.values.is_cuda
    assert actual.observation_values.is_cuda
    np.testing.assert_allclose(actual.values.cpu(), cpu.simulate(realizations=8, seed=7).values, atol=1e-9)
    np.testing.assert_allclose(gpu.moments(output_device='cuda').variance.cpu(), cpu.moments().variance, atol=1e-9)
