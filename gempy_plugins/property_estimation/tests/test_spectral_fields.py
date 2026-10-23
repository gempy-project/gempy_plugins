import numpy as np
import pytest

torch = pytest.importorskip('torch')

from gempy_plugins.property_estimation.gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding


def line_embedding(split=False):
    shape = (9, 1, 1)
    xyz = np.indices(shape).reshape(3, -1).T.astype(float)
    mask = np.ones(shape, bool)
    if split:
        mask[4] = False
    metric = GeologicalMetricField.from_gradients(xyz, np.tile([0, 0, 1], (9, 1)), mask, anisotropy_ratio=1,
                                                 metadata={'domain_keys': [[1, 0]]})
    return GeologicalEmbedding.from_graph(AnisotropicGraph.from_metric(metric), landmarks=8, dimensions=3)


def test_ensemble_matches_finite_feature_covariance_and_gaussian_marginals():
    geometry = line_embedding()
    simulator = GaussianFieldSimulator(geometry, GaussianCovariance(2, 4), n_frequencies=1024, dtype='float64')
    draws = simulator.simulate(realizations=12000, seed=12, mean=3, realization_batch_size=1000).values.numpy()
    actual = simulator.finite_covariance(np.arange(9)).numpy()
    np.testing.assert_allclose(draws.mean(axis=0), 3, atol=.06)
    np.testing.assert_allclose(np.cov(draws.T), actual, atol=.13)
    centered = draws - 3
    np.testing.assert_allclose(np.mean(centered ** 4, axis=0), 3 * 4 ** 2, rtol=.1)
    distance = np.abs(np.arange(9)[:, None] - np.arange(9)[None, :])
    np.testing.assert_allclose(actual, simulator.covariance(distance), atol=.18)
    np.testing.assert_allclose(actual.diagonal(), 4, atol=1e-12)


def test_chunking_caching_and_independent_components():
    geometry = line_embedding(split=True)
    simulator = GaussianFieldSimulator(geometry, n_frequencies=64)
    first = simulator.simulate(realizations=17, seed=4, point_chunk_size=1, realization_batch_size=3)
    second = simulator.simulate(realizations=17, seed=4, point_chunk_size=20, realization_batch_size=17)
    np.testing.assert_allclose(first.values, second.values, atol=1e-6)
    with pytest.raises(ValueError, match='exceeding'):
        simulator.cache_basis(max_bytes=1)
    simulator.cache_basis(max_bytes=simulator.basis_cache_bytes, point_chunk_size=3)
    cached = simulator.simulate(realizations=17, seed=4)
    np.testing.assert_allclose(first.values, cached.values, atol=1e-6)
    actual = simulator.finite_covariance(np.arange(8)).numpy()
    assert np.all(actual[:4, 4:] == 0)
    independent = simulator.simulate(realizations=6000, seed=7, realization_batch_size=1000).values.numpy()
    assert np.max(np.abs(np.cov(independent.T)[:4, 4:])) < .06
    field = first.to_property_field(0)
    assert field.domain_keys == [(1, 0)]
    assert np.isnan(field.values[4, 0, 0])
    np.testing.assert_allclose(field.values.ravel()[geometry.node_indices], first.values[0])
    batches = list(simulator.iter_simulate(realizations=17, seed=4, realization_batch_size=5))
    assert [batch.realization_start for batch in batches] == [0, 5, 10, 15]
    np.testing.assert_allclose(torch.cat([batch.values for batch in batches]), first.values, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cpu_cuda_parity_with_identical_frequencies_and_coefficients():
    geometry = line_embedding()
    cpu = GaussianFieldSimulator(geometry, basis_seed=123, device='cpu')
    gpu = GaussianFieldSimulator(geometry, basis_seed=123, device='cuda')
    np.testing.assert_allclose(cpu.simulate(realizations=8, seed=77).values,
                               gpu.simulate(realizations=8, seed=77).values, atol=2e-5)
