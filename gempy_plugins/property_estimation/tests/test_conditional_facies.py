import numpy as np
import pytest
from scipy.stats import truncnorm

torch = pytest.importorskip('torch')

from gempy_plugins.property_estimation.categorical_fields import TruncatedGaussianFacies
from gempy_plugins.property_estimation.categorical_fields.conditioning import _truncated_normal
from gempy_plugins.property_estimation.gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from .test_spectral_fields import line_embedding
from .test_gaussian_conditioning import query_geometry


@pytest.mark.parametrize('sampler', ['rejection', 'gibbs'])
def test_single_category_condition_matches_analytic_bivariate_probability(sampler):
    simulator = GaussianFieldSimulator(line_embedding(), GaussianCovariance(2, 4), n_frequencies=64, dtype='float64')
    facies = TruncatedGaussianFacies(simulator, [.5, .5])
    result = facies.condition([2], [1], sampler=sampler, burn_in=4).simulate(
        realizations=8000, seed=77, realization_batch_size=1000)
    np.testing.assert_array_equal(result.values[:, 2], 1)
    np.testing.assert_array_equal(result.observation_codes, 1)
    observed = result.latent_observation_values.numpy()[:, 0]
    assert abs(observed.mean() - 2 * np.sqrt(2 / np.pi)) < .04
    assert abs(observed.var() - 4 * (1 - 2 / np.pi)) < .08
    # P(Zx > 0 | Zo > 0) = 1/2 + asin(correlation)/pi.
    covariance = simulator.finite_covariance(np.arange(9)).numpy()
    expected = .5 + np.arcsin(np.clip(covariance[:, 2] / 4, -1, 1)) / np.pi
    np.testing.assert_allclose(result.probabilities()[1], expected, atol=.025)


def test_joint_gibbs_intervals_match_independent_rejection_reference():
    simulator = GaussianFieldSimulator(line_embedding(), GaussianCovariance(2), n_frequencies=128, dtype='float64')
    facies = TruncatedGaussianFacies(simulator, [.3, .4, .3])
    arguments = dict(realizations=5000, realization_batch_size=1000)
    reference = facies.condition([2, 4], [0, 2]).simulate(seed=23, **arguments)
    gibbs = facies.condition([2, 4], [0, 2], sampler='gibbs', burn_in=128).simulate(seed=24, **arguments)
    np.testing.assert_allclose(gibbs.probabilities(), reference.probabilities(), atol=.035)
    np.testing.assert_allclose(gibbs.latent_observation_values.mean(dim=0),
                               reference.latent_observation_values.mean(dim=0), atol=.04)
    np.testing.assert_allclose(np.cov(gibbs.latent_observation_values.numpy().T),
                               np.cov(reference.latent_observation_values.numpy().T), atol=.035)
    assert gibbs.metadata['independent_chains']
    assert gibbs.metadata['burn_in'] == 128


@pytest.mark.parametrize('sampler', ['rejection', 'gibbs'])
def test_fault_isolation_duplicates_and_batch_reproducibility(sampler):
    simulator = GaussianFieldSimulator(line_embedding(split=True), n_frequencies=64, dtype='float64')
    facies = TruncatedGaussianFacies(simulator, [.3, .4, .3])
    model = facies.condition([0, 0, 2], [0, 0, 2], sampler=sampler, burn_in=32)
    first = model.simulate(realizations=17, seed=5, realization_batch_size=3, point_chunk_size=1)
    batches = list(model.iter_simulate(realizations=17, seed=5, realization_batch_size=5, point_chunk_size=20))
    np.testing.assert_array_equal(torch.cat([batch.values for batch in batches]), first.values)
    np.testing.assert_allclose(torch.cat([batch.latent_observation_values for batch in batches]),
                               first.latent_observation_values, atol=1e-12)
    np.testing.assert_array_equal(first.observation_codes, np.tile([0, 0, 2], (17, 1)))
    prior = facies.simulate(realizations=17, seed=5, realization_batch_size=3, point_chunk_size=1)
    np.testing.assert_array_equal(first.values[:, 4:], prior.values[:, 4:])
    assert first.metadata['unique_observations'] == 2
    assert first.to_grid()[4, 0, 0] == -1
    with pytest.raises(ValueError, match='Conflicting'):
        facies.condition([0, 0], [0, 1])


def test_off_grid_facies_are_honored_and_latent_variance_scaling_is_invariant():
    graph, embedding = query_geometry()
    observations = embedding.query(graph, [[3.2, .3, .4], [4.5, 1.3, 2.4]])
    results = []
    for variance in (1., 4.):
        simulator = GaussianFieldSimulator(embedding, GaussianCovariance(.5, variance), n_frequencies=64, dtype='float64')
        fields = TruncatedGaussianFacies(simulator, [.3, .4, .3]).condition(observations, [0, 2]).simulate(realizations=12, seed=9)
        results.append(fields)
        np.testing.assert_array_equal(fields.observation_codes, np.tile([0, 2], (12, 1)))
    np.testing.assert_array_equal(results[0].values, results[1].values)
    np.testing.assert_allclose(results[0].latent_observation_values * 2, results[1].latent_observation_values)


def test_actual_gempy_local_continuity_and_conditional_facies():
    from gempy_plugins.property_estimation.examples.local_continuity_facies import run_demo
    cases = run_demo(('range_map', 'ratio_map', 'faulted_range'), realizations=32)
    for case in cases:
        assert case.report['honored_fraction'] == 1
        np.testing.assert_array_equal(case.conditional.values[:, case.observations], np.tile(case.codes, (32, 1)))
        regional = case.report['regional_prior_continuity']
        assert regional['right']['finite_feature_correlation'] > regional['left']['finite_feature_correlation'] + .08
        assert regional['right']['prior_same_facies_probability'] > regional['left']['prior_same_facies_probability'] + .08
        assert regional['right']['physical_lag'] == regional['left']['physical_lag'] == 8
        assert case.report['untouched_component_changed_codes'] == 0
    assert cases[-1].report['untouched_component_nodes'] == 862


def test_rank_limited_basis_and_exhausted_rejection_fail_explicitly():
    simulator = GaussianFieldSimulator(line_embedding(), n_frequencies=1, dtype='float64')
    facies = TruncatedGaussianFacies(simulator, [.5, .5])
    codes = facies.simulate(seed=4).values[0].numpy()
    result = facies.condition(np.arange(9), codes).simulate(realizations=5)
    np.testing.assert_array_equal(result.values, np.tile(codes, (5, 1)))
    with pytest.raises(ValueError, match='full-rank'):
        facies.condition(np.arange(9), codes, sampler='gibbs')
    rare = TruncatedGaussianFacies(simulator, [1 - 1e-12, 1e-12]).condition([0], [1], max_proposals=1)
    with pytest.raises(RuntimeError, match='exhausted'):
        rare.simulate()


@pytest.mark.parametrize('lower, upper', [(-np.inf, -.5), (.5, np.inf), (9., 9.1), (-12., -11.), (-1., 2.)])
def test_truncated_inverse_handles_extreme_tails(lower, upper):
    u = np.linspace(.001, .999, 100)
    actual = _truncated_normal(u, lower, upper, 0., 1.)
    np.testing.assert_allclose(actual, truncnorm.ppf(u, lower, upper), atol=1e-12)
    assert np.all((actual >= lower) & (actual < upper))


@pytest.mark.parametrize('codes, kwargs', [([3], {}), ([-1], {}), ([.5], {}), ([0, 1], {}),
                                         ([0], {'sampler': 'midpoint'}), ([0], {'burn_in': 0}),
                                         ([0], {'max_proposals': 0})])
def test_invalid_facies_observations(codes, kwargs):
    with pytest.raises(ValueError):
        TruncatedGaussianFacies(GaussianFieldSimulator(line_embedding()), [.3, .4, .3]).condition([0], codes, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_conditional_facies_cuda_parity():
    embedding = line_embedding(split=True)
    results = []
    for device in ('cpu', 'cuda'):
        simulator = GaussianFieldSimulator(embedding, n_frequencies=64, dtype='float64', device=device)
        fields = TruncatedGaussianFacies(simulator, [.3, .4, .3]).condition([1], [2]).simulate(
            realizations=12, seed=7, output_device=device)
        assert fields.values.device.type == device
        results.append(fields.values.cpu())
    np.testing.assert_array_equal(*results)
