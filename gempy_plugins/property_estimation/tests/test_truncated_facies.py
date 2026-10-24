import numpy as np
import pytest

torch = pytest.importorskip('torch')

from gempy_plugins.property_estimation.categorical_fields import TruncatedGaussianFacies, FaciesRealizations, facies_diagnostics
from gempy_plugins.property_estimation.gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding


def split_geometry():
    shape = (7, 1, 1)
    xyz = np.indices(shape).reshape(3, -1).T.astype(float)
    mask = np.ones(shape, bool)
    mask[2] = False
    labels = np.zeros(shape, dtype=int)
    labels[4:] = 1
    metric = GeologicalMetricField.from_gradients(
        xyz, np.tile([0, 0, 1], (len(xyz), 1)), mask, domain_labels=labels,
        anisotropy_ratio=1, metadata={'domain_keys': [[1, 0], [1, 1]]})
    graph = AnisotropicGraph.from_metric(metric, stencil_radius=3)
    return graph, GeologicalEmbedding.from_graph(graph, landmarks=8, dimensions=3)


def test_facies_marginals_variance_scaling_and_component_independence():
    _, embedding = split_geometry()
    simulator = GaussianFieldSimulator(embedding, GaussianCovariance(2, 4), n_frequencies=128)
    model = TruncatedGaussianFacies(simulator, (.3, .4, .3), names=('shale', 'limestone', 'sandstone'))
    fields = model.simulate(realizations=12000, seed=42, realization_batch_size=1000)
    codes = fields.values.numpy()
    for code, target in enumerate(model.proportions):
        np.testing.assert_allclose(np.mean(codes == code, axis=0), target, atol=.015)
    # Indicators in separate geological components remain independent.
    indicators = (codes == 0).astype(float)
    assert abs(np.cov(indicators[:, 0], indicators[:, -1])[0, 1]) < .008
    assert codes.dtype == np.int64
    assert fields.names == ('shale', 'limestone', 'sandstone')


def test_chunking_grid_mapping_and_normalized_latent_variance():
    _, embedding = split_geometry()
    first = TruncatedGaussianFacies(GaussianFieldSimulator(embedding, GaussianCovariance(2, 1)), (.3, .4, .3))
    second = TruncatedGaussianFacies(GaussianFieldSimulator(embedding, GaussianCovariance(2, 4)), (.3, .4, .3))
    fields = first.simulate(realizations=17, seed=8, point_chunk_size=1, realization_batch_size=3)
    batches = list(second.iter_simulate(realizations=17, seed=8, point_chunk_size=20, realization_batch_size=5))
    assert [batch.realization_start for batch in batches] == [0, 5, 10, 15]
    np.testing.assert_array_equal(fields.values, torch.cat([batch.values for batch in batches]))
    assert fields.to_grid()[2, 0, 0] == -1
    assert np.isnan(fields.to_property_field().values[2, 0, 0])
    np.testing.assert_array_equal(fields.to_grid().ravel()[embedding.node_indices], fields.values[0])
    with pytest.raises(IndexError):
        fields.to_grid(17)


def test_diagnostics_count_faces_without_crossing_faults_or_holes():
    graph, embedding = split_geometry()
    fields = FaciesRealizations(torch.zeros((2, len(embedding.coordinates)), dtype=torch.long),
                               embedding, ('present', 'absent'), (.5, .5))
    report = facies_diagnostics(graph, fields)
    assert report['facies'][0]['components_mean'] == 3
    assert report['facies'][0]['largest_component_fraction_when_present'] == .5
    assert report['facies'][1]['components_mean'] == 0
    assert report['facies'][1]['largest_component_fraction_when_present'] is None
    assert report['face_pairs'] == 3
    assert report['excluded_cross_component_faces'] == 1
    assert report['undirected_contact_counts_upper'] == [[6, 0], [0, 0]]
    assert report['directional_agreement']['parallel']['same_facies_probability'] == 1
    assert report['directional_agreement']['perpendicular']['same_facies_probability'] is None


@pytest.mark.parametrize('proportions', [[.3, .3], [1.1, -.1], [np.nan, .5], [[.5, .5]], []])
def test_invalid_proportions_are_rejected(proportions):
    _, embedding = split_geometry()
    with pytest.raises(ValueError, match='proportions'):
        TruncatedGaussianFacies(GaussianFieldSimulator(embedding), proportions)


def test_single_facies_and_invalid_names():
    _, embedding = split_geometry()
    simulator = GaussianFieldSimulator(embedding)
    np.testing.assert_array_equal(TruncatedGaussianFacies(simulator, [1.]).simulate().values, 0)
    with pytest.raises(ValueError, match='names'):
        TruncatedGaussianFacies(simulator, [.5, .5], names=('same', 'same'))
    with pytest.raises(ValueError, match='mean'):
        TruncatedGaussianFacies(simulator, [.5, .5]).simulate(mean=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_facies_cuda_output_matches_cpu():
    _, embedding = split_geometry()
    cpu = TruncatedGaussianFacies(GaussianFieldSimulator(embedding), [.3, .4, .3])
    gpu = TruncatedGaussianFacies(GaussianFieldSimulator(embedding, device='cuda'), [.3, .4, .3])
    fields = gpu.simulate(realizations=8, seed=7, output_device='cuda')
    assert fields.values.is_cuda
    np.testing.assert_array_equal(fields.values.cpu(), cpu.simulate(realizations=8, seed=7).values)
