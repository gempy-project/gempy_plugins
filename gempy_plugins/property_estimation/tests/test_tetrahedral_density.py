import numpy as np
import pytest


def test_combination_density_exact_noisy_and_unobserved_components(monkeypatch, tmp_path):
    import gempy as gp
    import gempy_engine
    from gempy_plugins.property_estimation.examples.tetrahedral_combination import make_combination_model
    from gempy_plugins.property_estimation.examples.tetrahedral_density import run_demo, as_pyvista

    model = make_combination_model()
    solution = model.solutions
    grid_values = model.grid.values.copy()

    def unexpected_solve(*args, **kwargs):
        pytest.fail('Density simulation must reuse the existing geological solve')

    monkeypatch.setattr(gp, 'compute_model', unexpected_solve)
    monkeypatch.setattr(gempy_engine, 'compute_model', unexpected_solve)
    case = run_demo(model=model, cells=240, remesh=False, realizations=16, landmarks=16, n_frequencies=128)
    assert model.solutions is solution
    np.testing.assert_array_equal(model.grid.values, grid_values)
    assert case.report['closed_volumes'] and case.report['mesh_quality']['inverted'] == 0
    assert case.report['exact_observations'] == 4
    assert case.report['noisy_observations'] == 2
    assert case.report['exact_observation_max_error'] < 1e-12
    exact = case.noise_variance == 0
    exact_nodes, noisy_nodes = case.observations[exact], case.observations[~exact]
    np.testing.assert_allclose(case.conditional.values[:, exact_nodes],
                               np.broadcast_to(case.observation_values[exact], (16, 4)), atol=1e-12)
    np.testing.assert_allclose(case.moments.mean[exact_nodes], case.observation_values[exact], atol=1e-12)
    assert np.max(case.moments.variance[exact_nodes].numpy()) < 1e-12
    assert np.all(case.moments.variance[noisy_nodes].numpy() > 0)
    assert np.all(case.moments.variance[noisy_nodes].numpy() < case.noise_variance[~exact])
    assert np.all(case.conditional.values[:, noisy_nodes].numpy().std(axis=0) > 0)
    untouched = ~np.isin(case.graph.component_ids, case.graph.component_ids[case.observations])
    assert untouched.any()
    np.testing.assert_array_equal(case.conditional.values[:, untouched], case.prior.values[:, untouched])
    np.testing.assert_allclose(case.moments.mean[untouched], 2.65)
    np.testing.assert_allclose(case.moments.variance[untouched], .10 ** 2, atol=1e-14)
    assert np.all(case.moments.variance.numpy() <= .10 ** 2 + 1e-14)
    assert np.isnan(case.cell_data['conditioned_density'][~case.graph.metric.active_mask]).all()

    pv = pytest.importorskip('pyvista')
    grid, boundary = as_pyvista(case)
    assert grid.n_cells == len(case.mesh.tetrahedra)
    assert np.all(grid.celltypes == pv.CellType.TETRA)
    assert boundary.n_cells == len(case.mesh.boundary_triangles)
    path = tmp_path / 'density.vtu'
    grid.save(path)
    restored = pv.read(path)
    cells = case.graph.node_indices[exact_nodes]
    np.testing.assert_allclose(restored.cell_data['conditioned_density'][cells], case.observation_values[exact], atol=1e-12)
    np.testing.assert_allclose(restored.cell_data['posterior_sd_density'], case.cell_data['posterior_sd_density'])
    assert restored.field_data['density_units'].tolist() == ['g/cm^3']
