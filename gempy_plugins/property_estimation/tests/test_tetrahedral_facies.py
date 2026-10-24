import numpy as np
import pytest


def test_combination_facies_honor_cells_preserve_unobserved_components_and_export(monkeypatch, tmp_path):
    import gempy as gp
    import gempy_engine
    from gempy_plugins.property_estimation.examples.tetrahedral_facies import make_combination_model, run_demo, as_pyvista

    model = make_combination_model()
    solution = model.solutions
    grid_values = model.grid.values.copy()

    def unexpected_solve(*args, **kwargs):
        pytest.fail('Meshing and centroid field queries must reuse the original solution')

    monkeypatch.setattr(gp, 'compute_model', unexpected_solve)
    monkeypatch.setattr(gempy_engine, 'compute_model', unexpected_solve)
    case = run_demo(model=model, cells=240, remesh=False, realizations=8, landmarks=16, n_frequencies=128)
    assert model.solutions is solution
    np.testing.assert_array_equal(model.grid.values, grid_values)
    assert case.mesh.unit_names == ['rock3', 'rock2', 'rock1', 'basement']
    assert case.report['closed_volumes']
    assert case.report['mesh_quality']['inverted'] == 0
    assert case.report['fault_blocks'] == 2
    assert case.report['invalid_gradient_count'] == 0
    assert case.report['gradient_half_step_relative_change_max'] < 1e-4
    assert len(case.observations) == 6
    assert case.report['honored_fraction'] == 1.
    assert case.report['untouched_component_cells'] > 0
    assert case.report['untouched_component_changed_codes'] == 0
    assert case.conditional.metadata['sampler'] == 'rejection'
    observed = np.broadcast_to(case.observed_codes, (8, 6))
    np.testing.assert_array_equal(case.conditional.values[:, case.observations], observed)
    np.testing.assert_array_equal(case.conditional.observation_codes, observed)

    cells = case.graph.node_indices
    mask = case.graph.metric.active_mask
    np.testing.assert_array_equal(mask, np.isin(case.mesh.tet_units, [1, 2]))
    a, b = cells[np.asarray(case.graph.adjacency.nonzero())]
    np.testing.assert_array_equal(case.mesh.tet_units[a], case.mesh.tet_units[b])
    np.testing.assert_array_equal(case.mesh.tet_blocks[a], case.mesh.tet_blocks[b])
    assert np.all(case.conditional.to_cells()[~mask] == -1)
    np.testing.assert_array_equal(case.conditional.to_cells()[cells], case.conditional.values[0])
    probabilities = np.stack([case.cell_data['probability_' + name] for name in case.prior.names])
    np.testing.assert_allclose(probabilities[:, mask].sum(axis=0), 1.)
    assert np.isnan(probabilities[:, ~mask]).all()
    observation_cells = cells[case.observations]
    np.testing.assert_array_equal(probabilities[:, observation_cells], np.eye(3)[case.observed_codes].T)
    unobserved = ~np.isin(case.graph.component_ids, case.graph.component_ids[case.observations])
    np.testing.assert_array_equal(case.conditional.values[:, unobserved], case.prior.values[:, unobserved])

    pv = pytest.importorskip('pyvista')
    grid, boundary = as_pyvista(case)
    assert np.all(grid.celltypes == pv.CellType.TETRA)
    assert boundary.n_cells == len(case.mesh.boundary_triangles)
    path = tmp_path / 'combined_facies.vtu'
    grid.save(path)
    restored = pv.read(path)
    np.testing.assert_array_equal(restored.cell_data['conditioned_facies'][observation_cells], case.observed_codes)
    np.testing.assert_allclose(restored.cell_data['probability_sandstone'], probabilities[2])
    assert restored.field_data['facies_names'].tolist() == ['shale', 'siltstone', 'sandstone']
