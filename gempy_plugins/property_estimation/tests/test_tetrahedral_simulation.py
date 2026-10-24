from types import SimpleNamespace

import numpy as np
import pytest

from gempy_plugins.property_estimation.geological_coordinates import AnisotropicGraph, GeologicalEmbedding
from gempy_plugins.property_estimation.geological_coordinates.tetrahedral import evaluate_tetrahedral_fields
from gempy_plugins.volume_mesh.background_grid import kuhn_grid


def _mesh(shape=(4, 2, 2)):
    nodes, tets = kuhn_grid((0, 1, 0, 1, 0, 1), shape)
    centers = nodes[tets].mean(axis=1)
    return SimpleNamespace(nodes=nodes, tetrahedra=tets,
                           tet_units=(centers[:, 2] > .5).astype(int),
                           tet_blocks=(centers[:, 0] > .5).astype(int),
                           units=[(0, 0), (1, 0)], block_sides=np.array([[-1], [1]]))


def test_centroid_gradients_use_world_offsets_per_cell_stack_and_fixed_fault_sides():
    mesh = _mesh()
    # Nonuniform scaling, rotation and translation. The evaluator receives WORLD
    # points, as GemPy's public wrapper does. Gradients obey the same chain rule.
    linear = np.array([[0., 2., 0.], [-3., 0., 0.], [0., 0., .5]])
    calls = []

    def evaluator(points, stacks, fault_sides):
        calls.append((points.copy(), fault_sides.copy()))
        q = points @ linear.T + [4., 5., 6.]
        x, y, z = q.T
        # A side-dependent slope tests derivatives of the actual selected branch,
        # rather than just a constant jump that would cancel in differences.
        return np.array([(stacks[0] + 1) * (x ** 2 + 2 * y + fault_sides[0] * z)])

    mask = np.arange(len(mesh.tetrahedra)) % 3 != 0
    # Large enough to cross physical fault planes: the supplied side must persist.
    values, gradients = evaluate_tetrahedral_fields(mesh, evaluator, mask=mask, step=[.1, .2, .3], chunk_size=7)
    points = mesh.nodes[mesh.tetrahedra].mean(axis=1)
    q = points @ linear.T + [4., 5., 6.]
    sides = mesh.block_sides[mesh.tet_blocks, 0]
    factor = mesh.tet_units + 1
    expected_values = factor * (q[:, 0] ** 2 + 2 * q[:, 1] + sides * q[:, 2])
    expected_gradient = factor[:, None] * (np.column_stack((2 * q[:, 0], np.full(len(q), 2.), sides)) @ linear)
    np.testing.assert_allclose(values[mask], expected_values[mask], atol=1e-12)
    np.testing.assert_allclose(gradients[mask], expected_gradient[mask], atol=1e-12)
    assert np.isnan(values[~mask]).all() and np.isnan(gradients[~mask]).all()
    assert max(len(points) for points, _ in calls) <= 7
    for offset in range(0, len(calls), 7):
        for _, side in calls[offset:offset + 7]:
            np.testing.assert_array_equal(side, calls[offset][1])


def test_shared_face_paths_stay_in_cells_and_use_each_local_tensor():
    # These opposite-side tets form a nonconvex union. The direct centroid
    # segment intersects z=0 outside the shared triangle; routing via its
    # centroid keeps the path inside both cells.
    mesh = SimpleNamespace(
        nodes=np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [4., 0, 1], [4., 0, -1]]),
        tetrahedra=np.array([[0, 1, 2, 3], [0, 2, 1, 4]]),
        tet_units=np.zeros(2, dtype=int), tet_blocks=np.zeros(2, dtype=int),
    )
    graph = AnisotropicGraph.from_tetrahedral_mesh(
        mesh, [[0, 0, 1], [0, 0, 1]], anisotropy_ratio=2, continuity_scale=[1., 2.],
    )
    face = mesh.nodes[:3].mean(axis=0)
    expected = 0.
    for position, scale in zip(graph.positions, (1., 2.)):
        delta = face - position
        expected += np.sqrt((delta[0] ** 2 + delta[1] ** 2) / 4 + delta[2] ** 2) / scale
    assert graph.adjacency[0, 1] == pytest.approx(expected)
    assert graph.adjacency[1, 0] == graph.adjacency[0, 1]
    np.testing.assert_allclose(graph.shortest_paths([0]), [[0, expected]])


def test_tetrahedral_domains_cache_export_and_conditioning(tmp_path):
    from gempy_plugins.property_estimation.gaussian_fields import GaussianCovariance, GaussianFieldSimulator

    mesh = _mesh()
    gradients = np.tile([0., 0., 1.], (len(mesh.tetrahedra), 1))
    gradients[0] = np.nan
    graph = AnisotropicGraph.from_tetrahedral_mesh(mesh, gradients, anisotropy_ratio=2.)
    assert graph.metric.invalid_gradient_count == 1
    assert 0 not in graph.node_indices
    assert len(np.unique(graph.component_ids)) == 4
    a, b = graph.node_indices[np.asarray(graph.adjacency.nonzero())]
    np.testing.assert_array_equal(mesh.tet_units[a], mesh.tet_units[b])
    np.testing.assert_array_equal(mesh.tet_blocks[a], mesh.tet_blocks[b])
    geometry = GeologicalEmbedding.from_graph(graph, landmarks=12, dimensions=3)
    path = tmp_path / 'tetrahedral_coordinates.npz'
    geometry.save(path)
    loaded = GeologicalEmbedding.load(path, expected_graph_fingerprint=graph.fingerprint)
    assert loaded.fingerprint == geometry.fingerprint
    assert loaded.grid_shape == (len(mesh.tetrahedra),)
    with pytest.raises(ValueError, match='voxel embedding'):
        loaded.query(graph, graph.positions[:1])
    with pytest.raises(ValueError, match='three-dimensional lattice'):
        AnisotropicGraph.from_metric(graph.metric)
    simulator = GaussianFieldSimulator(loaded, GaussianCovariance(.2), dtype='float64', n_frequencies=64)
    prior = simulator.simulate(realizations=4, seed=8)
    conditional = simulator.condition([0], [.37]).simulate(realizations=4, seed=8)
    np.testing.assert_allclose(conditional.values[:, 0], .37, atol=1e-12)
    other = graph.component_ids != graph.component_ids[0]
    np.testing.assert_array_equal(conditional.values[:, other], prior.values[:, other])
    full_cells = conditional.to_cells()
    assert np.isnan(full_cells[0])
    np.testing.assert_array_equal(full_cells[graph.node_indices], conditional.values[0].numpy())
    across_fault = np.flatnonzero(mesh.tet_blocks[graph.node_indices] != mesh.tet_blocks[graph.node_indices[0]])[0]
    assert np.isinf(graph.shortest_paths([0])[0, across_fault])
    assert simulator.finite_covariance([0], [across_fault]).item() == 0.


def test_mesh_demo_reuses_solved_gempy_model_and_exports_real_tetrahedra(monkeypatch, tmp_path):
    import gempy as gp
    import gempy_engine
    from gempy_plugins.property_estimation.examples.geological_models import make_geological_model
    from gempy_plugins.property_estimation.examples.tetrahedral_property_simulation import run_demo, as_pyvista

    model = make_geological_model('faulted', resolution=(8, 4, 8), compute_gradients=False)
    solution = model.solutions
    grid_values = model.grid.values.copy()

    def unexpected_solve(*args, **kwargs):
        pytest.fail('Meshing and centroid evaluation must reuse the original solve')

    monkeypatch.setattr(gp, 'compute_model', unexpected_solve)
    monkeypatch.setattr(gempy_engine, 'compute_model', unexpected_solve)
    case = run_demo(model=model, cells=80, realizations=3, landmarks=16, n_frequencies=128)
    assert model.solutions is solution
    np.testing.assert_array_equal(model.grid.values, grid_values)
    assert case.report['closed_volumes']
    assert case.report['mesh_quality']['inverted'] == 0
    assert case.report['components'] == 2
    assert case.report['invalid_gradient_count'] == 0
    assert case.report['gradient_half_step_relative_change_max'] < 1e-5
    assert case.report['exact_porosity_observation_max_error'] < 1e-12
    assert case.report['untouched_component_cells'] > 0
    assert case.report['untouched_component_max_latent_change'] == 0.
    cells = case.report['observation_tetrahedra']
    np.testing.assert_allclose(case.cell_data['conditioned_porosity'][cells], [.14, .27, .23], atol=1e-12)
    active = case.graph.node_indices
    assert np.all((case.cell_data['conditioned_porosity'][active] > 0)
                  & (case.cell_data['conditioned_porosity'][active] < 1))
    assert np.all(case.cell_data['posterior_p95_porosity'][active] >= case.cell_data['posterior_p05_porosity'][active])
    pv = pytest.importorskip('pyvista')
    grid, boundary = as_pyvista(case)
    assert grid.n_cells == len(case.mesh.tetrahedra)
    assert boundary.n_cells == len(case.mesh.boundary_triangles)
    assert np.all(grid.celltypes == pv.CellType.TETRA)
    path = tmp_path / 'porosity.vtu'
    grid.save(path)
    restored = pv.read(path)
    np.testing.assert_allclose(restored.cell_data['conditioned_porosity'], case.cell_data['conditioned_porosity'])
