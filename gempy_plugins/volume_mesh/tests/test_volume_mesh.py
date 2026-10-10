"""Volume-first meshing: background grid, field cut, closed complex, remeshing and the GeoModel API."""

from collections import Counter

import numpy as np
import pytest

from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine import classify_units

from gempy_plugins.volume_mesh.background_grid import kuhn_grid, signed_volumes
from gempy_plugins.volume_mesh.field_cut import _Mesh, cut_field_volumes
from gempy_plugins.volume_mesh.quality import dihedral_angles, quality_report
from gempy_plugins.volume_mesh.volume_complex import (OUTWARD_FACES, closed_volume_complex, connected_cells,
                                                      count_hanging_nodes)


def _face_counts(tets):
    return Counter(map(tuple, np.sort(tets[:, OUTWARD_FACES].reshape(-1, 3), axis=1).tolist()))


# region Modules on analytic fields

def test_kuhn_grid_fills_the_box_conformingly():
    nodes, tets = kuhn_grid((0, 2, 0, 3, 0, 1), (2, 3, 1))
    volumes = signed_volumes(nodes, tets)
    assert np.all(volumes > 0) and volumes.sum() == pytest.approx(6.)
    counts = _face_counts(tets)
    assert set(counts.values()) <= {1, 2}
    lone = np.array([face for face, n in counts.items() if n == 1])
    on_box = np.isclose(nodes[lone], nodes.min(axis=0)) | np.isclose(nodes[lone], nodes.max(axis=0))
    assert on_box.all(axis=1).any(axis=1).all()


def test_edge_point_insertion_keeps_the_mesh_conforming():
    nodes, tets = kuhn_grid((0, 1, 0, 1, 0, 1), (3, 3, 3))
    mesh = _Mesh(nodes, tets, np.zeros(len(tets)))
    rng = np.random.default_rng(0)
    for _ in range(3):
        edges = np.unique(np.sort(mesh.tets[:, [[0, 1], [1, 2], [0, 3], [2, 3]]].reshape(-1, 2), axis=1), axis=0)
        chosen = edges[rng.choice(len(edges), 40, replace=False)]
        # Two points on some edges, in random order.
        lo = np.concatenate((chosen[:, 0], chosen[:10, 0]))
        hi = np.concatenate((chosen[:, 1], chosen[:10, 1]))
        mesh.insert(lo, hi, rng.uniform(.1, .9, len(lo)))
    volumes = signed_volumes(mesh.xyz, mesh.tets)
    assert np.all(volumes > 0) and volumes.sum() == pytest.approx(1., rel=1e-12)
    assert set(_face_counts(mesh.tets).values()) <= {1, 2}


def _two_faults(shape=(9, 9, 9), snap=0.):
    """Faults x = 3.3 and x + .4 z = 5.9; one horizon thrown across each."""
    def fault_values(points):
        return np.stack([points[:, 0]-3.3, points[:, 0]+.4*points[:, 2]-5.9])

    def block_fields(points, stacks, sides):
        assert list(stacks) == [2]
        return np.stack([points[:, 2]-3.4-.45*(sides[0] > 0)-.3*(sides[1] > 0)])

    nodes, tets = kuhn_grid((0, 8, 0, 8, 0, 8), shape)
    isovalues = [np.array([0.]), np.array([0.]), np.array([0.])]
    cut = cut_field_volumes(nodes, tets, fault_values=fault_values, block_fields=block_fields,
                            stack_isovalues=isovalues, fault_stacks=[0, 1], snap=snap)
    stack, interval = classify_units(cut.stack_values, [R.FAULT, R.FAULT, R.BASEMENT], isovalues)
    complex_ = closed_volume_complex(cut, stack, interval, signed_volumes(cut.nodes, cut.tetrahedra),
                                     [0, 1], [(2, 0., 2)])
    return cut, stack, interval, complex_


def test_two_crossing_faults_give_exact_piecewise_linear_volumes():
    cut, stack, interval, volumes = _two_faults()
    report = volumes.report
    assert report['all_closed'] and report['hanging_nodes'] == 0 and report['volumes_sum_to_region']
    assert report['blocks'] == 4
    # Below the horizon: 3.4 left of x=3.3, 3.85 right of it, 4.15 right of the tilted fault.
    assert volumes.volumes[(2, 1)] == pytest.approx(243.4, rel=1e-12)
    assert volumes.volumes[(2, 0)] == pytest.approx(268.6, rel=1e-12)
    kinds = np.array(volumes.patch_kinds)
    assert {'interface', 'fault', 'fault_internal', 'extent'} == set(kinds)
    heights = volumes.vertices[np.unique(volumes.triangles[kinds == 'interface'])][:, 2]
    assert np.isclose(heights[:, None], [3.4, 3.85, 4.15], rtol=0, atol=1e-12).any(axis=1).all()
    assert all(np.array(volumes.patch_surfaces)[kinds == 'interface'] == 2)
    internal = np.flatnonzero(kinds == 'fault_internal')
    assert all(volumes.unit_sides[i][0] == volumes.unit_sides[i][1] for i in internal)
    assert count_hanging_nodes(volumes.vertices, volumes.triangles) == 0
    # One connected (unit, block) cell per region seed.
    labels = np.unique(np.stack((interval, cut.blocks), axis=1), axis=0, return_inverse=True)[1].reshape(-1)
    assert connected_cells(cut.tetrahedra, labels).max()+1 == len(report['region_seeds']) == 7


def test_snapping_moves_surfaces_by_less_than_the_snap_fraction():
    *_, exact = _two_faults()
    *_, snapped = _two_faults(snap=.15)
    assert snapped.report['all_closed'] and snapped.report['hanging_nodes'] == 0
    for unit, volume in exact.volumes.items():
        assert abs(snapped.volumes[unit]-volume) < .15*8/9*64
    assert len(snapped.triangles) < len(exact.triangles)


def test_quality_of_a_regular_tetrahedron():
    nodes = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    tets = np.array([[0, 1, 2, 3]])
    if signed_volumes(nodes, tets)[0] < 0:
        tets = tets[:, [0, 2, 1, 3]]
    np.testing.assert_allclose(dihedral_angles(nodes, tets), np.degrees(np.arccos(1/3)))
    assert quality_report(nodes, tets)['shape_median'] == pytest.approx(1.)


def test_mmg_remesh_keeps_units_and_fault_faces():
    pytest.importorskip('mmgpy')
    from gempy_plugins.volume_mesh.remesh import remesh_mmg

    cut, stack, interval, complex_ = _two_faults(shape=(10, 10, 10), snap=.15)
    labels = interval*len(cut.block_sides)+cut.blocks
    cells = connected_cells(cut.tetrahedra, labels)
    kinds = {}
    refs = np.array([kinds.setdefault(k, len(kinds)+1) for k in complex_.patch_kinds])
    result = remesh_mmg(cut.nodes, cut.tetrahedra, cells+1, complex_.vertex_ids[complex_.triangles], refs)
    assert np.all(signed_volumes(result['nodes'], result['tets']) > 0)
    cell_interval = np.zeros(cells.max()+1, dtype=int)
    cell_interval[cells] = interval
    out = cell_interval[result['tet_refs']-1]
    below = signed_volumes(result['nodes'], result['tets'])[out == 1].sum()
    assert below == pytest.approx(sum(v for u, v in complex_.volumes.items() if u[1] == 1), rel=5e-3)
    assert kinds['fault_internal'] in set(result['triangle_refs'].tolist())

# endregion


# region GeoModel API

@pytest.fixture(scope='module')
def combination():
    import gempy as gp
    from gempy.core.data.enumerators import ExampleModel
    from gempy_engine.config import AvailableBackends

    geo_model = gp.generate_example_model(ExampleModel.COMBINATION, compute_model=False)
    gp.compute_model(geo_model, engine_config=gp.data.GemPyEngineConfig(
        backend=AvailableBackends.numpy, dtype='float64'))
    return geo_model


@pytest.mark.parametrize('remesh', [False, True])
def test_combination_model_volume_mesh(combination, remesh, tmp_path):
    from gempy_plugins.volume_mesh import compute_volume_mesh, write_volume_mesh

    if remesh:
        pytest.importorskip('mmgpy')
    mesh = compute_volume_mesh(combination, cells=4000, remesh=remesh)
    assert mesh.unit_names == ['rock3', 'rock2', 'rock1', 'basement']
    assert len(mesh.block_sides) == 2 and set(mesh.tet_blocks.tolist()) == {0, 1}
    assert mesh.report['complex']['all_closed'] and mesh.report['quality']['inverted'] == 0
    assert set(mesh.boundary_kinds.tolist()) == {'interface', 'fault', 'fault_internal', 'extent'}
    # Model coordinates: the box is the model extent and the units fill it.
    extent = np.asarray(combination.grid.regular_grid.extent, dtype=float)
    np.testing.assert_allclose(mesh.nodes.min(axis=0), extent[::2], atol=1e-6*np.ptp(extent))
    np.testing.assert_allclose(mesh.nodes.max(axis=0), extent[1::2], atol=1e-6*np.ptp(extent))
    assert sum(mesh.unit_volumes.values()) == pytest.approx(np.prod(extent[1::2]-extent[::2]), rel=1e-9)
    assert mesh.report['unit_volume_change'] < (1e-2 if remesh else 1e-9)
    fault = [i for i, name in enumerate(mesh.surface_names) if name == 'fault']
    assert set(mesh.boundary_surfaces[mesh.boundary_kinds == 'fault'].tolist()) == set(fault)
    path = write_volume_mesh(mesh, tmp_path/'combination')
    assert path.exists() and path.with_name('combination_boundary.vtp').exists()

# endregion
