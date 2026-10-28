"""Volume-first tetrahedral meshes of GemPy models, cut from the model's own fields.

Orchestration only: the engine evaluates the fields (``prepare_field_evaluator``)
and classifies units (``classify_units``); the modules of this package build the
background grid, cut it, extract and audit the closed-volume complex, remesh it
with MMG3D and export it. No dual-contouring surface is used.
"""

import copy
import time

import numpy as np

from .background_grid import background_shape, kuhn_grid, signed_volumes
from .data import OUTSIDE, VolumeMesh
from .export import to_polydata, to_unstructured_grid
from .field_cut import cut_field_volumes
from .quality import quality_report
from .remesh import remesh_mmg
from .volume_complex import closed_volume_complex, connected_cells

AUTO_HMAX_CELL_EDGES = 4  # hmax='auto': longest edge MMG may make, in background cube edges


def compute_volume_mesh(geo_model, *, cells=50_000, snap=.15, remesh=True, hausd=1e-3, hgrad=1.4, hmax='auto',
                        hmin=None, ar=None, engine_config=None) -> VolumeMesh:
    """Conforming tetrahedral mesh of every unit of ``geo_model``, in model coordinates.

    Solves the model once more (meshes off) so the field evaluator sees exactly
    the inputs of its solve. ``cells`` is the approximate number of background
    cubes; ``snap`` the fraction of an edge within which a surface moves onto a
    grid vertex (fewer slivers, surfaces move by at most that much). With
    ``remesh`` the cut mesh is improved by MMG3D (needs ``mmgpy``); ``hausd``,
    ``hmax`` and ``hmin`` are fractions of the box diagonal; ``hmax='auto'``
    caps edges at four background cube edges (None: no cap, which lets MMG
    coarsen flat models to a few dozen tetrahedra), ``ar`` is MMG's
    ridge angle in degrees (see ``remesh_mmg``; ~85 removes most slivers left
    by the cut's stair-steps, None keeps MMG's 45). The engine backend
    must be float64 (NumPy or Torch, optionally KeOps); ``engine_config`` sets
    it as ``gempy.compute_model`` does.
    """
    import gempy_engine
    from gempy.modules.data_manipulation import (input_data_descriptor_from_geo_model,
                                                 interpolation_input_from_structural_frame)
    from gempy_engine import prepare_field_evaluator

    if engine_config is not None:
        _set_backend(engine_config)
    start = time.perf_counter()
    interpolation_input = interpolation_input_from_structural_frame(geo_model)
    descriptor = input_data_descriptor_from_geo_model(geo_model)
    options = copy.deepcopy(geo_model.interpolation_options)
    options.evaluation_options.mesh_extraction = False
    solutions = gempy_engine.compute_model(interpolation_input, options, descriptor)
    solve_seconds = time.perf_counter()-start
    evaluator = prepare_field_evaluator(solutions, interpolation_input, descriptor, options)
    extent = _host(solutions.octrees_output[0].grid.octree_grid.orthogonal_extent).astype(float)
    mesh = volume_mesh_from_evaluator(
        evaluator, extent, cells=cells, snap=snap, remesh=remesh, hausd=hausd, hgrad=hgrad, hmax=hmax, hmin=hmin,
        ar=ar, unit_values=interpolation_input.unit_values,
        number_of_surfaces_per_stack=descriptor.stack_structure.number_of_surfaces_per_stack)
    names = geo_model.structural_frame.elements_names
    mesh.unit_names = [names[i-1] for i in mesh.unit_ids.tolist()]
    mesh.surface_names = names[:-1]
    mesh.nodes = _to_world(geo_model, mesh.nodes)
    mesh.complex.vertices = _to_world(geo_model, mesh.complex.vertices)
    mesh.report['seconds']['solve'] = solve_seconds
    mesh.report['coordinates'] = 'model'
    return mesh


def volume_mesh_from_evaluator(evaluator, extent, *, cells=50_000, snap=.15, remesh=True, hausd=1e-3, hgrad=1.4,
                               hmax='auto', hmin=None, ar=None, unit_values=None,
                               number_of_surfaces_per_stack=None) -> VolumeMesh:
    """Tetrahedral mesh of ``extent`` using an already prepared field evaluator.

    Coordinates follow the evaluator: engine coordinates for an engine
    ``FieldEvaluator``, or world coordinates for ``gp.prepare_field_evaluator``.
    Supply ``extent`` in the same system. The report defaults to 'engine'; a
    world-coordinate caller should set ``mesh.report['coordinates'] = 'model'``.

    ``unit_values`` and ``number_of_surfaces_per_stack`` (from the engine
    inputs) give the production unit ids; without them ``unit_ids`` are -1.
    ``hmax='auto'`` is ``AUTO_HMAX_CELL_EDGES`` background cube edges.
    """
    from gempy_engine import classify_units
    from gempy_engine.modules.activator.unit_classification import unit_ids as production_unit_ids

    seconds = {}
    clock = time.perf_counter()
    shape = background_shape(extent, cells)
    if isinstance(hmax, str):
        if hmax != 'auto':
            raise ValueError(f"invalid_hmax: a fraction of the box diagonal, None or 'auto', got {hmax!r}")
        hmax = _auto_hmax(extent, shape)
    nodes, tets = kuhn_grid(extent, shape)
    cut = cut_field_volumes(
        nodes, tets, fault_values=evaluator.fault_values,
        block_fields=lambda points, stacks, sides: evaluator(points, stacks=stacks, fault_sides=sides),
        stack_isovalues=evaluator.stack_isovalues, fault_stacks=evaluator.fault_stacks, snap=snap)
    seconds['cut'] = time.perf_counter()-clock

    # region Units, surfaces and the closed-volume complex
    clock = time.perf_counter()
    unit_stack, interval = classify_units(cut.stack_values, evaluator.stack_relations, evaluator.stack_isovalues)
    offsets = np.concatenate(([0], np.cumsum([len(levels) for levels in evaluator.stack_isovalues])))
    fault_surfaces = [int(offsets[f]) for f in evaluator.fault_stacks]
    horizons = [(g, float(level), int(offsets[g]+k)) for g, levels in enumerate(evaluator.stack_isovalues)
                if g not in evaluator.fault_stacks for k, level in enumerate(levels)]
    tet_volumes = signed_volumes(cut.nodes, cut.tetrahedra)
    complex_ = closed_volume_complex(cut, unit_stack, interval, tet_volumes, fault_surfaces, horizons)
    _check_closed(complex_.report)
    units = complex_.units
    unit_index = {u: i for i, u in enumerate(units)}
    tet_unit = np.array([unit_index[u] for u in zip(unit_stack.tolist(), interval.tolist())], dtype=np.int64)
    cell = connected_cells(cut.tetrahedra, tet_unit*len(cut.block_sides)+cut.blocks)
    cell_unit = np.zeros(cell.max()+1, dtype=np.int64)
    cell_block = np.zeros(cell.max()+1, dtype=np.int64)
    cell_unit[cell], cell_block[cell] = tet_unit, cut.blocks
    seconds['complex'] = time.perf_counter()-clock
    # endregion

    # region Boundary labels and remeshing
    labels = {}
    triangle_refs = np.array([labels.setdefault((kind, sides, surface), len(labels)+1) for kind, sides, surface in
                              zip(complex_.patch_kinds, complex_.unit_sides, complex_.patch_surfaces)], dtype=np.int64)
    label_kinds = np.array([''] + [kind for kind, _, _ in labels], dtype=object)
    label_surfaces = np.array([-1] + [surface for _, _, surface in labels], dtype=np.int64)
    boundary = complex_.vertex_ids[complex_.triangles]
    report = dict(background_shape=shape.tolist(), snap=snap, hmax=hmax, ar=ar, cut=cut.stats,
                  complex=complex_.report, cut_quality=quality_report(cut.nodes, cut.tetrahedra),
                  cells=int(cell.max()+1), seconds=seconds)
    if remesh:
        clock = time.perf_counter()
        result = remesh_mmg(cut.nodes, cut.tetrahedra, cell+1, boundary, triangle_refs, hausd=hausd, hgrad=hgrad,
                            hmax=hmax, hmin=hmin, ar=ar)
        seconds['remesh'] = time.perf_counter()-clock
        out_nodes, out_tets, out_cells = result['nodes'], result['tets'], result['tet_refs']-1
        boundary, triangle_refs = result['triangles'], result['triangle_refs']
        report['mmg_version'] = result['mmg_version']
    else:
        out_nodes, out_tets, out_cells = cut.nodes, cut.tetrahedra, cell
    report['quality'] = quality_report(out_nodes, out_tets)
    # endregion

    tet_units, tet_blocks = cell_unit[out_cells], cell_block[out_cells]
    ids = np.full(len(units), -1, dtype=np.int64)
    if unit_values is not None and number_of_surfaces_per_stack is not None:
        stacks, intervals = np.array(units, dtype=np.int64).reshape(-1, 2).T
        ids = production_unit_ids(stacks, intervals, number_of_surfaces_per_stack, _host(unit_values))
    volumes = np.bincount(tet_units, weights=signed_volumes(out_nodes, out_tets), minlength=len(units))
    report['unit_volume_change'] = float(max(abs(volumes[i]/complex_.volumes[u]-1) for i, u in enumerate(units)))
    report['coordinates'] = 'engine'
    return VolumeMesh(nodes=out_nodes, tetrahedra=out_tets, tet_units=tet_units, tet_blocks=tet_blocks, units=units,
                      unit_ids=np.asarray(ids, dtype=np.int64), block_sides=cut.block_sides,
                      boundary_triangles=boundary, boundary_kinds=label_kinds[triangle_refs].astype(str),
                      boundary_surfaces=label_surfaces[triangle_refs], complex=complex_, report=report)


def to_pyvista(mesh: VolumeMesh):
    """(tetrahedra ``UnstructuredGrid``, boundary ``PolyData``) with unit, block and surface arrays."""
    grid = to_unstructured_grid(mesh.nodes, mesh.tetrahedra, dict(
        unit=mesh.tet_units, unit_id=mesh.unit_ids[mesh.tet_units], block=mesh.tet_blocks))
    boundary = to_polydata(mesh.nodes, mesh.boundary_triangles, dict(
        surface=mesh.boundary_surfaces, kind=mesh.boundary_kinds))
    return grid, boundary


def write_volume_mesh(mesh: VolumeMesh, path):
    """Write the tetrahedra (``.vtu``) and their boundary triangles (``<stem>_boundary.vtp``)."""
    from pathlib import Path

    path = Path(path).with_suffix('.vtu')
    grid, boundary = to_pyvista(mesh)
    grid.save(path)
    boundary.save(path.with_name(path.stem+'_boundary.vtp'))
    return path


def _auto_hmax(extent, shape):
    """``AUTO_HMAX_CELL_EDGES`` background cube edges, as a fraction of the box diagonal."""
    lengths = np.asarray(extent, dtype=float)[1::2]-np.asarray(extent, dtype=float)[::2]
    return float(AUTO_HMAX_CELL_EDGES*np.max(lengths/shape)/np.linalg.norm(lengths))


def _check_closed(report):
    if not (report['all_closed'] and report['volumes_sum_to_region'] and not report['hanging_nodes']):
        raise ValueError('closed_volume_audit_failed: closed={} volume_error={:.3e} hanging_nodes={} '
                         'degenerate_triangles={}'.format(report['all_closed'], report['volume_error'],
                                                          report['hanging_nodes'], report['degenerate_triangles']))


def _set_backend(engine_config):
    from gempy_engine.core.backend_tensor import BackendTensor

    BackendTensor._change_backend(engine_backend=engine_config.backend, use_gpu=engine_config.use_gpu,
                                  dtype=engine_config.dtype, grads=engine_config.compute_grads)


def _to_world(geo_model, points):
    points = geo_model.input_transform.apply_inverse(np.asarray(points, dtype=float))
    return geo_model.grid.transform.apply_inverse_with_cached_pivot(points)


def _host(values):
    return values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values)


__all__ = ['compute_volume_mesh', 'volume_mesh_from_evaluator', 'to_pyvista', 'write_volume_mesh', 'OUTSIDE']
