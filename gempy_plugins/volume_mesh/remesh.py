"""Quality remeshing of a labelled tetrahedral mesh with MMG3D (optional ``mmgpy``).

MMG keeps the faces between tetrahedra of different references and the
supplied boundary triangles (with their references), so giving each connected
(unit, block) cell its own reference preserves every interface and fault face,
also fault faces inside one unit.
"""

import time

import numpy as np

from ..optional_dependencies import require_mmgpy


def remesh_mmg(nodes, tets, tet_refs, triangles, triangle_refs, *, hausd=1e-3, hgrad=1.4, hmax=None, hmin=None,
               ar=None):
    """MMG3D remesh; ``tet_refs`` / ``triangle_refs`` are positive integers.

    ``hausd`` (Hausdorff distance to the input surfaces), ``hmax`` and ``hmin``
    are fractions of the box diagonal. ``ar`` is MMG's ridge angle in degrees
    (MMG's default: 45): a crease between two triangles of one surface sharper
    than ``ar`` is kept as a feature edge. The field cut leaves stair-steps of
    up to ~90° along the background grid, so ~85 lets MMG smooth them; edges
    between different references stay sharp whatever ``ar`` is. Returns the
    remeshed nodes, tetrahedra, their references, the boundary triangles with
    references, and the time.
    """
    mmg = require_mmgpy()
    from mmgpy._mmgpy import MmgMesh3D

    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    if np.any(np.asarray(tet_refs) < 1) or np.any(np.asarray(triangle_refs) < 1):
        raise ValueError('invalid_mmg_references: positive references required')
    mesh = MmgMesh3D()
    mesh.set_mesh_size(vertices=len(nodes), tetrahedra=len(tets), triangles=len(triangles))
    mesh.set_vertices(nodes)
    mesh.set_tetrahedra(np.ascontiguousarray(tets, dtype=np.int32), refs=np.asarray(tet_refs, dtype=np.int64))
    mesh.set_triangles(np.ascontiguousarray(triangles, dtype=np.int32), refs=np.asarray(triangle_refs, dtype=np.int64))
    diagonal = float(np.linalg.norm(np.ptp(nodes, axis=0)))
    options = dict(hausd=hausd*diagonal, hgrad=float(hgrad), verbose=-1)
    if hmax is not None:
        options['hmax'] = hmax*diagonal
    if hmin is not None:
        options['hmin'] = hmin*diagonal
    if ar is not None:
        if not 0 < ar <= 180:
            raise ValueError(f'invalid_mmg_ridge_angle: ar must be in (0, 180] degrees, got {ar}')
        options['ar'] = float(ar)
    start = time.perf_counter()
    mesh.remesh(**options)
    seconds = time.perf_counter()-start
    out_tets, refs = mesh.get_elements_with_refs()
    out_triangles, out_triangle_refs = mesh.get_triangles_with_refs()
    return dict(nodes=np.asarray(mesh.get_vertices(), dtype=np.float64), tets=np.asarray(out_tets, dtype=np.int64),
                tet_refs=np.asarray(refs, dtype=np.int64), triangles=np.asarray(out_triangles, dtype=np.int64),
                triangle_refs=np.asarray(out_triangle_refs, dtype=np.int64), seconds=seconds,
                mmg_version=getattr(mmg, '__version__', 'unknown'))
