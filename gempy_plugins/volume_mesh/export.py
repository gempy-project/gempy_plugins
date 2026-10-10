"""Export of volume meshes as VTK unstructured grids (optional ``pyvista``)."""

import numpy as np

from ..optional_dependencies import require_pyvista


def to_unstructured_grid(nodes, tets, cell_data=None):
    """``pyvista.UnstructuredGrid`` of tetrahedra with per-cell arrays."""
    pv = require_pyvista()
    tets = np.asarray(tets, dtype=np.int64)
    cells = np.column_stack((np.full(len(tets), 4), tets)).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8),
                               np.asarray(nodes, dtype=float))
    for name, values in (cell_data or {}).items():
        grid.cell_data[name] = np.asarray(values)
    return grid


def to_polydata(vertices, triangles, cell_data=None):
    """``pyvista.PolyData`` of triangles with per-face arrays."""
    pv = require_pyvista()
    triangles = np.asarray(triangles, dtype=np.int64)
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    surface = pv.PolyData(np.asarray(vertices, dtype=float), faces)
    for name, values in (cell_data or {}).items():
        surface.cell_data[name] = np.asarray(values)
    return surface
