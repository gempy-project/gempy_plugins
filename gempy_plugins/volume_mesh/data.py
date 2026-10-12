"""Data classes of the volume-first mesher."""

from dataclasses import dataclass, field

import numpy as np

OUTSIDE = ('outside',)


@dataclass
class FieldCut:
    """A background tetrahedral grid cut by every fault and, per fault block, every horizon.

    Every tetrahedron lies on one side of every surface. ``blocks[i]`` indexes
    ``block_sides`` (one ±1 row per fault block, one column per fault);
    ``stack_values`` (S, T) are the tetrahedra's mean one-sided stack values.
    """
    nodes: np.ndarray
    tetrahedra: np.ndarray
    blocks: np.ndarray
    block_sides: np.ndarray
    stack_values: np.ndarray
    stats: dict = field(default_factory=dict)


@dataclass
class VolumeComplex:
    """Closed unit volumes as one conforming triangle complex (the faces of a labelled tet mesh).

    ``unit_sides[i] = (negative-side unit, positive-side unit)`` of triangle
    ``i`` with respect to its winding normal; units are ``(stack, interval)``
    and ``OUTSIDE``. ``patch_kinds`` are 'interface', 'fault' (fault face
    between different units), 'fault_internal' (fault face inside one unit, in
    no shell) and 'extent'; ``patch_surfaces`` the global surface index (-1 on
    the box). ``shell(unit)`` returns that unit's triangles oriented outward.
    """
    vertices: np.ndarray
    triangles: np.ndarray
    unit_sides: list
    patch_kinds: list
    patch_surfaces: list
    vertex_ids: np.ndarray  #: Cut-mesh node of each vertex.
    report: dict = field(default_factory=dict)

    @property
    def units(self):
        return sorted({u for sides in self.unit_sides for u in sides if u != OUTSIDE})

    @property
    def volumes(self):
        return dict(self.report.get('volumes', {}))

    def shell(self, unit):
        """(vertices, outward triangles) of one unit, indexing the shared vertices."""
        rows = []
        for triangle, (negative, positive) in zip(self.triangles, self.unit_sides):
            if negative == positive:
                continue
            if negative == unit:
                rows.append(triangle)
            elif positive == unit:
                rows.append(triangle[[0, 2, 1]])
        return self.vertices, np.asarray(rows, dtype=np.int64).reshape(-1, 3)


@dataclass
class VolumeMesh:
    """Conforming tetrahedral mesh of the model's units, in model (world) coordinates.

    ``tet_units[i]`` indexes ``units`` ((stack, interval) pairs) and
    ``tet_blocks[i]`` the fault blocks (rows of ``block_sides``). ``unit_ids``
    are the production lithology ids of ``units`` and ``unit_names`` their
    element names when known. ``boundary_triangles`` lie on the complex: box,
    interfaces and faults, with ``boundary_kinds`` and ``boundary_surfaces``
    (global surface index, -1 on the box; ``surface_names`` when known).
    ``complex`` is the closed-volume complex before remeshing.
    """
    nodes: np.ndarray
    tetrahedra: np.ndarray
    tet_units: np.ndarray
    tet_blocks: np.ndarray
    units: list
    unit_ids: np.ndarray
    block_sides: np.ndarray
    boundary_triangles: np.ndarray
    boundary_kinds: np.ndarray
    boundary_surfaces: np.ndarray
    unit_names: list = None
    surface_names: list = None
    complex: VolumeComplex = None
    report: dict = field(default_factory=dict)

    @property
    def unit_volumes(self):
        p = self.nodes[self.tetrahedra]
        volumes = np.einsum('ij,ij->i', p[:, 1]-p[:, 0], np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]))/6
        totals = np.bincount(self.tet_units, weights=volumes, minlength=len(self.units))
        return {unit: float(v) for unit, v in zip(self.units, totals)}
