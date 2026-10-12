"""
Tetrahedral Meshes for Simulation
=================================

Conforming unit tetrahedra of a faulted model, cut straight from its scalar fields.
"""

# %%
# Dual contouring gives good surfaces for display, but a simulation mesh needs
# closed, conforming volumes: every unit a watertight region, faults and
# interfaces shared by the tetrahedra on both sides. This plugin builds them
# *volume-first*: a background tetrahedral grid fills the model box and is cut
# by every fault, then, inside each fault block, by every horizon, evaluating
# the model's own fields with the solved weights. Each fault block sees its own
# side of the fault drift, so the throw is never smeared across the fault.
# The cut is conforming by construction; MMG3D (``pip install mmgpy``) then
# improves element quality while keeping every interface and fault face.

# %%
import gempy as gp
from gempy.core.data.enumerators import ExampleModel
from gempy_engine.config import AvailableBackends

from gempy_plugins.volume_mesh import compute_volume_mesh, to_pyvista, write_volume_mesh

# %%
# A faulted model with an unconformity
# ------------------------------------
# The field evaluation needs a float64 engine backend.

geo_model = gp.generate_example_model(ExampleModel.COMBINATION, compute_model=False)
gp.compute_model(geo_model, engine_config=gp.data.GemPyEngineConfig(backend=AvailableBackends.numpy,
                                                                    dtype='float64'))

# %%
# The volume mesh
# ---------------
# ``cells`` is the approximate number of background cubes; ``snap`` moves a
# surface onto a grid vertex within that fraction of an edge (fewer slivers).

mesh = compute_volume_mesh(geo_model, cells=20_000, snap=.15, remesh=True)
print(f"{len(mesh.tetrahedra)} tetrahedra, units {mesh.unit_names}, {len(mesh.block_sides)} fault blocks")
print("quality:", {k: round(v, 4) for k, v in mesh.report['quality'].items()})
print("unit volumes:", {name: round(v) for name, v in zip(mesh.unit_names, mesh.unit_volumes.values())})

# %%
# Each tetrahedron carries its unit, production lithology id and fault block;
# boundary triangles carry their surface and kind (interface, fault,
# fault_internal, extent).

grid, boundary = to_pyvista(mesh)
grid.plot(scalars='unit_id', show_edges=True, cmap='viridis')

# %%
# Export
# ------
# ``write_volume_mesh`` writes the tetrahedra (``.vtu``) and boundary triangles
# (``_boundary.vtp``), readable by ParaView and meshio-based simulators.

write_volume_mesh(mesh, 'combination_volume_mesh.vtu')
