"""
Auto-Generating Orientations from a Point Cloud
==================================================

Deriving orientation data from clusters of surface points, when no dip/strike
measurements are available.
"""

# %%
# Many real datasets -- points digitized from a geological map, contour lines, or a
# dense borehole/outcrop survey -- give you plenty of surface points but no orientation
# measurements at all. gempy's interpolation needs at least one orientation per
# structural group to compute a model, so normally you'd have to guess one by hand,
# which only ever captures the *average* tilt of a surface, not how it actually varies.
#
# The ``orientations_generator`` plugin instead derives a plausible local orientation
# directly from the point cloud itself: for each point, it fits a plane through that
# point's nearest neighbors, and uses the plane's normal as the orientation there.

# %%
import numpy as np

import gempy as gp
import gempy_viewer as gpv
from gempy_plugins.orientations_generator import select_nearest_surfaces_points

np.random.seed(42)


# %%
# A synthetic, folded horizon
# ------------------------------
# Standing in for a set of points digitized off a geological map: a grid of XY
# locations, each with a Z read off a folded surface -- and nothing else.

# %%
def folded_surface(x, y):
    return 500 + 350 * np.sin(np.pi * x / 1000) * np.cos(np.pi * y / 2000)


grid_x, grid_y = np.meshgrid(np.linspace(50, 1950, 10), np.linspace(200, 1800, 6))
x, y = grid_x.ravel(), grid_y.ravel()
z = folded_surface(x, y)

surface_points_xyz = np.column_stack([x, y, z])
surface_points_xyz.shape

# %%
# Setting up the model
# -----------------------
# One element, no orientations yet:

# %%
geo_model = gp.create_geomodel(
    project_name='auto_orientations',
    extent=[0, 2000, 0, 2000, 0, 1000],
    resolution=[50, 50, 50],
    refinement=4,
    structural_frame=gp.data.StructuralFrame.initialize_default_structure()
)
geo_model.structural_frame.structural_elements[0].name = 'Horizon'

gp.add_surface_points(
    geo_model=geo_model,
    x=surface_points_xyz[:, 0],
    y=surface_points_xyz[:, 1],
    z=surface_points_xyz[:, 2],
    elements_names='Horizon'
)

# %%
# With 60 surface points and zero orientations, this model can't be computed yet --
# gempy needs at least one orientation per structural group before it can interpolate.
#
# Finding local neighborhoods
# ------------------------------
# Instead of guessing one orientation for the whole surface, look up each point's four
# nearest neighbors (itself included) and fit a local plane through them:

# %%
neighbours = select_nearest_surfaces_points(
    surface_points_xyz=surface_points_xyz,
    searchcrit=4
)
neighbours.shape

# %%
# Generating one orientation per point
# ----------------------------------------
# ``gp.create_orientations_from_surface_points_coords`` turns each neighborhood into an
# orientation, centered on that local plane's centroid:

# %%
orientations = gp.create_orientations_from_surface_points_coords(
    xyz_coords=surface_points_xyz,
    subset=neighbours,
    element_name='Horizon'
)

gp.add_orientations(
    geo_model=geo_model,
    x=orientations.data['X'],
    y=orientations.data['Y'],
    z=orientations.data['Z'],
    pole_vector=orientations.grads,
    elements_names='Horizon'
)

# %%
# Computing and visualizing
# -----------------------------
# 60 auto-generated orientations later, the model already reflects the fold -- without a
# single orientation having been picked by hand. With this many independent, closely-
# spaced orientations, gempy needs its input coordinates rescaled onto a numerically
# stable range for kriging before computing, via ``update_transform``:

# %%
geo_model.update_transform(gp.data.GlobalAnisotropy.CUBE)
gp.compute_model(geo_model)

# %%
gpv.plot_3d(geo_model, show_lith=False, show_surfaces=True, show_data=True)

# %%
gpv.plot_2d(geo_model, direction='y', show_data=True)