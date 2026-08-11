"""
Planar Dipping Layers from GIS Data
========================================

A two-layer, south-dipping structural model built from digitized vector layers
(interfaces, topographic contours, strike lines) rather than a pre-made DEM or
orientation layer. Reproduces cgre-aachen/gemgis's `"Example 1 -- Planar Dipping
Layers" <https://gemgis.readthedocs.io/en/latest/getting_started/example/example01.html>`_
dataset.
"""

# %%
# Two ways to derive orientations, compared below: `orientations_from_strike_lines`
# reads dip/azimuth off elevation-paired strike lines, the classical hand method for a
# digitized map; `gp.create_orientations_from_surface_points_coords` fits a best-fit
# plane through a formation's own draped surface points instead, with no separate
# orientation data required at all.

# %%
import os

import gempy as gp
import gempy_viewer as gpv
from gempy.modules.data_manipulation.manipulate_points import compute_adp_from_gradients

import gempy_plugins.gis_data as _gis_data_pkg
from gempy_plugins.gis_data.conversion import (
    nearest_contour_elevation, orientations_from_strike_lines, raster_to_xyz, relabel_orientations,
    surface_points_from_geodataframe,
)
from gempy_plugins.gis_data.interpolation import interpolate_raster
from gempy_plugins.gis_data.io import (
    export_orientations_csv, export_surface_points_csv, orientations_to_dataframe, surface_points_to_dataframe,
)
from gempy_plugins.gis_data.plotting import plot_input_layers, plot_raster_2d, plot_raster_3d
from gempy_plugins.optional_dependencies import require_geopandas

gpd = require_geopandas()

# %%
# Input data
# --------------
# Interface lines, topographic contours, and strike lines -- vendored from
# `cgre-aachen/gemgis_data
# <https://github.com/cgre-aachen/gemgis_data/tree/main/data/example01_planar_dipping_layers>`_
# (CC BY 4.0, see `examples/data_01/README.rst`). The DEM is interpolated from the
# contours directly, no pre-made raster needed.
#
# Located via the installed `gempy_plugins.gis_data` package rather than this script's
# own `__file__` -- IDEs that run `# %%` cells through a plain IPython/Jupyter kernel
# (rather than as a full script) don't define `__file__` for the executed cell at all.
_DATA = os.path.join(os.path.dirname(_gis_data_pkg.__file__), "examples", "data_01")

interfaces = gpd.read_file(os.path.join(_DATA, "interfaces.shp"))
topo_contours = gpd.read_file(os.path.join(_DATA, "topo.shp"))
strike_lines = gpd.read_file(os.path.join(_DATA, "strike_lines.shp"))

# %%
# All three raw layers together: topographic contours (brown, elevation-labeled),
# digitized interfaces (one color per formation), strike lines (black).
plot_input_layers(interfaces=interfaces, topo_contours=topo_contours, strike_lines=strike_lines)

# %%
# Building a DEM from the topographic contours
# ------------------------------------------------------
# `interpolate_raster` fits a radial basis function through every contour vertex (each
# one a sample point) to build a full elevation grid. `bbox` fixes the DEM's
# real-world extent; `resolution` its cell count -- together they set cell size, e.g.
# a 972 m wide `bbox` at `resolution=200` gives ~4.9 m cells.
#
# `dem_transform` is the grid's affine transform: `x, y = transform * (col, row)`.
# Every function downstream that consumes `dem` needs it too, since `dem` alone is
# just a plain array with no notion of where it sits in space.
_minx, _miny, _maxx, _maxy = topo_contours.total_bounds
dem, dem_transform = interpolate_raster(
    topo_contours, value_column="Z", resolution=(200, 200), bbox=(_minx, _miny, _maxx, _maxy),
)

# %%
# DEM, top-down and as a 3D relief surface.
plot_raster_2d(dem, dem_transform)
plot_raster_3d(dem, dem_transform)

# %%
# Convert to GemPy input
# -------------------------
# Interface points come from draping the digitized contact lines onto the DEM.
surface_points = surface_points_from_geodataframe(
    interfaces, formation_column="formation", dem=dem, dem_transform=dem_transform,
)
print(surface_points_to_dataframe(surface_points))  # pretty-print as a GemPy-format table
# export_surface_points_csv(surface_points, "surface_points.csv")  # write to GemPy's own CSV format

# %%
# Deriving orientations from strike lines
# --------------------------------------------
# `strike_lines.shp` carries only an `id` column, no elevation and no formation of its
# own -- by construction, each strike line sits exactly on one `topo.shp` contour
# (distance 0), so `nearest_contour_elevation` recovers each line's elevation from its
# nearest contour.
strike_lines = strike_lines.copy()
strike_lines["elevation"] = nearest_contour_elevation(strike_lines, topo_contours)
strike_lines["formation"] = "contact"  # no formation of its own -- a single planar contact
raw_orientations = orientations_from_strike_lines(
    strike_lines, formation_column="formation", elevation_column="elevation",
)

# %%
# One shared dip for the whole two-layer stack, so it doesn't matter which one real
# formation name `relabel_orientations` folds it into here: GemPy interpolates one
# combined scalar field per series from *all* of that series' surface points and
# orientations pooled together (see `StructuralFrame.from_data_tables`), not a
# separate field per formation, so this one orientation constrains both (see
# `example02.py` for the general case, where several real formations each get their
# own strike-line-derived orientation instead).
orientations = relabel_orientations(
    raw_orientations, rename={"contact": "Sand1"}, name_id_map=surface_points.name_id_map,
)
print(orientations_to_dataframe(orientations))  # pretty-print as a GemPy-format table
# export_orientations_csv(orientations, "orientations.csv")  # write to GemPy's own CSV format

_azimuth, _dip, _ = compute_adp_from_gradients(
    orientations.data["G_x"], orientations.data["G_y"], orientations.data["G_z"],
)
print(f"orientations_from_strike_lines: dip={_dip.mean():.1f} deg, azimuth={_azimuth.mean():.1f} deg "
      f"(n={len(orientations)}, hand-digitized ground truth: 30.5 deg / 180.0 deg)")

# %%
# Alternative: with no strike lines (or other dip/azimuth data) at all,
# `create_orientations_from_surface_points_coords` fits a plane through one formation's
# own points instead. It fits one point cloud at a time, so for several formations
# you'd call it once per formation and combine the results. Uncomment to try it:
#
# plane_fit = gp.create_orientations_from_surface_points_coords(
#     xyz_coords=surface_points.get_surface_points_by_id(surface_points.name_id_map["Sand1"]).xyz,
#     element_name="Sand1",
# )

# %%
# Build and compute the model
# -------------------------------
geo_model = gp.create_geomodel(
    project_name="gis_data_example01",
    extent=[0, 1000, 0, 1000, 200, 800],
    resolution=[50, 50, 50],
    structural_frame=gp.data.StructuralFrame.from_data_tables(surface_points, orientations),
)

# `set_topography_from_gdal`/`set_topography_from_array` raise `NotImplementedError`,
# so `raster_to_xyz` flattens the DEM into the XYZ vertices `set_topography_from_arrays`
# expects.
gp.set_topography_from_arrays(geo_model.grid, raster_to_xyz(dem, transform=dem_transform))

gp.compute_model(geo_model)

# %%
# Results
# ----------
gpv.plot_2d(geo_model, direction="y", show_data=True, show_topography=True)
gpv.plot_3d(geo_model)
