"""
Unconformable Faulted Layers from GIS Data
================================================

A fault plus two unconformity-bound series (five formations total), built from
digitized vector layers. Reproduces cgre-aachen/gemgis's `"Example 2 -- Unconformable
Faulted Layers" <https://github.com/cgre-aachen/gemgis_data/blob/main/notebooks/06_combined_models/example02_unconformable_faulted_layers.ipynb>`_
dataset.
"""

# %%
# Strike lines are digitized as several location clusters per formation (e.g. `B`'s
# dip is measured at four map locations, `"B1"`..`"B4"`) rather than one row per
# formation. `orientations_from_strike_lines` groups by whatever column it's given, so
# one call handles every cluster's own pairing; `relabel_orientations` then folds the
# per-cluster results into their real formations by name.

# %%
import os

import gempy as gp
import gempy_viewer as gpv

import gempy_plugins.gis_data as _gis_data_pkg
from gempy_plugins.gis_data.conversion import (
    orientations_from_strike_lines, raster_to_xyz, relabel_orientations, surface_points_from_geodataframe,
    unique_names,
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
# 7 formations (`Fault`, `B`, `C`, `D`, `F`, `G`, `H`) in `interfaces.shp`, elevation
# contours in `topo.shp`, strike-line clusters in `strikes.shp` -- vendored from
# `cgre-aachen/gemgis_data <https://github.com/cgre-aachen/gemgis_data/tree/main/data/example16_all_features>`_
# (CC BY 4.0, see `examples/data_02/README.rst`). The DEM is interpolated from
# `topo.shp` directly, no pre-made raster needed.
_DATA = os.path.join(os.path.dirname(_gis_data_pkg.__file__), "examples", "data_02")

interfaces = gpd.read_file(os.path.join(_DATA, "interfaces.shp"))
topo_contours = gpd.read_file(os.path.join(_DATA, "topo.shp"))
strikes = gpd.read_file(os.path.join(_DATA, "strikes.shp"))

# %%
# All three raw layers together: contours (brown, elevation-labeled), interfaces (one
# color per formation), strike lines (black, still labeled by raw cluster id).
plot_input_layers(interfaces=interfaces, topo_contours=topo_contours, strike_lines=strikes)

# %%
# Building a DEM from the topographic contours
# ------------------------------------------------------
# `interpolate_raster` fits a radial basis function through every contour vertex
# (each one a sample point) to build a full elevation grid. `bbox` fixes the DEM's
# real-world extent; `resolution` its cell count -- together they set cell size.
_minx, _miny, _maxx, _maxy = topo_contours.total_bounds
dem, dem_transform = interpolate_raster(
    topo_contours, value_column="Z", resolution=(200, 140), bbox=(_minx, _miny, _maxx, _maxy),
)

# %%
# DEM, top-down and as a 3D relief surface.
plot_raster_2d(dem, dem_transform)
plot_raster_3d(dem, dem_transform)

# %%
# Convert to GemPy input
# -------------------------
# Interface points come from draping digitized contact lines onto the DEM;
# `interfaces.shp`'s `formation` column already holds real formation names.
surface_points = surface_points_from_geodataframe(
    interfaces, formation_column="formation", dem=dem, dem_transform=dem_transform,
)
print(surface_points_to_dataframe(surface_points))  # pretty-print as a GemPy-format table
# export_surface_points_csv(surface_points, "surface_points.csv")  # write to GemPy's own CSV format

# %%
# Deriving orientations from strike-line clusters
# ----------------------------------------------------
# `strikes.shp` carries `Z` directly, but `formation` holds a cluster id
# (`"B1"`..`"B4"`, `"C1"`..`"C5"`, ...), not the final formation name -- pairing
# within one cluster gives a real, local dip; pairing across clusters would combine
# unrelated, distant measurements into a meaningless one.
raw_orientations = orientations_from_strike_lines(strikes, formation_column="formation", elevation_column="Z")

# %%
# The cluster -> formation mapping depends entirely on how this specific GIS layer was
# digitized and named, so it can't be derived automatically -- it has to be built by
# hand, matching each raw cluster label to the real formation names used in the GemPy
# model (`surface_points`' own formations). Inspect the raw labels actually present...
print(unique_names(raw_orientations))
# ...and map each one to its real formation:
rename = {
    "Fault": "Fault", "Fault1": "Fault", "Fault2": "Fault", "Fault3": "Fault",
    "B"    : "B", "B1": "B", "B2": "B", "B3": "B", "B4": "B",
    "C"    : "C", "C1": "C", "C2": "C", "C3": "C", "C4": "C", "C5": "C",
    "D"    : "D", "D1": "D", "D2": "D", "D3": "D", "D4": "D", "D5": "D",
    "G"    : "G",
}

# `on_unmapped="raise"` (default) stops on any label `rename` misses -- no silent data
# loss. Pass `on_unmapped="drop"` to discard unmapped labels deliberately instead.
orientations = relabel_orientations(raw_orientations, rename=rename, name_id_map=surface_points.name_id_map)
print(orientations_to_dataframe(orientations))  # pretty-print as a GemPy-format table
# export_orientations_csv(orientations, "orientations.csv")  # write to GemPy's own CSV format

# %%
# Build and compute the model
# -------------------------------
# One default group from `surface_points`/`orientations` first, then
# `map_stack_to_surfaces` reorganizes elements into the real structure: `Fault` alone
# on its own fault series, `F`/`G`/`H` and `B`/`C`/`D` as the two unconformity-bound
# series.
geo_model = gp.create_geomodel(
    project_name="gis_data_example02",
    extent=[0, 3968, 0, 2731, 0, 1000],
    resolution=[50, 50, 50],
    structural_frame=gp.data.StructuralFrame.from_data_tables(surface_points, orientations),
)
gp.map_stack_to_surfaces(
    geo_model,
    {
        "Strata1": ["F", "G", "H"],
        "Fault1" : ["Fault"],
        "Strata2": ["B", "C", "D"],
    },
)
gp.set_is_fault(geo_model, ["Fault1"])

gp.set_topography_from_arrays(geo_model.grid, raster_to_xyz(dem, transform=dem_transform))

# %%
# Input preview
# ----------------
gpv.plot_3d(geo_model)

# %%
gp.compute_model(geo_model)

# %%
# Results
# ----------
gpv.plot_2d(geo_model, direction="y", show_data=True, show_topography=True)
gpv.plot_3d(geo_model)
