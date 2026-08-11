"""
Aligning and Cropping Rotated GIS Data
============================================

A faulted two-formation model built from vector layers digitized at an arbitrary map
orientation, showcasing `alignment.py`'s rotation and bounding-box crop helpers.
Geology from cgre-aachen/gemgis's `"Faulted Layers"
<https://github.com/cgre-aachen/gemgis_data/blob/main/notebooks/04_faulted_layers/example02_faulted_layers.ipynb>`_
dataset, rotated 35 degrees for this example (see `examples/data_03/README.rst`).
"""

# %%
# Real digitized data is rarely conveniently axis-aligned -- a survey grid, a scanned
# map's own orientation, whatever the digitizer happened to use. `align_to_minimum_extent`
# rotates data back so GemPy's regular grid doesn't have to cover a much bigger,
# diagonally-oriented box than the data actually needs. Once it's axis-aligned,
# `crop_surface_points_to_bbox`/`crop_orientations_to_bbox`/`crop_points_to_bbox` trim
# surface points, orientations, and topography down to one shared bounding box -- e.g.
# to pull out a smaller sub-model from a larger dataset.

# %%
import os

import gempy as gp
import gempy_viewer as gpv

import gempy_plugins.gis_data as _gis_data_pkg
from gempy_plugins.gis_data.alignment import (
    align_to_minimum_extent, crop_orientations_to_bbox, crop_points_to_bbox, crop_surface_points_to_bbox,
)
from gempy_plugins.gis_data.conversion import (
    orientations_from_strike_lines, raster_to_xyz, relabel_orientations, surface_points_from_geodataframe,
)
from gempy_plugins.gis_data.interpolation import interpolate_raster
from gempy_plugins.gis_data.plotting import plot_gis_data_3d, plot_input_layers
from gempy_plugins.optional_dependencies import require_geopandas

gpd = require_geopandas()

# %%
# Input data
# --------------
# `interfaces.shp` (formations `A`, `B`, `F` -- `F` the fault), `topo.shp`,
# `strikes.shp` -- all rotated 35 degrees about their shared centroid, so the
# footprint below is visibly diagonal.
_DATA = os.path.join(os.path.dirname(_gis_data_pkg.__file__), "examples", "data_03")

interfaces = gpd.read_file(os.path.join(_DATA, "interfaces.shp"))
topo_contours = gpd.read_file(os.path.join(_DATA, "topo.shp"))
strikes = gpd.read_file(os.path.join(_DATA, "strikes.shp"))

plot_input_layers(interfaces=interfaces, topo_contours=topo_contours, strike_lines=strikes)

# %%
# Convert to GemPy input
# -------------------------
# Same conversion functions as `example01.py`/`example02.py`, condensed: a DEM from
# the contours, surface points draped onto it, and orientations from strike-line
# clusters (`strikes.shp` mixes real formation names with a couple of extra
# location clusters, `"A1"`/`"B1"` -- see `example02.py` for why that has to be
# relabeled by hand rather than derived automatically).
minx, miny, maxx, maxy = topo_contours.total_bounds
dem, dem_transform = interpolate_raster(topo_contours, value_column="Z", resolution=(150, 150), bbox=(minx, miny, maxx, maxy))

surface_points = surface_points_from_geodataframe(
    interfaces, formation_column="formation", dem=dem, dem_transform=dem_transform,
)
raw_orientations = orientations_from_strike_lines(strikes, formation_column="formation", elevation_column="Z")
orientations = relabel_orientations(
    raw_orientations, rename={"A": "A", "A1": "A", "B": "B", "B1": "B", "F": "F"},
    name_id_map=surface_points.name_id_map,
)

# %%
# Aligning to the minimum extent
# -----------------------------------
# `align_to_minimum_extent` rotates `surface_points`/`orientations` (points and
# gradient vectors both) about their combined centroid to align their footprint with
# the X/Y axes, and returns the `Transform` used.
surface_points, orientations, alignment = align_to_minimum_extent(surface_points, orientations)

# Cropping to an axis-aligned box *before* aligning would clip the data's extremal
# points onto that box's own straight edges, erasing the diagonal footprint
# `align_to_minimum_extent` needs to detect in the first place (confirmed -- cropping
# first drops the detected rotation to a flat 0 degrees). Align on the full data
# first, crop what comes out of it.

# The alignment `Transform` composes with any other data derived from the same
# original space -- applying it to the flattened topography point cloud keeps it
# consistent with the now-realigned `surface_points`/`orientations`.
topography_xyz = alignment.apply(raster_to_xyz(dem, transform=dem_transform))

# %%
# Cropping everything to one bounding box
# ---------------------------------------------
# A sub-region of the now axis-aligned data -- e.g. to pull a smaller sub-model out of
# a larger survey. `bbox = (minx, miny, maxx, maxy, minz, maxz)`: any values work, this
# one just happens to sit well inside the aligned footprint. The same `bbox` crops
# `surface_points`/`orientations` in all three dimensions; `topography_xyz`'s crop
# ignores the Z bounds (`crop_points_to_bbox` always does -- GemPy's own model extent
# already crops topography to the right Z range when computing/plotting, so cropping
# it here too would just punch holes in the surface instead).
bbox = (-1100, -1000, 1100, 1200, -250, 200)

# Preview what the crop keeps/cuts before committing to it.
plot_gis_data_3d(surface_points=surface_points, orientations=orientations, topography_xyz=topography_xyz, bbox=bbox)

# %%
surface_points = crop_surface_points_to_bbox(surface_points, bbox)
orientations = crop_orientations_to_bbox(orientations, bbox)
topography_xyz = crop_points_to_bbox(topography_xyz, bbox)

# %%
# Build and compute the model
# -------------------------------
# The crop's own `bbox` doubles as the model extent. `Fault1`/`F` and `Strata1`/`B`,`A`
# reproduce the source dataset's real structure (see `example02.py` for the full
# explanation of series/fault mapping).
minx, miny, maxx, maxy, minz, maxz = bbox
geo_model = gp.create_geomodel(
    project_name="gis_data_example03",
    extent=[minx, maxx, miny, maxy, minz, maxz],
    resolution=[50, 50, 50],
    structural_frame=gp.data.StructuralFrame.from_data_tables(surface_points, orientations),
)
gp.map_stack_to_surfaces(geo_model, {"Fault1": ["F"], "Strata1": ["B", "A"]})
gp.set_is_fault(geo_model, ["Fault1"])

gp.set_topography_from_arrays(geo_model.grid, topography_xyz)

gp.compute_model(geo_model)

# %%
# Results
# ----------
gpv.plot_2d(geo_model, direction="y", show_data=True, show_topography=True)
gpv.plot_3d(geo_model)
