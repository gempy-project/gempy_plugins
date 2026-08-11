"""Interpolate a scattered vector layer (e.g. elevation-tagged topographic contour
lines) into a regular raster grid.
"""
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator

from gempy_plugins.gis_data.conversion import _geometry_coords
from gempy_plugins.optional_dependencies import require_rasterio

Bbox2D = Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


def interpolate_raster(
        gdf, value_column: str, resolution: Tuple[int, int], bbox: Optional[Bbox2D] = None,
        kernel: str = "thin_plate_spline",
) -> Tuple[np.ndarray, object]:
    """Interpolate a vector layer's per-feature scalar (e.g. one elevation per
    topographic contour line) into a regular raster grid via radial basis function
    interpolation.

    Reimplements gemgis's `interpolate_raster` (built on the legacy
    `scipy.interpolate.Rbf`) on the modern `scipy.interpolate.RBFInterpolator`. Every
    vertex of a feature is used as one interpolation sample carrying that feature's
    `value_column` value -- e.g. every vertex along one contour line contributes a
    sample at that contour's elevation.

    Args:
        gdf: A GeoDataFrame of Point/LineString/Polygon geometries.
        value_column: Column in `gdf` holding each feature's scalar value.
        resolution: (width, height) of the output raster, in cells.
        bbox: Optional (minx, miny, maxx, maxy) to interpolate over; defaults to
            `gdf`'s own total bounds.
        kernel: RBF kernel, passed straight to `RBFInterpolator`.

    Returns:
        raster: (height, width) array of interpolated values.
        transform: The output raster's `affine.Affine` transform, so the result
            composes with `raster_to_xyz`/`extract_xyz`/`orientations_from_dem`.
    """
    rasterio = require_rasterio()

    xy, values = _vertex_xy_values(gdf, value_column)

    if bbox is None:
        minx, miny = xy.min(axis=0)
        maxx, maxy = xy.max(axis=0)
    else:
        minx, miny, maxx, maxy = bbox

    width, height = resolution
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    query_x, query_y = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    query_xy = np.column_stack([query_x, query_y])

    interpolator = RBFInterpolator(xy, values, kernel=kernel)
    raster = interpolator(query_xy).reshape(height, width)

    return raster, transform


def _vertex_xy_values(gdf, value_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """Explode every geometry in `gdf` into vertex XY coordinates, each carrying its
    source feature's `value_column` value."""
    xy, values = [], []
    for value, geom in zip(gdf[value_column].to_numpy(), gdf.geometry):
        coords, _ = _geometry_coords(geom)
        for coord in coords:
            xy.append((coord[0], coord[1]))
            values.append(value)
    return np.asarray(xy, dtype=float), np.asarray(values, dtype=float)
