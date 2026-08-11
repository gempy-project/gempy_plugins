"""Rotate GIS-derived data to shrink GemPy's regular grid footprint, and crop data to
a bounding box -- vector layers and raw point arrays before conversion
(`crop_vector_to_bbox`/`crop_points_to_bbox`), rasters (`crop_raster_to_bbox`), or
already-converted `SurfacePointsTable`/`OrientationsTable` after it
(`crop_surface_points_to_bbox`/`crop_orientations_to_bbox`).
"""
from typing import Tuple

import numpy as np
from gempy_engine.core.data.transforms import Transform

from gempy.core.data.orientations import OrientationsTable
from gempy.core.data.surface_points import SurfacePointsTable
from gempy_plugins.optional_dependencies import require_geopandas, require_rasterio, require_shapely

BBox = Tuple[float, float, float, float, float, float]  # (minx, miny, maxx, maxy, minz, maxz)


def compute_alignment_rotation(xyz: np.ndarray) -> float:
    """Z-axis rotation (degrees) that aligns the XY footprint's minimum-area bounding
    rectangle with the X/Y axes.

    Rotating data by this angle lets a tight, axis-aligned model `extent` cover it,
    instead of the much larger box needed for diagonally-oriented data -- shrinking
    the volume GemPy's regular grid has to be computed over.
    """
    xy = np.asarray(xyz)[:, :2]
    if len(xy) < 3:
        return 0.0

    shapely = require_shapely()
    hull = shapely.MultiPoint(xy).convex_hull
    if hull.geom_type != "Polygon":
        return 0.0  # collinear points -- no well-defined minimum rectangle

    rect = shapely.oriented_envelope(hull)
    coords = np.asarray(rect.exterior.coords)
    edge = coords[1] - coords[0]
    angle = np.degrees(np.arctan2(edge[1], edge[0])) % 90
    if angle > 45:
        angle -= 90
    return -angle


def alignment_transform(xyz: np.ndarray) -> Transform:
    """A `Transform` that rotates `xyz` about its centroid to align its footprint
    with the X/Y axes -- see `compute_alignment_rotation`.

    Use `.apply()`/`.apply_inverse()` on the returned `Transform` to rotate data in
    and back out, or hand it straight to `geo_model.input_transform` to rotate inside
    GemPy itself instead of pre-rotating the input data.
    """
    xyz = np.asarray(xyz)
    pivot = xyz.mean(axis=0) if len(xyz) else np.zeros(3)
    angle = compute_alignment_rotation(xyz)
    return Transform(
        position=-pivot,
        rotation=np.array([0.0, 0.0, angle]),
        scale=np.ones(3),
    )


def align_to_minimum_extent(
        surface_points: SurfacePointsTable,
        orientations: OrientationsTable,
) -> Tuple[SurfacePointsTable, OrientationsTable, Transform]:
    """Rotate `surface_points` and `orientations` (points and gradient vectors both)
    to minimize their combined XY bounding box, per `alignment_transform`.

    Returns the rotated tables plus the `Transform` used, so callers can
    `.apply_inverse()` exported meshes/vertices back to true-world orientation later.
    """
    combined_xyz = np.concatenate([surface_points.xyz, orientations.xyz], axis=0)
    transform = alignment_transform(combined_xyz)

    rotated_sp = SurfacePointsTable(data=surface_points.data.copy(), name_id_map=surface_points.name_id_map)
    if len(rotated_sp) > 0:
        rotated_sp.xyz_view = transform.apply(surface_points.xyz)

    rotated_ori = OrientationsTable(data=orientations.data.copy(), name_id_map=orientations.name_id_map)
    if len(rotated_ori) > 0:
        rotated_ori.xyz_view = transform.apply(orientations.xyz)
        rotated_ori.grads_view = transform.transform_gradient(orientations.grads)

    return rotated_sp, rotated_ori, transform


def crop_vector_to_bbox(gdf, bbox: BBox):
    """Clip a GeoDataFrame's geometries to `bbox`'s XY footprint -- the Z bounds are
    ignored, since vector geometries don't carry a spatial Z axis `.clip()` can filter
    on (an elevation like a contour's `Z` is just an attribute column, not part of the
    geometry's own coordinate space here)."""
    require_geopandas()
    minx, miny, maxx, maxy, _minz, _maxz = bbox
    return gdf.clip((minx, miny, maxx, maxy))


def crop_points_to_bbox(xyz: np.ndarray, bbox: BBox) -> np.ndarray:
    """Filter an (n, 3) point array down to those within `bbox`'s XY footprint -- the Z
    bounds are ignored (e.g. for a topography point cloud: GemPy's own model `extent`
    already crops it to the right Z range when computing/plotting, so cropping it here
    too would just punch holes in the surface instead)."""
    minx, miny, maxx, maxy, _minz, _maxz = bbox
    x, y = xyz[:, 0], xyz[:, 1]
    mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    return xyz[mask]


def crop_surface_points_to_bbox(surface_points: SurfacePointsTable, bbox: BBox) -> SurfacePointsTable:
    """Filter `surface_points` down to those within `bbox = (minx, miny, maxx, maxy,
    minz, maxz)`, keeping `id`/`nugget` (and `name_id_map`) intact -- unlike
    `crop_points_to_bbox`, which only handles a bare xyz array and would drop that
    bookkeeping."""
    minx, miny, maxx, maxy, minz, maxz = bbox
    x, y, z = surface_points.data["X"], surface_points.data["Y"], surface_points.data["Z"]
    mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy) & (z >= minz) & (z <= maxz)
    return SurfacePointsTable(data=surface_points.data[mask].copy(), name_id_map=surface_points.name_id_map)


def crop_orientations_to_bbox(orientations: OrientationsTable, bbox: BBox) -> OrientationsTable:
    """Filter `orientations` down to those within `bbox = (minx, miny, maxx, maxy,
    minz, maxz)`, keeping `id`/`nugget`/gradients (and `name_id_map`) intact -- see
    `crop_surface_points_to_bbox`."""
    minx, miny, maxx, maxy, minz, maxz = bbox
    x, y, z = orientations.data["X"], orientations.data["Y"], orientations.data["Z"]
    mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy) & (z >= minz) & (z <= maxz)
    return OrientationsTable(data=orientations.data[mask].copy(), name_id_map=orientations.name_id_map)


def crop_raster_to_bbox(dem: np.ndarray, transform, bbox: BBox):
    """Slice a DEM array + its affine `transform` down to `bbox`'s XY footprint (Z
    bounds ignored -- a DEM's cell values *are* elevation, there's no separate Z axis
    to crop; see `crop_points_to_bbox`).

    Returns the cropped array and its updated transform.
    """
    rasterio = require_rasterio()
    minx, miny, maxx, maxy, _minz, _maxz = bbox

    row_a, col_a = rasterio.transform.rowcol(transform, minx, maxy)
    row_b, col_b = rasterio.transform.rowcol(transform, maxx, miny)
    row_start, row_stop = sorted((row_a, row_b))
    col_start, col_stop = sorted((col_a, col_b))

    row_start = max(row_start, 0)
    col_start = max(col_start, 0)
    row_stop = min(row_stop + 1, dem.shape[0])
    col_stop = min(col_stop + 1, dem.shape[1])

    cropped = dem[row_start:row_stop, col_start:col_stop]
    new_transform = transform * rasterio.Affine.translation(col_start, row_start)
    return cropped, new_transform
