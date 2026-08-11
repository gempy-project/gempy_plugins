"""Vector/raster GIS data -> GemPy `SurfacePointsTable`/`OrientationsTable` conversion.

Consolidates what used to be ~9 separate `extract_xyz_*` functions in gemgis
(cgre-aachen/gemgis) into one geometry-type-dispatching function, and reuses GemPy's
own `convert_orientation_to_pole_vector` for the dip/azimuth -> gradient conversion
instead of reimplementing that math a second time.
"""
import warnings
from typing import Optional, Tuple

import numpy as np

import gempy as gp
from gempy.modules.data_manipulation.manipulate_points import convert_orientation_to_pole_vector

from gempy_plugins.optional_dependencies import require_rasterio


def extract_xyz(gdf, dem=None, dem_transform=None) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-vertex XYZ coordinates from a GeoDataFrame's geometries.

    Handles Point/LineString/Polygon (and their Multi- variants). Z comes from the
    geometry itself where present (`geom.has_z`), otherwise from `dem` sampled at each
    vertex's XY -- either a rasterio dataset (has a `.sample()` method), or a plain
    `np.ndarray` + `dem_transform` (an `affine.Affine`).

    Args:
        gdf: A GeoDataFrame with Point/LineString/Polygon geometries.
        dem: Optional elevation source used for vertices that lack a Z coordinate.
        dem_transform: Required alongside `dem` when it is a plain `np.ndarray`.

    Returns:
        xyz: (n, 3) array of extracted coordinates, one row per vertex.
        source_index: (n,) array mapping each row back to its source `gdf` row --
            use it to broadcast other columns (e.g. formation names) onto the
            exploded per-vertex output.
    """
    xs, ys, zs, source_index = [], [], [], []
    for row_idx, geom in enumerate(gdf.geometry):
        coords, has_z = _geometry_coords(geom)
        for coord in coords:
            xs.append(coord[0])
            ys.append(coord[1])
            zs.append(coord[2] if has_z else np.nan)
            source_index.append(row_idx)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    source_index = np.asarray(source_index, dtype=int)

    missing_z = np.isnan(zs)
    if missing_z.any():
        if dem is None:
            raise ValueError("Some geometries have no Z coordinate and no `dem` was given to sample it from.")
        zs[missing_z] = _sample_dem(xs[missing_z], ys[missing_z], dem, dem_transform)

    return np.column_stack([xs, ys, zs]), source_index


def _geometry_coords(geom):
    """Return (list of vertex coordinate tuples, has_z) for one shapely geometry,
    recursing into Multi-geometries."""
    geom_type = geom.geom_type
    if geom_type == "Point":
        return [tuple(geom.coords)[0]], geom.has_z
    if geom_type in ("LineString", "LinearRing"):
        return list(geom.coords), geom.has_z
    if geom_type == "Polygon":
        return list(geom.exterior.coords), geom.has_z
    if geom_type.startswith("Multi") or geom_type == "GeometryCollection":
        coords, has_z = [], False
        for part in geom.geoms:
            part_coords, part_has_z = _geometry_coords(part)
            coords.extend(part_coords)
            has_z = has_z or part_has_z
        return coords, has_z
    raise ValueError(f"Unsupported geometry type: {geom_type}")


def _vertex_xy(gdf) -> np.ndarray:
    """Explode every geometry in `gdf` into an (n, 2) array of vertex XY coordinates,
    ignoring Z."""
    xy = []
    for geom in gdf.geometry:
        coords, _ = _geometry_coords(geom)
        xy.extend((coord[0], coord[1]) for coord in coords)
    return np.asarray(xy, dtype=float)


def _sample_dem(xs: np.ndarray, ys: np.ndarray, dem, dem_transform) -> np.ndarray:
    if hasattr(dem, "sample"):
        return np.array([value[0] for value in dem.sample(zip(xs, ys))], dtype=float)

    if dem_transform is None:
        raise ValueError("`dem_transform` is required when `dem` is a plain array.")

    rasterio = require_rasterio()
    rows, cols = rasterio.transform.rowcol(dem_transform, xs, ys)
    rows = np.clip(np.asarray(rows), 0, dem.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, dem.shape[1] - 1)
    return dem[rows, cols].astype(float)


def raster_to_xyz(dem, transform=None) -> np.ndarray:
    """Flatten a DEM raster into an (n, 3) array of every cell's XYZ coordinates --
    e.g. to add it to a GemPy model as topography via
    ``gp.set_topography_from_arrays(geo_model.grid, raster_to_xyz(dem))``, since
    GemPy's own `set_topography_from_gdal`/`set_topography_from_array` helpers raise
    `NotImplementedError`.

    Args:
        dem: A rasterio dataset (has a `.read()` method), or a plain `np.ndarray`
            (`transform` is then required).
        transform: The DEM's `affine.Affine` transform, required when `dem` is a
            plain array.
    """
    if hasattr(dem, "read"):
        transform = dem.transform
        dem = dem.read(1)

    if transform is None:
        raise ValueError("`transform` is required when `dem` is a plain array.")

    rasterio = require_rasterio()
    rows, cols = np.meshgrid(np.arange(dem.shape[0]), np.arange(dem.shape[1]), indexing="ij")
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    return np.column_stack([np.asarray(xs), np.asarray(ys), dem.ravel()])


def surface_points_from_geodataframe(
        gdf, formation_column: str, dem=None, dem_transform=None, nugget: Optional[np.ndarray] = None,
) -> gp.data.SurfacePointsTable:
    """Build a `SurfacePointsTable` from a vector layer (contact points, contact
    lines, or formation-boundary polygons), pulling Z from the geometry or a DEM.

    Args:
        gdf: A GeoDataFrame with one row per feature; `formation_column` gives each
            feature's formation name.
        formation_column: Column in `gdf` holding formation names.
        dem: Optional elevation source used for geometries without Z, see `extract_xyz`.
        dem_transform: Required alongside `dem` when it is a plain `np.ndarray`.
        nugget: Optional per-point nugget values (broadcast to match the exploded
            per-vertex output length if scalar).
    """
    xyz, source_index = extract_xyz(gdf, dem=dem, dem_transform=dem_transform)
    names = gdf[formation_column].to_numpy()[source_index]
    return gp.data.SurfacePointsTable.from_arrays(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], names=names, nugget=nugget,
    )


def orientations_from_dem(
        dem: np.ndarray, transform, formation: str,
        sample_at=None, every_n: Optional[int] = None, nugget: Optional[np.ndarray] = None,
) -> gp.data.OrientationsTable:
    """Sample dip/azimuth orientations from a DEM's slope and convert them to GemPy's
    gradient-vector orientation format.

    Args:
        dem: Elevation array.
        transform: The DEM's `affine.Affine` transform.
        formation: Formation name assigned to every sampled orientation.
        sample_at: Optional GeoDataFrame whose vertices give explicit sample
            locations (e.g. a fault-trace line).
        every_n: Optional cell stride for sampling a regular subgrid of `dem`
            directly, used when `sample_at` isn't given.
        nugget: Optional per-orientation nugget values.
    """
    if sample_at is None and every_n is None:
        raise ValueError("Provide either `sample_at` or `every_n` to choose sample locations.")

    rasterio = require_rasterio()

    dzdrow, dzdcol = np.gradient(dem)
    dzdx = dzdcol / transform.a
    dzdy = dzdrow / transform.e

    if sample_at is not None:
        xy = _vertex_xy(sample_at)
        xs, ys = xy[:, 0], xy[:, 1]
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)
        rows = np.clip(np.asarray(rows), 0, dem.shape[0] - 1)
        cols = np.clip(np.asarray(cols), 0, dem.shape[1] - 1)
    else:
        rows_grid, cols_grid = np.meshgrid(
            np.arange(0, dem.shape[0], every_n),
            np.arange(0, dem.shape[1], every_n),
            indexing="ij",
        )
        rows, cols = rows_grid.ravel(), cols_grid.ravel()
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        xs, ys = np.asarray(xs), np.asarray(ys)

    zs = dem[rows, cols]
    sample_dzdx = dzdx[rows, cols]
    sample_dzdy = dzdy[rows, cols]

    dip = np.degrees(np.arctan(np.hypot(sample_dzdx, sample_dzdy)))
    azimuth = np.degrees(np.arctan2(-sample_dzdx, -sample_dzdy)) % 360
    polarity = np.ones_like(dip)

    gradients = np.asarray(convert_orientation_to_pole_vector(azimuth, dip, polarity))

    return gp.data.OrientationsTable.from_arrays(
        x=xs, y=ys, z=zs,
        G_x=gradients[:, 0], G_y=gradients[:, 1], G_z=gradients[:, 2],
        names=np.full(len(xs), formation), nugget=nugget,
    )


def nearest_contour_elevation(gdf, topo_contours, elevation_column: str = "Z") -> np.ndarray:
    """Look up each of `gdf`'s geometries' elevation from its nearest feature in
    `topo_contours` -- e.g. a strike-line layer with no elevation of its own that, by
    construction, sits exactly on one topographic contour (distance 0).

    Args:
        gdf: Geometries to look up an elevation for, e.g. strike lines.
        topo_contours: Elevation-tagged reference geometries to search for the nearest
            match, e.g. topographic contours.
        elevation_column: Column in `topo_contours` holding each feature's elevation.

    Returns:
        One elevation per row of `gdf`, in the same order.
    """
    return np.array([
        topo_contours.loc[topo_contours.geometry.distance(geom).idxmin(), elevation_column]
        for geom in gdf.geometry
    ], dtype=float)


def orientations_from_strike_lines(
        gdf, formation_column: str, elevation_column: str,
        nugget: Optional[np.ndarray] = None, name_id_map: Optional[dict] = None,
) -> gp.data.OrientationsTable:
    """Derive orientations from strike lines -- lines connecting a geological
    contact's intersections with topographic contours of equal elevation -- the
    classical hand method for reading dip/azimuth off a digitized geological map.

    For each formation, strike lines are sorted by `elevation_column` and every
    consecutive pair yields one orientation: dip from
    ``arctan(elevation difference / horizontal distance between the two lines)``
    (distance = shapely's minimum distance between the two geometries), and azimuth
    from the upward-oriented normal of a plane fit through both lines' vertices
    together -- same normal convention as `orientations_from_surface_points`.

    Reimplements gemgis's `calculate_orientations_from_strike_lines`, generalized to
    LineStrings with any number of vertices (gemgis required exactly two: start and
    end point).

    Args:
        gdf: GeoDataFrame of LineString geometries, one row per strike line.
        formation_column: Column in `gdf` giving each strike line's formation.
        elevation_column: Column in `gdf` giving each strike line's elevation.
        nugget: Optional per-orientation nugget values.
        name_id_map: Optional explicit name-to-id mapping, e.g. a
            `SurfacePointsTable.name_id_map` this result needs to line up with (ids
            are otherwise generated fresh from just the names seen here, which only
            happens to match another table's ids for reasons -- like alphabetical
            order -- that don't hold in general).

    Returns:
        One orientation per consecutive elevation pair of strike lines, per
        formation. Formations with fewer than 2 strike lines contribute none.
    """
    xs, ys, zs, dips, azimuths, names = [], [], [], [], [], []

    for formation, group in gdf.groupby(formation_column):
        group = group.sort_values(elevation_column)
        lines = list(group.geometry)
        elevations = group[elevation_column].to_numpy(dtype=float)

        for i in range(len(lines) - 1):
            line_a, line_b = lines[i], lines[i + 1]
            distance = line_a.distance(line_b)
            if distance == 0:
                continue

            dip = np.degrees(np.arctan(np.abs(elevations[i + 1] - elevations[i]) / distance))

            points = np.array(
                [(coord[0], coord[1], elevations[i]) for coord in line_a.coords]
                + [(coord[0], coord[1], elevations[i + 1]) for coord in line_b.coords]
            )
            centroid = points.mean(axis=0)
            _, _, vt = np.linalg.svd(points - centroid)
            normal = vt[2]
            if normal[2] < 0:
                normal = -normal
            azimuth = np.degrees(np.arctan2(normal[0], normal[1])) % 360

            midpoint_a = line_a.interpolate(0.5, normalized=True)
            midpoint_b = line_b.interpolate(0.5, normalized=True)

            xs.append((midpoint_a.x + midpoint_b.x) / 2)
            ys.append((midpoint_a.y + midpoint_b.y) / 2)
            zs.append((elevations[i] + elevations[i + 1]) / 2)
            dips.append(dip)
            azimuths.append(azimuth)
            names.append(formation)

    dips = np.asarray(dips)
    azimuths = np.asarray(azimuths)
    polarity = np.ones_like(dips)
    gradients = np.asarray(convert_orientation_to_pole_vector(azimuths, dips, polarity))

    return gp.data.OrientationsTable.from_arrays(
        x=np.asarray(xs), y=np.asarray(ys), z=np.asarray(zs),
        G_x=gradients[:, 0], G_y=gradients[:, 1], G_z=gradients[:, 2],
        names=np.asarray(names), nugget=nugget, name_id_map=name_id_map,
    )


def unique_names(table) -> list:
    """Every distinct name in `table.name_id_map`, as plain `str` (not `numpy.str_`)
    and sorted -- e.g. to inspect the raw labels actually present in some GIS-derived
    data before working out a `relabel_orientations` `rename` mapping by hand.

    Args:
        table: A `SurfacePointsTable` or `OrientationsTable`.
    """
    return sorted(str(name) for name in table.name_id_map)


def relabel_orientations(
        orientations: gp.data.OrientationsTable, rename: dict, name_id_map: Optional[dict] = None,
        on_unmapped: str = "raise",
) -> gp.data.OrientationsTable:
    """Reassign every orientation's label via `rename` (current name -> real target
    name), and rebuild the table's ids against `name_id_map`.

    For strike-line data digitized as several separate clusters per formation (e.g.
    `"B1"`, `"B2"`, `"B3"`, `"B4"` all really being formation `"B"`, each cluster
    just a different map location where a dip pair happened to be measured),
    `orientations_from_strike_lines` still needs each cluster kept separate -- pairing
    lines *across* clusters by elevation would combine unrelated, spatially distant
    measurements into a meaningless dip. Grouping by the raw cluster column first and
    relabeling the result afterward keeps both correct: one call handles every
    cluster's own internal pairing, and this function folds the results into their
    real, shared formations -- and real, shared ids, via `name_id_map` (see
    `orientations_from_strike_lines`'s own note on why ids don't just match up
    otherwise).

    Args:
        orientations: Table to relabel, e.g. one `orientations_from_strike_lines` call
            grouped by a convenient column that isn't the final formation name.
        rename: Maps every label currently in `orientations` to its real target name.
            There's no reliable way to derive this automatically -- it depends
            entirely on whatever naming convention the source data happens to use
            (and a convention that looks obvious, like a trailing cluster number, can
            just as easily be a real, meaningful part of a formation's own name in a
            different dataset). Work it out by inspecting the actual raw labels (e.g.
            `sorted(orientations.name_id_map)`) against the real target formations.
        name_id_map: Id map the renamed result should use, e.g. a
            `SurfacePointsTable.name_id_map` this result needs to line up with.
            Defaults to generating fresh ids from the renamed names.
        on_unmapped: What to do about raw labels present in `orientations` that
            `rename` doesn't cover. `"raise"` (default) stops with a `ValueError`
            naming them -- a missing or misspelled entry never silently drops data.
            `"drop"` discards those rows instead (after a `warnings.warn` naming
            them), for when leaving some labels out is a deliberate choice.

    Returns:
        A new OrientationsTable with the covered rows, renamed and re-keyed.
    """
    if on_unmapped not in ("raise", "drop"):
        raise ValueError(f"`on_unmapped` must be 'raise' or 'drop', got {on_unmapped!r}.")

    id_to_name = {id_: name for name, id_ in orientations.name_id_map.items()}
    old_names = np.array([id_to_name[id_] for id_ in orientations.data["id"]])

    unmapped = sorted(set(old_names) - set(rename))
    if unmapped:
        if on_unmapped == "raise":
            raise ValueError(
                f"`rename` doesn't cover these labels present in `orientations`: {unmapped}. "
                f"Add them to `rename`, or pass on_unmapped='drop' to discard them deliberately."
            )
        warnings.warn(f"relabel_orientations: dropping orientations with unmapped labels: {unmapped}")

    keep = np.isin(old_names, list(rename))
    new_names = np.array([rename[name] for name in old_names[keep]])

    xyz = orientations.xyz[keep]
    grads = orientations.grads[keep]
    return gp.data.OrientationsTable.from_arrays(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        G_x=grads[:, 0], G_y=grads[:, 1], G_z=grads[:, 2],
        names=new_names, nugget=orientations.data["nugget"][keep], name_id_map=name_id_map,
    )


def orientations_from_surface_points(
        surface_points: gp.data.SurfacePointsTable, nugget: Optional[np.ndarray] = None,
) -> gp.data.OrientationsTable:
    """Derive one orientation per formation by fitting a plane through that
    formation's own surface points -- e.g. an interface line draped onto a DEM
    already carries enough 3D shape to recover the contact's dip/azimuth directly,
    with no separate orientation dataset needed.

    Formations with fewer than 3 points are skipped (a plane isn't well-defined) and
    simply get no orientation in the result.

    Args:
        surface_points: Surface points to fit, one plane per distinct formation id.
        nugget: Optional per-orientation nugget values.
    """
    id_to_name = {id_: name for name, id_ in surface_points.name_id_map.items()}

    xs, ys, zs, gxs, gys, gzs, names = [], [], [], [], [], [], []
    for group in surface_points.get_surface_points_by_id_groups():
        if len(group) < 3:
            continue

        xyz = group.xyz
        centroid = xyz.mean(axis=0)
        _, _, vt = np.linalg.svd(xyz - centroid)
        normal = vt[2]
        if normal[2] < 0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)

        xs.append(centroid[0])
        ys.append(centroid[1])
        zs.append(centroid[2])
        gxs.append(normal[0])
        gys.append(normal[1])
        gzs.append(normal[2])
        names.append(id_to_name[group.id])

    return gp.data.OrientationsTable.from_arrays(
        x=np.asarray(xs), y=np.asarray(ys), z=np.asarray(zs),
        G_x=np.asarray(gxs), G_y=np.asarray(gys), G_z=np.asarray(gzs),
        names=np.asarray(names), nugget=nugget,
        name_id_map=surface_points.name_id_map,
    )
