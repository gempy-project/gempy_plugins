import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from gempy_plugins.gis_data.conversion import raster_to_xyz
from gempy_plugins.gis_data.interpolation import interpolate_raster


def _plane_contour(a: float, b: float, c: float, level: float, x0: float, x1: float) -> LineString:
    """A straight-line contour of the plane z = a*x + b*y + c at elevation `level`,
    spanning x in [x0, x1]."""
    y0 = (level - c - a * x0) / b
    y1 = (level - c - a * x1) / b
    return LineString([(x0, y0), (x1, y1)])


def test_interpolate_raster_recovers_linear_field():
    a, b, c = 0.5, -0.3, 10.0
    levels = [0.0, 5.0, 10.0, 15.0, 20.0]
    lines = [_plane_contour(a, b, c, level, 0.0, 10.0) for level in levels]
    gdf = gpd.GeoDataFrame({"elevation": levels}, geometry=lines)

    raster, transform = interpolate_raster(gdf, "elevation", resolution=(15, 15))

    xyz = raster_to_xyz(raster, transform)
    expected = a * xyz[:, 0] + b * xyz[:, 1] + c
    assert np.allclose(xyz[:, 2], expected, atol=1e-6)


def test_interpolate_raster_shape_matches_resolution():
    lines = [
        LineString([(0, 0), (10, 0)]),
        LineString([(0, 10), (10, 10)]),
    ]
    gdf = gpd.GeoDataFrame({"z": [0.0, 10.0]}, geometry=lines)

    raster, _ = interpolate_raster(gdf, "z", resolution=(8, 4))

    assert raster.shape == (4, 8)


def test_interpolate_raster_defaults_bbox_to_gdf_bounds():
    lines = [
        LineString([(0, 0), (10, 0)]),
        LineString([(0, 10), (10, 10)]),
    ]
    gdf = gpd.GeoDataFrame({"z": [0.0, 10.0]}, geometry=lines)

    _, transform = interpolate_raster(gdf, "z", resolution=(10, 10))

    minx, maxy = transform * (0, 0)
    maxx, miny = transform * (10, 10)
    assert np.isclose(minx, 0.0)
    assert np.isclose(maxx, 10.0)
    assert np.isclose(miny, 0.0)
    assert np.isclose(maxy, 10.0)
