import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, Point

from gempy.core.data.surface_points import SurfacePointsTable

from gempy_plugins.gis_data.conversion import (
    extract_xyz, orientations_from_dem, orientations_from_strike_lines, orientations_from_surface_points,
    raster_to_xyz, surface_points_from_geodataframe,
)


def test_extract_xyz_from_point_z_geometries():
    gdf = gpd.GeoDataFrame({"formation": ["a", "b"]}, geometry=[Point(0, 0, 10), Point(1, 1, 20)])

    xyz, source_index = extract_xyz(gdf)

    assert np.allclose(xyz, [[0, 0, 10], [1, 1, 20]])
    assert list(source_index) == [0, 1]


def test_extract_xyz_samples_dem_for_missing_z():
    dem = np.array([[1.0, 2.0], [3.0, 4.0]])
    transform = Affine(1, 0, 0, 0, -1, 2)
    gdf = gpd.GeoDataFrame({"formation": ["a"]}, geometry=[LineString([(0.5, 1.5), (1.5, 0.5)])])

    xyz, source_index = extract_xyz(gdf, dem=dem, dem_transform=transform)

    assert np.allclose(xyz[:, :2], [[0.5, 1.5], [1.5, 0.5]])
    assert np.allclose(xyz[:, 2], [1.0, 4.0])
    assert list(source_index) == [0, 0]


def test_surface_points_from_geodataframe():
    gdf = gpd.GeoDataFrame(
        {"formation": ["a", "b", "a"]},
        geometry=[Point(0, 0, 10), Point(1, 1, 20), Point(2, 2, 30)],
    )

    surface_points = surface_points_from_geodataframe(gdf, formation_column="formation")

    assert len(surface_points) == 3
    assert np.allclose(surface_points.xyz, [[0, 0, 10], [1, 1, 20], [2, 2, 30]])
    assert set(surface_points.name_id_map.keys()) == {"a", "b"}


def test_orientations_from_dem_recovers_plane_gradient():
    a, b = 2.0, 3.0
    transform = Affine(1, 0, 0, 0, -1, 10)
    rows, cols = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
    xs = cols.astype(float)
    ys = 10.0 - rows.astype(float)
    dem = a * xs + b * ys

    orientations = orientations_from_dem(dem, transform, formation="topo", every_n=2)

    expected = np.array([-a, -b, 1.0])
    expected /= np.linalg.norm(expected)

    assert len(orientations) > 0
    assert np.allclose(orientations.grads, np.tile(expected, (len(orientations), 1)), atol=1e-6)
    assert set(orientations.name_id_map.keys()) == {"topo"}


def test_orientations_from_dem_sample_at_points():
    a, b = 1.0, -2.0
    transform = Affine(1, 0, 0, 0, -1, 10)
    rows, cols = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
    xs = cols.astype(float)
    ys = 10.0 - rows.astype(float)
    dem = a * xs + b * ys

    sample_points = gpd.GeoDataFrame(geometry=[Point(2.5, 3.5), Point(6.5, 1.5)])
    orientations = orientations_from_dem(dem, transform, formation="topo", sample_at=sample_points)

    expected = np.array([-a, -b, 1.0])
    expected /= np.linalg.norm(expected)

    assert len(orientations) == 2
    assert np.allclose(orientations.grads, np.tile(expected, (2, 1)), atol=1e-6)


class _FakeRasterioDataset:
    """Minimal stand-in for a rasterio dataset, exposing just `.read()`/`.transform`."""

    def __init__(self, array: np.ndarray, transform: Affine):
        self._array = array
        self.transform = transform

    def read(self, band):
        return self._array


def test_raster_to_xyz():
    dem = np.array([[1.0, 2.0], [3.0, 4.0]])
    transform = Affine(1, 0, 0, 0, -1, 2)

    xyz = raster_to_xyz(dem, transform)

    expected = np.array([
        [0.5, 1.5, 1.0],
        [1.5, 1.5, 2.0],
        [0.5, 0.5, 3.0],
        [1.5, 0.5, 4.0],
    ])
    assert xyz.shape == (4, 3)
    assert np.allclose(xyz, expected)


def test_raster_to_xyz_accepts_rasterio_dataset():
    dem = np.array([[1.0, 2.0], [3.0, 4.0]])
    transform = Affine(1, 0, 0, 0, -1, 2)
    dataset = _FakeRasterioDataset(dem, transform)

    xyz = raster_to_xyz(dataset)

    assert xyz.shape == (4, 3)
    assert np.allclose(xyz[:, 2], [1.0, 2.0, 3.0, 4.0])


def test_raster_to_xyz_requires_transform_for_plain_array():
    with pytest.raises(ValueError):
        raster_to_xyz(np.zeros((2, 2)))


def test_orientations_from_surface_points_fits_plane():
    a, b = 0.3, -0.5
    x = np.array([0.0, 10.0, 0.0, 10.0, 5.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 5.0])
    z = a * x + b * y
    names = np.full(5, "horizon")

    surface_points = SurfacePointsTable.from_arrays(x=x, y=y, z=z, names=names)
    orientations = orientations_from_surface_points(surface_points)

    expected = np.array([-a, -b, 1.0])
    expected /= np.linalg.norm(expected)

    assert len(orientations) == 1
    assert np.allclose(orientations.grads[0], expected, atol=1e-8)
    assert orientations.name_id_map == surface_points.name_id_map


def test_orientations_from_surface_points_skips_sparse_formations():
    surface_points = SurfacePointsTable.from_arrays(
        x=np.array([0.0, 10.0, 0.0, 10.0, 5.0, 1.0]),
        y=np.array([0.0, 0.0, 10.0, 10.0, 5.0, 1.0]),
        z=np.zeros(6),
        names=np.array(["plenty", "plenty", "plenty", "plenty", "plenty", "sparse"]),
    )

    orientations = orientations_from_surface_points(surface_points)

    assert len(orientations) == 1
    assert set(orientations.name_id_map.keys()) == {"plenty", "sparse"}
    assert orientations.id == surface_points.name_id_map["plenty"]


def test_orientations_from_strike_lines_recovers_plane_gradient():
    dip_deg = 30.0
    horizontal_distance = 5.0
    dz = horizontal_distance * np.tan(np.radians(dip_deg))

    gdf = gpd.GeoDataFrame(
        {"formation": ["a", "a"], "elevation": [100.0, 100.0 - dz]},
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(0, 5), (10, 5)])],
    )

    orientations = orientations_from_strike_lines(gdf, formation_column="formation", elevation_column="elevation")

    expected = np.array([0.0, dz / horizontal_distance, 1.0])
    expected /= np.linalg.norm(expected)

    assert len(orientations) == 1
    assert np.allclose(orientations.grads[0], expected, atol=1e-6)
    assert set(orientations.name_id_map.keys()) == {"a"}


def test_orientations_from_strike_lines_sorts_by_elevation_regardless_of_row_order():
    dip_deg = 20.0
    dz = 5.0 * np.tan(np.radians(dip_deg))

    gdf = gpd.GeoDataFrame(
        {"formation": ["a", "a"], "elevation": [100.0 - dz, 100.0]},
        geometry=[LineString([(0, 5), (10, 5)]), LineString([(0, 0), (10, 0)])],
    )

    orientations = orientations_from_strike_lines(gdf, formation_column="formation", elevation_column="elevation")

    expected = np.array([0.0, dz / 5.0, 1.0])
    expected /= np.linalg.norm(expected)

    assert len(orientations) == 1
    assert np.allclose(orientations.grads[0], expected, atol=1e-6)


def test_orientations_from_strike_lines_skips_formations_with_one_line():
    gdf = gpd.GeoDataFrame(
        {"formation": ["sparse", "plenty", "plenty"], "elevation": [100.0, 100.0, 90.0]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 5), (10, 5)]),
        ],
    )

    orientations = orientations_from_strike_lines(gdf, formation_column="formation", elevation_column="elevation")

    assert len(orientations) == 1
    assert set(orientations.name_id_map.keys()) == {"plenty"}
