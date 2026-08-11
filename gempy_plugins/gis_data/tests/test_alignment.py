import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import Point

from gempy.core.data.orientations import OrientationsTable
from gempy.core.data.surface_points import SurfacePointsTable
from gempy_plugins.gis_data.alignment import (
    align_to_minimum_extent, crop_orientations_to_bbox, crop_points_to_bbox, crop_raster_to_bbox,
    crop_surface_points_to_bbox, crop_vector_to_bbox,
)

_NO_Z_LIMIT = (float("-inf"), float("inf"))  # xy-only crops ignore these


def _bbox_area(xy: np.ndarray) -> float:
    return float((xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min()))


def test_align_to_minimum_extent_shrinks_bbox():
    theta = np.radians(25.0)
    local = np.array([[-10.0, -2.0], [10.0, -2.0], [10.0, 2.0], [-10.0, 2.0]])
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    world_xy = local @ rotation.T + np.array([50.0, 50.0])
    xyz = np.column_stack([world_xy, np.zeros(4)])

    surface_points = SurfacePointsTable.from_arrays(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], names=np.full(4, "rock"),
    )
    orientations = OrientationsTable.initialize_empty()

    rotated_sp, rotated_ori, transform = align_to_minimum_extent(surface_points, orientations)

    original_area = _bbox_area(xyz[:, :2])
    rotated_area = _bbox_area(rotated_sp.xyz[:, :2])

    assert rotated_area < original_area
    assert rotated_area == pytest.approx(20.0 * 4.0, rel=1e-6)
    assert len(rotated_ori) == 0
    assert np.allclose(transform.apply_inverse(rotated_sp.xyz), xyz, atol=1e-8)


def test_align_to_minimum_extent_rotates_gradients():
    surface_points = SurfacePointsTable.from_arrays(
        x=np.array([0.0, 10.0, 10.0]), y=np.array([0.0, 0.0, 3.0]), z=np.zeros(3),
        names=np.full(3, "rock"),
    )
    orientations = OrientationsTable.from_arrays(
        x=np.array([5.0]), y=np.array([1.0]), z=np.array([0.0]),
        G_x=np.array([1.0]), G_y=np.array([0.0]), G_z=np.array([0.0]),
        names=np.array(["rock"]),
    )

    _, rotated_ori, transform = align_to_minimum_extent(surface_points, orientations)

    theta = np.radians(transform.rotation[2])
    expected = np.array([np.cos(theta), np.sin(theta), 0.0])
    assert np.allclose(rotated_ori.grads[0], expected, atol=1e-8)


def test_crop_vector_to_bbox():
    gdf = gpd.GeoDataFrame({"formation": ["a", "b"]}, geometry=[Point(1, 1), Point(20, 20)])

    cropped = crop_vector_to_bbox(gdf, (0.0, 0.0, 10.0, 10.0, *_NO_Z_LIMIT))

    assert len(cropped) == 1
    assert cropped.iloc[0]["formation"] == "a"


def test_crop_points_to_bbox():
    xyz = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0], [15.0, 15.0, 0.0]])

    cropped = crop_points_to_bbox(xyz, (0.0, 0.0, 10.0, 10.0, *_NO_Z_LIMIT))

    assert cropped.shape == (2, 3)


def test_crop_points_to_bbox_ignores_z_bounds():
    xyz = np.array([[5.0, 5.0, -1000.0], [5.0, 5.0, 1000.0]])

    cropped = crop_points_to_bbox(xyz, (0.0, 0.0, 10.0, 10.0, 0.0, 0.0))

    assert cropped.shape == (2, 3)  # both kept -- z bounds (0.0, 0.0) would exclude both if honored


def test_crop_raster_to_bbox():
    dem = np.arange(100, dtype=float).reshape(10, 10)
    transform = Affine(1, 0, 0, 0, -1, 10)

    cropped, new_transform = crop_raster_to_bbox(dem, transform, (2.0, 2.0, 5.0, 5.0, *_NO_Z_LIMIT))

    assert cropped.shape == (4, 4)
    assert cropped[0, 0] == dem[5, 2]
    assert (new_transform.c, new_transform.f) == (2.0, 5.0)


def test_crop_surface_points_to_bbox():
    surface_points = SurfacePointsTable.from_arrays(
        x=np.array([1.0, 5.0, 20.0]), y=np.array([1.0, 5.0, 20.0]), z=np.zeros(3),
        names=np.array(["a", "a", "b"]),
    )

    cropped = crop_surface_points_to_bbox(surface_points, (0.0, 0.0, 10.0, 10.0, *_NO_Z_LIMIT))

    assert len(cropped) == 2
    assert cropped.name_id_map == surface_points.name_id_map
    assert np.array_equal(cropped.data["id"], surface_points.data["id"][:2])


def test_crop_surface_points_to_bbox_respects_z_bounds():
    surface_points = SurfacePointsTable.from_arrays(
        x=np.array([1.0, 1.0]), y=np.array([1.0, 1.0]), z=np.array([-50.0, 50.0]),
        names=np.array(["a", "a"]),
    )

    cropped = crop_surface_points_to_bbox(surface_points, (0.0, 0.0, 10.0, 10.0, 0.0, 100.0))

    assert len(cropped) == 1
    assert cropped.data["Z"][0] == 50.0


def test_crop_orientations_to_bbox():
    orientations = OrientationsTable.from_arrays(
        x=np.array([1.0, 20.0]), y=np.array([1.0, 20.0]), z=np.zeros(2),
        G_x=np.array([0.0, 0.0]), G_y=np.array([0.0, 0.0]), G_z=np.array([1.0, 1.0]),
        names=np.array(["a", "a"]),
    )

    cropped = crop_orientations_to_bbox(orientations, (0.0, 0.0, 10.0, 10.0, *_NO_Z_LIMIT))

    assert len(cropped) == 1
    assert cropped.name_id_map == orientations.name_id_map
    assert cropped.grads[0].tolist() == [0.0, 0.0, 1.0]


def test_crop_orientations_to_bbox_respects_z_bounds():
    orientations = OrientationsTable.from_arrays(
        x=np.array([1.0, 1.0]), y=np.array([1.0, 1.0]), z=np.array([-50.0, 50.0]),
        G_x=np.array([0.0, 0.0]), G_y=np.array([0.0, 0.0]), G_z=np.array([1.0, 1.0]),
        names=np.array(["a", "a"]),
    )

    cropped = crop_orientations_to_bbox(orientations, (0.0, 0.0, 10.0, 10.0, 0.0, 100.0))

    assert len(cropped) == 1
    assert cropped.data["Z"][0] == 50.0
