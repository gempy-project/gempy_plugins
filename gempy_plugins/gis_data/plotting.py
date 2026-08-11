"""2D map-view plotting for GIS-derived data -- the raw vector/raster inputs and the
DEM built from them, so they can be inspected before building the actual GemPy model.
The model itself has its own, much richer plotting via `gempy_viewer` -- this module
only covers the steps upstream of that.

Every 2D function here works as a one-liner (``plot_raster_2d(dem, transform)`` alone
pops up a figure) and also composes into a multi-panel figure by passing an existing
`ax` in -- see `show`.
"""
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from gempy_plugins.optional_dependencies import require_pyvista

if TYPE_CHECKING:
    import pyvista as pv
    from matplotlib.axes import Axes

    import gempy as gp
    from gempy_plugins.gis_data.alignment import BBox


def _new_ax(ax: Optional["Axes"]) -> Tuple["Axes", bool]:
    """Returns `(ax, created)`: an existing `ax` is reused as-is (`created=False`, so
    the caller is assumed to be composing a figure of their own and manage
    showing/saving it); `ax=None` gets a fresh figure (`created=True`)."""
    created = ax is None
    if created:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    ax.set_aspect("equal")
    return ax, created


def _finish(ax: "Axes", created: bool, show: Optional[bool]) -> None:
    """Show the figure by default only when this call created it -- calling `show()`
    on a panel the caller passed `ax` in for would pop up the partially-built figure
    before the remaining panels are drawn. Pass `show=True`/`False` explicitly to
    override either way."""
    if show is None:
        show = created
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def plot_raster_2d(
        raster: np.ndarray, transform, ax: Optional["Axes"] = None,
        cmap: str = "terrain", label: str = "elevation", show: Optional[bool] = None, **kwargs,
) -> "Axes":
    """Plot a DEM/raster array (e.g. `interpolate_raster`'s output) top-down, in real-
    world coordinates, using `transform` to place it correctly regardless of
    resolution. See `plot_raster_3d` for the same data as an actual 3D relief surface.

    Args:
        show: Pop up the figure once drawn. Defaults to `True` when this call created
            its own figure (`ax=None`), `False` when plotting into an `ax` passed in
            (assumed to be one panel of a figure the caller is building themselves).

    Extra `kwargs` are passed through to `imshow`.
    """
    ax, created = _new_ax(ax)
    height, width = raster.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    image = ax.imshow(raster, extent=(left, right, bottom, top), cmap=cmap, **kwargs)
    ax.figure.colorbar(image, ax=ax, label=label)
    _finish(ax, created, show)
    return ax


def plot_raster_3d(
        raster: np.ndarray, transform, cmap: str = "terrain", label: str = "elevation",
        show: bool = True, **kwargs,
) -> "pv.Plotter":
    """Plot a DEM/raster array as an actual 3D relief surface -- each cell placed at
    its own real-world XY (via `transform`) and lifted to its own elevation, colored
    by that same elevation. `plot_raster_2d` shows the same values top-down instead.

    Extra `kwargs` are passed through to `add_mesh`.
    """
    pv = require_pyvista()
    height, width = raster.shape

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs = transform.a * cols + transform.b * rows + transform.c
    ys = transform.d * cols + transform.e * rows + transform.f

    grid = pv.StructuredGrid(xs, ys, np.asarray(raster, dtype=float))
    grid[label] = grid.points[:, 2]

    plotter = pv.Plotter(notebook=False)
    plotter.add_mesh(grid, scalars=label, cmap=cmap, **kwargs)
    plotter.show_bounds(grid=True, location="outer")
    if show:
        plotter.show()
    return plotter


def plot_gis_data_3d(
        surface_points: Optional["gp.data.SurfacePointsTable"] = None,
        orientations: Optional["gp.data.OrientationsTable"] = None,
        topography_xyz: Optional[np.ndarray] = None,
        bbox: Optional["BBox"] = None,
        show: bool = True,
) -> "pv.Plotter":
    """3D preview of GIS-derived data -- surface points (colored by formation),
    orientations (black arrows along the gradient), and/or a topography point cloud
    (e.g. from `raster_to_xyz`), all in one view. Every argument is optional, so this
    also doubles as a general-purpose 3D viewer for whatever data is on hand.

    Pass `bbox = (minx, miny, maxx, maxy, minz, maxz)` to also draw its outline -- e.g.
    to preview a crop region before actually cropping via
    `crop_surface_points_to_bbox`/`crop_orientations_to_bbox`/`crop_points_to_bbox`/
    `crop_raster_to_bbox`.
    """
    pv = require_pyvista()
    plotter = pv.Plotter(notebook=False)
    all_xyz = []

    if topography_xyz is not None and len(topography_xyz) > 0:
        topography_xyz = np.asarray(topography_xyz, dtype=float)
        cloud = pv.PolyData(topography_xyz)
        cloud["elevation"] = topography_xyz[:, 2]
        plotter.add_mesh(
            cloud, scalars="elevation", cmap="terrain", point_size=3, render_points_as_spheres=True,
        )
        all_xyz.append(topography_xyz)

    if surface_points is not None and len(surface_points) > 0:
        from gempy.core.color_generator import ColorsGenerator
        palette = ColorsGenerator().hex_colors
        xyz = surface_points.xyz
        for i, (name, formation_id) in enumerate(surface_points.name_id_map.items()):
            mask = surface_points.data["id"] == formation_id
            if not mask.any():
                continue
            plotter.add_mesh(
                pv.PolyData(xyz[mask]), color=palette[i % len(palette)], point_size=10,
                render_points_as_spheres=True, label=str(name),
            )
        all_xyz.append(xyz)

    if orientations is not None and len(orientations) > 0:
        xyz = orientations.xyz
        span = np.concatenate(all_xyz + [xyz])[:, :2] if all_xyz else xyz[:, :2]
        diagonal = float(np.hypot(*(span.max(axis=0) - span.min(axis=0))))
        arrows = pv.PolyData(xyz)
        arrows["vectors"] = orientations.grads
        glyphs = arrows.glyph(orient="vectors", scale=False, factor=0.08 * diagonal if diagonal > 0 else 1.0)
        plotter.add_mesh(glyphs, color="black", label="orientations")
        all_xyz.append(xyz)

    if bbox is not None:
        minx, miny, maxx, maxy, minz, maxz = bbox
        box = pv.Box(bounds=(minx, maxx, miny, maxy, minz, maxz))
        plotter.add_mesh(box.outline(), color="red", line_width=3, label="bbox")

    if surface_points is not None or orientations is not None or bbox is not None:
        plotter.add_legend()
    plotter.show_bounds(grid=True, location="outer")
    if show:
        plotter.show()
    return plotter


def plot_vector_layer(
        gdf, ax: Optional["Axes"] = None, column: Optional[str] = None,
        legend: bool = True, show: Optional[bool] = None, **kwargs,
) -> "Axes":
    """Plot a vector layer (topographic contours, digitized interfaces, strike lines,
    ...) in map view, optionally colored by one of its columns (e.g. an elevation or
    formation column).

    Args:
        show: See `plot_raster`.

    Extra `kwargs` are passed through to `GeoDataFrame.plot`.
    """
    ax, created = _new_ax(ax)
    # `GeoDataFrame.plot` auto-picks an aspect ratio from the data's CRS, assuming a
    # geographic (lat/lon) one whenever it can't tell otherwise -- wrong for the local
    # planar coordinates these layers actually carry, and prone to raising outright on
    # data that isn't in the -90..90 range `aspect="equal"` sidesteps that guess.
    kwargs.setdefault("aspect", "equal")
    gdf.plot(ax=ax, column=column, legend=legend and column is not None, **kwargs)
    _finish(ax, created, show)
    return ax


def plot_input_layers(
        interfaces=None, topo_contours=None, strike_lines=None,
        formation_column: str = "formation", elevation_column: str = "Z",
        ax: Optional["Axes"] = None, show: Optional[bool] = None,
) -> "Axes":
    """Plot digitized interfaces, topographic contours, and strike lines together in
    one map view -- the raw vector inputs to a GIS-to-GemPy conversion, overlaid so
    their spatial relationship (e.g. which strike line sits on which contour) is
    visible at a glance. Every layer is optional, so this works with whatever subset
    of raw data is on hand.

    - `topo_contours` are drawn in shades of brown (darker = higher), each labeled
      inline with its elevation.
    - `interfaces` get one color per formation from GemPy's own default palette
      (`gempy.core.color_generator.ColorsGenerator`), each formation labeled inline
      once, at a representative point on its own geometry.
    - `strike_lines` are drawn last, in solid black, on top of the other two.

    Args:
        formation_column: Column in `interfaces` giving each line's formation name.
        elevation_column: Column in `topo_contours` giving each line's elevation.
        show: See `plot_raster`.
    """
    ax, created = _new_ax(ax)

    if topo_contours is not None:
        _plot_contours(ax, topo_contours, elevation_column)
    if interfaces is not None:
        _plot_interfaces(ax, interfaces, formation_column)
    if strike_lines is not None:
        strike_lines.plot(ax=ax, color="black", linewidth=1.5, aspect="equal")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    _finish(ax, created, show)
    return ax


def _plot_contours(ax: "Axes", gdf, elevation_column: str) -> None:
    gdf.plot(ax=ax, column=elevation_column, cmap="YlOrBr", linewidth=1.0, aspect="equal")
    for value, geom in zip(gdf[elevation_column].to_numpy(dtype=float), gdf.geometry):
        _label_at(ax, geom.representative_point(), f"{value:.0f}", color="#3e2412")


def _plot_interfaces(ax: "Axes", gdf, formation_column: str) -> None:
    from gempy.core.color_generator import ColorsGenerator
    palette = ColorsGenerator().hex_colors

    names = list(dict.fromkeys(gdf[formation_column]))  # unique, first-seen order
    colors = {name: palette[i % len(palette)] for i, name in enumerate(names)}

    for name, group in gdf.groupby(formation_column, sort=False):
        color = colors[name]
        group.plot(ax=ax, color=color, linewidth=2.5, aspect="equal")
        _label_at(ax, group.geometry.union_all().representative_point(), str(name), color=color)


def _label_at(ax: "Axes", point, text: str, color: str) -> None:
    """Inline text label at a single point, e.g. one contour's elevation or one
    formation's name -- a white halo behind the text keeps it legible over whatever
    line/other labels it ends up sitting on."""
    ax.annotate(
        text, (point.x, point.y), color=color, fontsize=8, fontweight="bold",
        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
    )
