"""Visualization for property estimation: the domain partitioning itself, and the
resulting property field(s). Mostly independent of gempy_viewer -- it has no hook for
arbitrary custom scalar fields, only its own lithology/scalar-field/values arrays -- so
most of this builds its own pyvista StructuredGrid, mirroring the corner-point/cell-data
construction gempy_viewer itself uses internally. `plot_conditioning_data` is the
exception: it reuses gempy_viewer's own surface rendering, which is exactly its
strength and not something worth reimplementing.
"""
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

import gempy as gp
from gempy_plugins.optional_dependencies import require_gempy_viewer, require_pyvista
from gempy_plugins.property_estimation.domains import DomainKey, describe_domains, domain_keys_from_arrays
from gempy_plugins.property_estimation.kriging import PropertyField

if TYPE_CHECKING:
    import pyvista as pv

    from gempy_plugins.property_estimation.conditioning_data import ConditioningData


def _build_structured_grid(geo_model: gp.data.GeoModel) -> "pv.StructuredGrid":
    pv = require_pyvista()
    rg = geo_model.grid.regular_grid
    resolution = rg.resolution
    corners = rg.get_values_vtk_format()
    grid_3d = corners.reshape(*(resolution + 1), 3).T
    return pv.StructuredGrid(*grid_3d)


def _domain_colors(domain_keys: List[DomainKey]) -> Dict[DomainKey, Tuple[float, float, float]]:
    """One base hue per lithology (from a qualitative colormap), with each of that
    lithology's fault-block variants rendered as a distinct shade of the same hue --
    lighter for lower fault-block ids, darker for higher ones. Makes it easy to spot
    "same rock unit, different fault block" at a glance, instead of unrelated colors.
    """
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    lith_ids = sorted({lith_id for lith_id, _ in domain_keys})
    base_cmap = mpl.colormaps['tab10']
    lith_id_to_base_rgb = {lith_id: base_cmap(i % 10)[:3] for i, lith_id in enumerate(lith_ids)}

    colors: Dict[DomainKey, Tuple[float, float, float]] = {}
    for lith_id in lith_ids:
        fault_ids = sorted(fault_id for l, fault_id in domain_keys if l == lith_id)
        h, s, v = mcolors.rgb_to_hsv(lith_id_to_base_rgb[lith_id])
        n = len(fault_ids)
        for i, fault_id in enumerate(fault_ids):
            value_factor = 1.0 if n == 1 else 1.0 - 0.45 * (i / (n - 1))
            colors[(lith_id, fault_id)] = tuple(mcolors.hsv_to_rgb((h, s, v * value_factor)))
    return colors


def plot_domains(
        geo_model: gp.data.GeoModel,
        lith_array: np.ndarray,
        fault_array: np.ndarray,
        domain_keys: Optional[List[DomainKey]] = None,
        show: bool = True,
        **kwargs,
) -> "pv.Plotter":
    """Visualize the domain partitioning -- a sanity check before committing conditioning
    data/configs to it. Each lithology gets its own base color; its different fault-block
    domains get distinct shades of that same color, so "same rock unit, different fault
    block" is visually obvious.
    """
    pv = require_pyvista()
    import matplotlib.colors as mcolors

    grid = _build_structured_grid(geo_model)

    domain_keys = domain_keys or domain_keys_from_arrays(lith_array, fault_array)
    n_domains = len(domain_keys)

    # a sequential 0..n-1 index per domain, for genuinely discrete/categorical coloring
    # -- a sparse combined id like lith_id*1000+fault_id would be treated as a
    # *continuous* scalar range by pyvista, bunching nearby domains into near-identical
    # colors and wasting most of the color range on the (large) unused gaps between them
    key_to_index = {domain_key: i for i, domain_key in enumerate(domain_keys)}
    combined = np.stack([lith_array.ravel(), fault_array.ravel()], axis=-1)
    index_array = np.array([key_to_index[(int(l), int(f))] for l, f in combined], dtype=np.int64)
    grid.cell_data['domain'] = index_array

    descriptions = describe_domains(geo_model, domain_keys)
    colors = _domain_colors(domain_keys)
    listed_cmap = mcolors.ListedColormap([colors[domain_key] for domain_key in domain_keys])

    plotter = pv.Plotter(notebook=False)
    plotter.add_mesh(
        grid, scalars='domain', show_edges=True, cmap=listed_cmap,
        n_colors=n_domains, clim=[0, n_domains - 1],
        show_scalar_bar=False, **kwargs
    )
    # a real item-by-item legend instead of a scalar bar -- scalar bars with many
    # annotations render unclearly and get cut off, a plain legend is far more robust
    plotter.add_legend(
        labels=[[descriptions[domain_key], colors[domain_key]] for domain_key in domain_keys],
        bcolor='white', border=True, face='rectangle', size=(0.18, 0.035 * n_domains), loc='upper left',
    )
    plotter.show_bounds(bounds=geo_model.grid.regular_grid.extent, location='furthest', grid=True)
    if show:
        plotter.show()
    return plotter


def plot_fault_blocks(
        geo_model: gp.data.GeoModel,
        fault_array: np.ndarray,
        show: bool = True,
        **kwargs,
) -> "pv.Plotter":
    """Visualize fault-block membership alone, ignoring lithology -- shows exactly
    which cells belong to which fault block. `plot_domains` shows fault blocks only as
    shades of a lithology's color, which doesn't say *which* block is which; this is
    the complementary view for actually deciding whether domains should be merged
    across a fault (e.g. a negligible-offset fault, or one that doesn't affect a given
    lithology at all -- see `domains.py`'s notes on `fault_block` being purely
    geometric, independent of `fault_relations`).
    """
    pv = require_pyvista()
    import matplotlib as mpl

    grid = _build_structured_grid(geo_model)

    fault_ids = sorted(int(f) for f in np.unique(fault_array))
    key_to_index = {fault_id: i for i, fault_id in enumerate(fault_ids)}
    index_array = np.array([key_to_index[int(f)] for f in fault_array.ravel()], dtype=np.int64)
    grid.cell_data['fault_block'] = index_array

    n = len(fault_ids)
    base_cmap = mpl.colormaps['tab10']
    colors = {fault_id: base_cmap(index % 10)[:3] for fault_id, index in key_to_index.items()}
    listed_cmap = mpl.colors.ListedColormap([colors[fault_id] for fault_id in fault_ids])

    plotter = pv.Plotter(notebook=False)
    plotter.add_mesh(
        grid, scalars='fault_block', show_edges=True, cmap=listed_cmap,
        n_colors=n, clim=[0, n - 1], show_scalar_bar=False, **kwargs
    )
    plotter.add_legend(
        labels=[[f"fault block {fault_id}", colors[fault_id]] for fault_id in fault_ids],
        bcolor='white', border=True, face='rectangle', size=(0.18, 0.035 * n), loc='upper left',
    )
    plotter.show_bounds(bounds=geo_model.grid.regular_grid.extent, location='furthest', grid=True)
    if show:
        plotter.show()
    return plotter


def plot_conditioning_data(
        geo_model: gp.data.GeoModel,
        conditioning_data: "ConditioningData",
        show: bool = True,
        point_size: float = 10.0,
        cmap: str = "viridis",
        **kwargs,
) -> "pv.Plotter":
    """Visualize conditioning data points, colored by value, together with the
    computed model's structural surfaces alone (no lithology block, no input-data
    overlay) -- useful for seeing where samples actually sit relative to the structure
    before running any kriging/simulation.

    Extra `kwargs` are passed through to `gempy_viewer.plot_3d`.
    """
    gpv = require_gempy_viewer()
    pv = require_pyvista()

    p3d = gpv.plot_3d(geo_model, show_lith=False, show_surfaces=True, show_data=False, show=False, **kwargs)

    points = pv.PolyData(conditioning_data.xyz)
    points['value'] = conditioning_data.values
    p3d.p.add_mesh(points, scalars='value', point_size=point_size, render_points_as_spheres=True, cmap=cmap)

    if show:
        p3d.p.show()
    return p3d


def plot_property_field(
        geo_model: gp.data.GeoModel,
        field: PropertyField,
        show: bool = True,
        **kwargs,
) -> "pv.Plotter":
    """Visualize a `PropertyField` result. Cells outside `field.domain_keys` (never
    computed) are `nan` and simply aren't rendered -- a partial run shows exactly the
    domains that were actually processed.
    """
    pv = require_pyvista()
    grid = _build_structured_grid(geo_model)
    grid.cell_data['property'] = field.values.ravel()
    grid = grid.threshold(scalars='property')  # drops nan cells (unpopulated domains)

    plotter = pv.Plotter(notebook=False)
    plotter.add_mesh(grid, scalars='property', show_edges=True, **kwargs)
    plotter.show_bounds(bounds=geo_model.grid.regular_grid.extent, location='furthest', grid=True)
    if show:
        plotter.show()
    return plotter
