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
from gempy_plugins.property_estimation.domains import (
    DomainKey, compute_domains, describe_domains, domain_keys_from_arrays,
)
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


def plot_property_field_interactive(
        geo_model: gp.data.GeoModel,
        field: PropertyField,
        domain_configs: Optional[Dict] = None,
        show: bool = True,
        window_size: Tuple[int, int] = (1400, 900),
        **kwargs,
) -> "pv.Plotter":
    """Interactive `PropertyField` viewer: each domain gets its own visibility checkbox
    and a min/max pair of sliders that thresholds just that domain's displayed cells --
    e.g. to isolate high-value cells in one domain while hiding another entirely.
    `plot_property_field` stays the simple default; this is the heavier, exploratory
    alternative.

    `field.domain_keys` is a flat list of individual `DomainKey`s -- a merged group
    (e.g. one lithology merged across two fault blocks) populates several entries in
    it, not one. Pass the same `domain_configs` dict used for `run_kriging`/
    `run_simulation` to collapse those back into a single row per group, matching how
    the run was actually configured; without it, every individual domain key gets its
    own row.
    """
    pv = require_pyvista()
    from gempy_plugins.property_estimation.domains import as_group, group_mask

    lith_array, fault_array, _ = compute_domains(geo_model)
    descriptions = describe_domains(geo_model, field.domain_keys)
    base_grid = _build_structured_grid(geo_model)

    field_key_set = set(field.domain_keys)
    if domain_configs is not None:
        candidate_groups = [as_group(k) for k in domain_configs.keys()]
    else:
        candidate_groups = [(k,) for k in field.domain_keys]
    # keep only the members that were actually populated, dropping groups left empty
    groups = [tuple(k for k in g if k in field_key_set) for g in candidate_groups]
    groups = [g for g in groups if g]

    def group_label(group):
        # same lithology merged across fault blocks -- the fault-block suffix in
        # `descriptions` is exactly what merging is meant to hide, so strip it
        names = []
        for domain_key in group:
            name = descriptions[domain_key].split(" (fault block")[0]
            if name not in names:
                names.append(name)
        return " + ".join(names)

    plotter = pv.Plotter(notebook=False, window_size=list(window_size))
    plotter.show_bounds(bounds=geo_model.grid.regular_grid.extent, location='furthest', grid=True)

    n_rows = len(groups)
    # a compact, fixed-height row stacked tightly in the top-left corner -- not spread
    # across the full window height, and not wide enough to reach into the plot itself.
    # Caps at `0.85 / n_rows` only so many rows don't run off the bottom of the window.
    row_height = min(0.09, 0.85 / max(n_rows, 1))
    checkbox_size = 12
    slider_kwargs = dict(title_height=0.012, slider_width=0.015, tube_width=0.005)

    for i, group in enumerate(groups):
        mask = group_mask(lith_array, fault_array, group).ravel()
        cell_indices = np.where(mask)[0]

        domain_grid = base_grid.copy()
        domain_grid.cell_data['property'] = field.values.ravel()
        full_mesh = domain_grid.extract_cells(cell_indices)
        display_mesh = full_mesh.copy()

        vmin = float(np.nanmin(full_mesh.cell_data['property']))
        vmax = float(np.nanmax(full_mesh.cell_data['property']))
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5

        actor = plotter.add_mesh(
            display_mesh, scalars='property', show_edges=True, clim=[vmin, vmax], **kwargs
        )

        thresholds = {'lo': vmin, 'hi': vmax}

        def make_threshold_callback(full_mesh=full_mesh, display_mesh=display_mesh, thresholds=thresholds, key=None):
            def callback(value):
                thresholds[key] = value
                lo, hi = sorted((thresholds['lo'], thresholds['hi']))
                display_mesh.shallow_copy(full_mesh.threshold(value=[lo, hi], scalars='property'))
                # shallow_copy alone doesn't trigger a re-render -- the mapper needs an
                # explicit nudge to notice the dataset's cell count actually changed
                display_mesh.Modified()
                plotter.render()

            return callback

        row_top = 0.95 - i * row_height
        # checkbox and label both anchor at their bottom-left corner in plain pixel
        # coordinates -- sharing one coordinate system (rather than the checkbox's
        # pixels against the label's normalized-viewport fraction) is what actually
        # keeps them level, instead of relying on two independently-tuned offsets
        row_bottom_px = row_top * window_size[1] - checkbox_size

        plotter.add_checkbox_button_widget(
            callback=actor.SetVisibility,
            value=True,
            position=(10.0, row_bottom_px),
            size=checkbox_size,
        )
        plotter.add_text(
            group_label(group),
            position=(10.0 + checkbox_size + 6.0, row_bottom_px),
            font_size=8,
        )
        # sliders sit directly under the checkbox/label, narrow and confined to the
        # same left-hand corner -- not stretched out across (and over) the plot
        slider_y = row_top - row_height * 0.62
        lo_slider = plotter.add_slider_widget(
            callback=make_threshold_callback(key='lo'),
            rng=[vmin, vmax], value=vmin, title="min",
            pointa=(0.045, slider_y), pointb=(0.125, slider_y),
            **slider_kwargs,
        )
        hi_slider = plotter.add_slider_widget(
            callback=make_threshold_callback(key='hi'),
            rng=[vmin, vmax], value=vmax, title="max",
            pointa=(0.15, slider_y), pointb=(0.23, slider_y),
            **slider_kwargs,
        )
        # `title_height`/`slider_width`/`tube_width` above don't touch the handle or end
        # caps at all -- pyvista hardcodes those (`SetSliderLength(0.05)`,
        # `SetEndCapLength(0.01)`) with no kwarg for either, so they have to be shrunk
        # directly on the representation after creation
        for slider in (lo_slider, hi_slider):
            rep = slider.GetSliderRepresentation()
            rep.SetSliderLength(0.008)
            rep.SetEndCapLength(0.008)
            rep.SetEndCapWidth(0.008)
            rep.SetLabelHeight(0.012)

    if show:
        plotter.show()
    return plotter
