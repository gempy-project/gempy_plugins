"""Interactive PyVista summaries of the geological-coordinate experiments.

Run in separate processes to keep both native VTK windows interactive::

    python -m gempy_plugins.property_estimation.examples.experiment_views --view facies
    python -m gempy_plugins.property_estimation.examples.experiment_views --view distances

Use --no-show --screenshot image.png for a reproducible headless rendering.
"""
import argparse

import numpy as np

from gempy_plugins.optional_dependencies import require_pyvista
from .geological_models import make_geological_model
from .truncated_gaussian_facies import run_demo
from ..gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from ..geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding
from ..geological_coordinates.analytical import constant_metric_coordinates
from ..geological_coordinates.diagnostics import constant_metric_quality
from ..plotting import _build_structured_grid


BACKGROUND = '#f4f6f8'
INK = '#182738'
FACIES_COLORS = ['#616b7b', '#86b6b0', '#edc66e']
CAMERA = [(155, -195, 130), (50, 20, 45), (0, 0, 1)]


def _plotter(title, off_screen):
    pv = require_pyvista()
    plotter = pv.Plotter(shape=(2, 2), window_size=(1400, 1000), title=title,
                         notebook=False, off_screen=off_screen, border_color='#c4ccd5')
    plotter.set_background(BACKGROUND, all_renderers=True)
    return plotter


def _active_mesh(model, embedding, values, name):
    grid = _build_structured_grid(model)
    # Cell scalars and physical positions must use the same ordering.
    np.testing.assert_allclose(grid.cell_centers().points, model.grid.dense_grid.values, atol=1e-7)
    active = np.zeros(grid.n_cells, dtype=np.uint8)
    active[embedding.node_indices] = 1
    scalars = np.full(grid.n_cells, np.nan)
    scalars[embedding.node_indices] = values
    grid.cell_data['active'] = active
    grid.cell_data[name] = scalars
    return grid.threshold(.5, scalars='active')


def _outline(plotter, mesh):
    plotter.add_mesh(mesh.outline(), color='#a2adba', line_width=1, opacity=.5)
    plotter.add_axes(color=INK, viewport=(.82, 0., 1., .18))


def _cameras(plotter):
    for row in range(2):
        for column in range(2):
            plotter.subplot(row, column)
            plotter.camera_position = CAMERA
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = 63
    plotter.link_views()
    framing = {'scale': 63.}

    def fit_resize(*_):
        width, height = plotter.window_size
        if width <= 0 or height <= 0:
            return
        scale = max(63., 63. * height / width)
        # Preserve user zoom relative to the framing when the window is resized.
        plotter.camera.parallel_scale *= scale / framing['scale']
        framing['scale'] = scale

    plotter.iren.add_observer('ConfigureEvent', fit_resize)
    fit_resize()


def _cutaway_toggle(plotter, meshes_and_actors, cut_y=20.):
    state = {'cut': True}

    def toggle():
        state['cut'] = not state['cut']
        for mesh, actor in meshes_and_actors:
            actor.mapper.dataset = (mesh.clip(normal='y', origin=(0, cut_y, 0), invert=False)
                                    if state['cut'] else mesh)
        plotter.render()

    plotter.add_key_event('c', toggle)


def facies_view(*, off_screen=False):
    """Show one continuous/facies pair, independent fault blocks, and a failed case."""
    pv = require_pyvista()
    from matplotlib.colors import ListedColormap

    cases = run_demo(models=('folded', 'faulted', 'thin_layer'))
    model, fields, report = cases[0]
    simulator = GaussianFieldSimulator(fields.embedding, GaussianCovariance(length_scale=5), n_frequencies=1024)
    latent = simulator.simulate(realizations=1, seed=13).values[0].numpy()
    # These are the same latent draw and thresholds used for the facies panel.
    thresholds = np.asarray(fields.metadata['standardized_thresholds'])
    np.testing.assert_array_equal(np.searchsorted(thresholds, latent, side='right'), fields.values[0].numpy())
    density = 2.65 + .1 * latent
    title = 'GemPy experiments 1 - Gaussian properties and facies'
    plotter = _plotter(title, off_screen)
    actors = []
    panels = [(model, fields, report, density, 'Density'),
              (model, fields, report, fields.values[0].numpy(), 'Facies'),
              (*cases[1], cases[1][1].values[0].numpy(), 'Facies'),
              (*cases[2], cases[2][1].values[0].numpy(), 'Facies')]
    for panel, (model, fields, report, values, scalar) in enumerate(panels):
        plotter.subplot(panel // 2, panel % 2)
        mesh = _active_mesh(model, fields.embedding, values, scalar)
        clipped = mesh.clip(normal='y', origin=(0, 20, 0), invert=False)
        options = dict(scalars=scalar, preference='cell', show_edges=False, lighting=False)
        if scalar == 'Density':
            options.update(cmap='viridis', clim=(2.35, 2.95), scalar_bar_args=dict(
                title='Density (example units)', color=INK, vertical=False,
                position_x=.20, position_y=.06, width=.60, height=.09,
                title_font_size=13, label_font_size=11))
        else:
            options.update(cmap=ListedColormap(FACIES_COLORS), clim=(-.5, 2.5), n_colors=3, show_scalar_bar=False)
            plotter.add_legend(labels=list(zip(('Shale', 'Limestone', 'Sandstone'), FACIES_COLORS)),
                               bcolor=BACKGROUND, face='rectangle', loc='lower left', size=(.22, .16))
        actor = plotter.add_mesh(clipped, **options)
        actors.append((mesh, actor))
        _outline(plotter, mesh)
        quality = report['quality']
        if panel == 0:
            text = 'ANTICLINE: Gaussian density\nOne realization, following bedding'
        elif panel == 1:
            agreement = report['facies']['directional_agreement']
            along = agreement['parallel']['same_facies_probability']
            across = agreement['perpendicular']['same_facies_probability']
            text = ('ANTICLINE: Same draw, converted to facies\n'
                    f'64-draw agreement: along {along:.0%} / across {across:.0%}')
        elif panel == 2:
            text = ('FAULT-SEPARATED FOLD\nIndependent draws on each side\n'
                    'Red plane: barrier, no imposed displacement')
            plane = pv.Plane(center=(50, 20, 50), direction=(1, 0, 0), i_size=80, j_size=40)
            plotter.add_mesh(plane, color='#c44343', opacity=.15, lighting=False)
            plotter.add_mesh(plane.outline(), color='#c44343', line_width=2, lighting=False)
        else:
            text = ('THIN LAYER: NOT VALIDATED\n'
                    f'Neighbor overlap {quality["neighbor_overlap"]:.0%}; required >=90%\n'
                    'Plausible appearance does not validate geometry')
        plotter.add_text(text, position='upper_left', font_size=12, color='#a42e24' if panel == 3 else INK)
        print(f'{report["geometry"]}: overlap={quality["neighbor_overlap"]:.3f}, '
              f'covariance P90={quality["covariance_absolute_p90"]:.3f}, '
              f'gate={quality["passes_provisional_targets"]}', flush=True)
    _cameras(plotter)
    _cutaway_toggle(plotter, actors)
    return plotter, title


def distance_view(*, off_screen=False):
    """Separate graph and MDS distortion against analytical horizontal anisotropy."""
    pv = require_pyvista()
    model = make_geological_model('horizontal')
    metric = GeologicalMetricField.from_gempy(model, 'lower', anisotropy_ratio=5)
    exact_coordinates = constant_metric_coordinates(metric)
    positions = metric.positions[metric.active_mask]
    source = int(np.argmin(np.linalg.norm(positions - [50, 20, 43], axis=1)))
    source_position = positions[source]
    exact_distance = np.linalg.norm(exact_coordinates - exact_coordinates[source], axis=1)
    target = np.exp(-.5 * (exact_distance / 5.) ** 2)
    title = 'GemPy experiments 2 - Where distance errors come from'
    plotter = _plotter(title, off_screen)
    actors = []
    for row, (radius, dimensions) in enumerate(((1, 6), (5, 3))):
        graph = AnisotropicGraph.from_metric(metric, stencil_radius=radius)
        embedding = GeologicalEmbedding.from_graph(graph, landmarks=32, dimensions=dimensions)
        quality = constant_metric_quality(graph, embedding, length_scale=5)
        distances = (graph.shortest_paths([source])[0],
                     np.linalg.norm(embedding.coordinates - embedding.coordinates[source], axis=1))
        for column, (kind, distance) in enumerate(zip(('graph', 'embedding'), distances)):
            plotter.subplot(row, column)
            error = np.abs(np.exp(-.5 * (distance / 5.) ** 2) - target)
            error[exact_distance > 15] = np.nan
            mesh = _active_mesh(model, embedding, error, 'Covariance error')
            clipped = mesh.clip(normal='y', origin=source_position, invert=False)
            actor = plotter.add_mesh(clipped, scalars='Covariance error', preference='cell',
                                     cmap='YlOrRd', clim=(0., .35), nan_color=BACKGROUND,
                                     lighting=False, show_scalar_bar=False)
            actors.append((mesh, actor))
            # Keep one scalar bar per renderer so the scale remains visible when tiled.
            plotter.add_scalar_bar(title=f'Covariance error: {kind}, radius {radius}', mapper=actor.mapper, color=INK,
                                   vertical=False, position_x=.20, position_y=.06,
                                   width=.60, height=.09, title_font_size=13, label_font_size=11)
            plotter.add_mesh(pv.Sphere(radius=1.8, center=source_position), color='#00a6c8', lighting=False)
            _outline(plotter, mesh)
            p90 = quality[f'{kind}_vs_analytical']['covariance_absolute_p90']
            heading = 'ORIGINAL' if row == 0 else 'IMPROVED'
            stage = f'Graph, radius {radius}' if column == 0 else f'{dimensions}D embedding, radius {radius}'
            plotter.add_text(f'{heading}: {stage}\n'
                             f'32-source P90 covariance error: {p90:.3f}\n'
                             'Colors: cyan-source error vs analytical',
                             position='upper_left', font_size=12, color=INK)
            print(f'{heading} {stage}: analytical covariance P90={p90:.4f}', flush=True)
    _cameras(plotter)
    _cutaway_toggle(plotter, actors, cut_y=float(source_position[1]))
    return plotter, title


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', choices=('facies', 'distances'), default='facies')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--screenshot')
    args = parser.parse_args()
    build = facies_view if args.view == 'facies' else distance_view
    plotter, title = build(off_screen=args.no_show)
    if args.no_show:
        plotter.show(screenshot=args.screenshot, interactive=False)
        return
    plotter.show(interactive_update=True, auto_close=False)
    plotter.update(stime=200, force_redraw=True)
    if args.screenshot:
        plotter.screenshot(args.screenshot)
    print(f'WINDOW_OPEN: {title}; mapped={plotter.ren_win.GetMapped()}; '
          f'backend={plotter.ren_win.GetClassName()}', flush=True)
    print('Drag: rotate linked panels. Scroll: zoom. C: toggle cutaway. Q: close window.', flush=True)
    try:
        plotter.iren.start()
    finally:
        plotter.close()


if __name__ == '__main__':
    main()
