"""Seed-matched density conditioning in an anticline and across a fault barrier.

Two exact off-grid observations and one noisy observation constrain each model.
Run --no-show --output conditioning.png --report conditioning.json for sections,
or --volume for a linked PyVista comparison of the last requested model.
"""
import argparse
from dataclasses import dataclass
import json

import numpy as np

from .geological_models import make_geological_model
from ..gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from ..geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding
from ..geological_coordinates.diagnostics import embedding_quality


@dataclass
class ConditioningCase:
    model: object
    graph: AnisotropicGraph
    prior: object
    conditional: object
    moments: object
    positions: np.ndarray
    data: np.ndarray
    noise_variance: np.ndarray
    report: dict


def run_demo(models=('folded', 'faulted'), *, realizations=64, device='cpu'):
    cases = []
    for name in models:
        if name not in ('folded', 'faulted'):
            raise ValueError("The conditioning demo supports folded and faulted models")
        model = make_geological_model(name)
        metric = GeologicalMetricField.from_gempy(model, 'lower', anisotropy_ratio=5)
        graph = AnisotropicGraph.from_metric(metric, stencil_radius=3)
        embedding = GeologicalEmbedding.from_graph(graph, landmarks=32, dimensions=3)
        simulator = GaussianFieldSimulator(embedding, GaussianCovariance(5, .1 ** 2),
                                           n_frequencies=1024, dtype='float64', device=device)
        targets = ([[20, 22, 48], [50, 22, 63], [80, 22, 48]] if name == 'folded'
                   else [[20, 22, 48], [40, 22, 60], [30, 22, 57]])
        indices = [np.argmin(np.linalg.norm(graph.positions - target, axis=1)) for target in targets]
        # Choose reproducible synthetic locations in the unit, then deliberately
        # move off voxel centers. These are the actual locations being honored.
        positions = graph.positions[indices] + [.5, 0, .3]
        observations = embedding.query(graph, positions)
        data, noise = np.array([2.80, 2.49, 2.72]), np.array([0., 0., .03 ** 2])
        conditioned = simulator.condition(observations, data, mean=2.65, noise_variance=noise)
        prior = simulator.simulate(realizations=realizations, seed=13, mean=2.65)
        draws = conditioned.simulate(realizations=realizations, seed=13)
        moments = conditioned.moments()
        exact_error = np.max(np.abs(draws.observation_values.cpu().numpy()[:, noise == 0] - data[noise == 0]))
        unobserved = ~np.isin(embedding.component_ids, observations.component_ids)
        unchanged = (float(np.max(np.abs(draws.values[:, unobserved].numpy() - prior.values[:, unobserved].numpy())))
                     if unobserved.any() else None)
        report = dict(geometry=name, realizations=realizations, prior_mean=2.65, prior_standard_deviation=.1,
                      observation_positions=positions.tolist(), observation_values=data.tolist(),
                      observation_noise_variance=noise.tolist(), observation_components=observations.component_ids.tolist(),
                      exact_observation_max_error=float(exact_error),
                      noisy_observation_latent_std=float(draws.observation_values[:, 2].std().cpu()) if realizations > 1 else None,
                      untouched_component_nodes=int(unobserved.sum()), untouched_component_max_change=unchanged,
                      conditioning=conditioned.diagnostics,
                      quality=embedding_quality(graph, embedding, length_scale=5))
        cases.append(ConditioningCase(model, graph, prior, draws, moments, positions, data, noise, report))
    return cases


def plot_sections(cases):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(len(cases), 4, figsize=(18, 4 * len(cases)), squeeze=False, constrained_layout=True)
    titles = ('Prior draw', 'Conditioned, same seed', 'Posterior mean', 'Posterior standard deviation')
    for row, case in enumerate(cases):
        mean, variance = case.moments.to_grids()
        arrays = (case.prior.to_grid(), case.conditional.to_grid(), mean, np.sqrt(variance))
        grid = case.model.grid.dense_grid
        y = int(grid.resolution[1]) // 2
        extent = np.asarray(grid.extent)[[0, 1, 4, 5]]
        for column, (axis, array) in enumerate(zip(axes[row], arrays)):
            section = array[:, y, :].T
            uncertainty = column == 3
            image = axis.imshow(section, origin='lower', extent=extent, interpolation='nearest',
                                cmap='magma' if uncertainty else 'viridis',
                                vmin=0 if uncertainty else 2.35, vmax=.1 if uncertainty else 2.95)
            axis.contour(grid.x_coord, grid.z_coord, np.isfinite(section), levels=[.5], colors='#333333', linewidths=.6)
            for noisy, marker in ((False, 'o'), (True, '^')):
                selected = (case.noise_variance > 0) == noisy
                axis.scatter(case.positions[selected, 0], case.positions[selected, 2], marker=marker,
                             c='white', edgecolors='black', s=55, zorder=5)
            if case.report['geometry'] == 'faulted':
                axis.axvline(50, color='#c44343', lw=1.2)
            axis.set(title=f'{case.report["geometry"]}: {titles[column]}', xlabel='X', ylabel='Z', aspect='equal')
            if column in (2, 3):
                figure.colorbar(image, ax=axis, shrink=.7, label='Density SD' if uncertainty else 'Density')
    handles = [Line2D([], [], marker=marker, color='none', markerfacecolor='white', markeredgecolor='black', label=label)
               for marker, label in [('o', 'Exact measurements'), ('^', 'Noisy measurement (SD 0.03)')]]
    figure.legend(handles=handles, loc='outside lower center', ncol=2)
    figure.suptitle('Continuous density conditioning | 5:1 bedding anisotropy | prior 2.65 +/- 0.10')
    return figure


def plot_volume(case, *, off_screen=False):
    from gempy_plugins.optional_dependencies import require_pyvista
    from .experiment_views import _plotter, _active_mesh, _outline, _cameras, _cutaway_toggle, INK

    pv = require_pyvista()
    title = f'GemPy - Continuous conditioning ({case.report["geometry"]})'
    plotter = _plotter(title, off_screen)
    arrays = (case.prior.values[0].numpy(), case.conditional.values[0].numpy(),
              case.moments.mean.numpy(), np.sqrt(case.moments.variance.numpy()))
    titles = ('PRIOR: matched random draw', 'CONDITIONED: two exact, one noisy',
              'POSTERIOR MEAN: influence follows bedding', 'UNCERTAINTY: blocked by the fault')
    if case.report['geometry'] != 'faulted':
        titles = (*titles[:3], 'UNCERTAINTY: reduced near observations')
    actors = []
    for panel, (values, heading) in enumerate(zip(arrays, titles)):
        plotter.subplot(panel // 2, panel % 2)
        mesh = _active_mesh(case.model, case.prior.embedding, values, 'value')
        clipped = mesh.clip(normal='y', origin=case.positions[0], invert=False)
        actor = plotter.add_mesh(clipped, scalars='value', preference='cell', lighting=False,
                                 cmap='magma' if panel == 3 else 'viridis',
                                 clim=(0, .1) if panel == 3 else (2.35, 2.95), show_scalar_bar=False)
        actors.append((mesh, actor))
        plotter.add_scalar_bar(title=f'{"Density SD" if panel == 3 else "Density"} - panel {panel + 1}',
                               mapper=actor.mapper, color=INK, position_x=.20, position_y=.06,
                               width=.60, height=.09, title_font_size=13, label_font_size=11)
        for position, datum, noise in zip(case.positions, case.data, case.noise_variance):
            marker = pv.Sphere(radius=1.3, center=position)
            plotter.add_mesh(marker, color='#e7a73d' if noise else 'white', lighting=False)
        if panel == 1:
            labels = [f'{value:.2f}' + (' (noisy)' if noise else '') for value, noise in zip(case.data, case.noise_variance)]
            plotter.add_point_labels(case.positions, labels, font_size=12, text_color=INK,
                                     point_size=0, always_visible=True, shape_opacity=.7)
        if case.report['geometry'] == 'faulted':
            plane = pv.Plane(center=(50, 20, 50), direction=(1, 0, 0), i_size=80, j_size=40)
            plotter.add_mesh(plane, color='#c44343', opacity=.12, lighting=False)
            plotter.add_mesh(plane.outline(), color='#c44343', line_width=2)
        plotter.add_text(heading + '\nWhite: exact data. Gold: noisy datum.', position='upper_left', font_size=12, color=INK)
        _outline(plotter, mesh)
    fit_camera = _cameras(plotter)
    _cutaway_toggle(plotter, actors, cut_y=float(case.positions[0, 1]))
    return plotter, title, fit_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default='folded,faulted')
    parser.add_argument('--realizations', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--volume', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--report')
    args = parser.parse_args()
    cases = run_demo(args.models.split(','), realizations=args.realizations, device=args.device)
    text = json.dumps([case.report for case in cases], indent=2, allow_nan=False)
    print(text, flush=True)
    if args.report:
        with open(args.report, 'w') as stream:
            stream.write(text + '\n')
    if args.volume:
        plotter, title, fit_camera = plot_volume(cases[-1], off_screen=args.no_show)
        if args.no_show:
            plotter.show(screenshot=args.output, interactive=False)
        else:
            plotter.show(interactive_update=True, auto_close=False)
            plotter.update(stime=200, force_redraw=True)
            # Native window tiling can replace the initial viewport after show().
            fit_camera(reset=True)
            plotter.render()
            if args.output:
                plotter.screenshot(args.output)
            print(f'WINDOW_OPEN: {title}; mapped={plotter.ren_win.GetMapped()}', flush=True)
            try:
                plotter.iren.start()
            finally:
                plotter.close()
    else:
        if args.no_show:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure = plot_sections(cases)
        if args.output:
            figure.savefig(args.output, dpi=160)
        if not args.no_show:
            plt.show()
        plt.close(figure)


if __name__ == '__main__':
    main()
