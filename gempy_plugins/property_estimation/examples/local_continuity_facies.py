"""Conditional facies with short continuity on the left and long on the right.

Compare uniform ranges, a local range multiplier, a local anisotropy ratio, and
fault-isolated conditioning. --volume opens the range-map case in linked PyVista.
"""
import argparse
from dataclasses import dataclass
import json

import numpy as np

from .geological_models import make_geological_model
from ..categorical_fields import TruncatedGaussianFacies
from ..gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from ..geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding
from ..geological_coordinates.diagnostics import embedding_quality


CASE_NAMES = ('uniform', 'range_map', 'ratio_map', 'faulted_range')


@dataclass
class LocalContinuityCase:
    model: object
    graph: AnisotropicGraph
    simulator: GaussianFieldSimulator
    prior: object
    conditional: object
    observations: np.ndarray
    codes: np.ndarray
    report: dict


def regional_continuity(graph, simulator, fields):
    """Compare equal 8-unit Y lags, tangent to bedding in the extruded examples.

    Prior fields isolate the continuity effect from conditioning. Errors against
    the local endpoint metric, embedded target, and finite basis remain separate.
    """
    mapping = np.full(np.prod(graph.metric.grid_shape), -1, dtype=int)
    mapping[graph.node_indices] = np.arange(len(graph.node_indices))
    mapping = mapping.reshape(graph.metric.grid_shape)
    a, b = mapping[:, :-2, :].ravel(), mapping[:, 2:, :].ravel()
    valid = (a >= 0) & (b >= 0)
    a, b = a[valid], b[valid]
    same = graph.component_ids[a] == graph.component_ids[b]
    a, b = a[same], b[same]
    midpoint = (graph.positions[a, 0] + graph.positions[b, 0]) / 2
    reports = {}
    values = fields.values.cpu().numpy()
    for name, selected in [('left', midpoint < 30), ('right', midpoint > 70)]:
        start, end = a[selected], b[selected]
        physical = np.linalg.norm(graph.positions[start] - graph.positions[end], axis=1)
        metric_distance = graph.metric.edge_costs(graph.node_indices[start], graph.node_indices[end])
        embedded = np.linalg.norm(fields.embedding.coordinates[start] - fields.embedding.coordinates[end], axis=1)
        finite = (simulator._basis(start) * simulator._basis(end)).sum(dim=1).cpu().numpy()
        agreement = np.mean(values[:, start] == values[:, end], axis=1)
        reports[name] = dict(pairs=len(start), physical_lag=float(physical.mean()),
                             intended_local_correlation=float(simulator.covariance(metric_distance).mean()),
                             embedded_correlation=float(simulator.covariance(embedded).mean()),
                             finite_feature_correlation=float(finite.mean()),
                             prior_same_facies_probability=float(agreement.mean()),
                             agreement_standard_error=float(agreement.std(ddof=1) / np.sqrt(len(agreement))) if len(agreement) > 1 else None)
    return reports


def run_demo(cases=CASE_NAMES, *, realizations=128, dimensions=3, stencil_radius=3, seed=13, device='cpu',
             continuity_bounds=(.6, 1.4), anisotropy_bounds=(2., 8.), landmark_seed=0):
    for bounds, minimum in ((continuity_bounds, 0.), (anisotropy_bounds, 1.)):
        bounds = np.asarray(bounds)
        if (bounds.shape != (2,) or not np.isfinite(bounds).all() or np.any(bounds <= 0)
                or np.any(bounds < minimum)):
            raise ValueError("Bounds must contain two finite positive endpoint scales, with anisotropy ratios >= 1")
    results = []
    models = {}
    for name in cases:
        if name not in CASE_NAMES:
            raise ValueError(f"Unknown case {name!r}; choose from {CASE_NAMES}")
        geometry = 'faulted' if name == 'faulted_range' else 'folded'
        if geometry not in models:
            models[geometry] = make_geological_model(geometry)
        model = models[geometry]
        x = model.grid.dense_grid.values[:, 0]
        t = np.clip((x - 20) / 60, 0, 1)
        blend = t * t * (3 - 2 * t)
        scale = (continuity_bounds[0] + (continuity_bounds[1] - continuity_bounds[0]) * blend
                 if name in ('range_map', 'faulted_range') else 1.)
        ratio = anisotropy_bounds[0] + (anisotropy_bounds[1] - anisotropy_bounds[0]) * blend if name == 'ratio_map' else 5.
        metric = GeologicalMetricField.from_gempy(model, 'lower', anisotropy_ratio=ratio, continuity_scale=scale)
        graph = AnisotropicGraph.from_metric(metric, stencil_radius=stencil_radius)
        embedding = GeologicalEmbedding.from_graph(graph, landmarks=32, dimensions=dimensions, seed=landmark_seed)
        simulator = GaussianFieldSimulator(embedding, GaussianCovariance(5), n_frequencies=1024,
                                           dtype='float64', device=device)
        facies = TruncatedGaussianFacies(simulator, (.3, .4, .3), names=('shale', 'limestone', 'sandstone'))
        targets = ([[20, 22, 48], [50, 22, 64], [80, 22, 48]] if geometry == 'folded'
                   else [[20, 22, 48], [40, 22, 62], [30, 22, 56]])
        nodes = np.array([np.argmin(np.linalg.norm(graph.positions - target, axis=1)) for target in targets])
        codes = np.array([2, 0, 2] if geometry == 'folded' else [2, 0, 1])
        prior = facies.simulate(realizations=realizations, seed=seed)
        conditioned = facies.condition(nodes, codes, sampler='rejection')
        draws = conditioned.simulate(realizations=realizations, seed=seed)
        untouched = ~np.isin(embedding.component_ids, embedding.component_ids[nodes])
        changes = int(np.count_nonzero(draws.values[:, untouched].numpy() != prior.values[:, untouched].numpy()))
        report = dict(case=name, geometry=geometry, realizations=realizations, dimensions=dimensions,
                      stencil_radius=stencil_radius, landmark_seed=landmark_seed, simulation_seed=seed,
                      scale_parameters=metric.scale_metadata,
                      observation_positions=graph.positions[nodes].tolist(), observation_codes=codes.tolist(),
                      observation_names=[facies.names[code] for code in codes], sampler='rejection',
                      honored_fraction=float(np.mean(draws.observation_codes.numpy() == codes)),
                      untouched_component_nodes=int(untouched.sum()), untouched_component_changed_codes=changes,
                      quality=embedding_quality(graph, embedding, length_scale=5),
                      regional_prior_continuity=regional_continuity(graph, simulator, prior))
        results.append(LocalContinuityCase(model, graph, simulator, prior, draws, nodes, codes, report))
    return results


def plot_sections(cases):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from .experiment_views import FACIES_COLORS

    figure, axes = plt.subplots(len(cases), 4, figsize=(18, 3.5 * len(cases)), squeeze=False, constrained_layout=True)
    cmap, norm = ListedColormap(FACIES_COLORS), BoundaryNorm(np.arange(4) - .5, 3)
    headings = ('Parallel range (intended)', 'Prior facies', 'Conditioned facies', 'P(sandstone | observations)')
    range_limits = (min(10., min(case.graph.metric.local_scales()[0].min() * 5 for case in cases)),
                    max(40., max(case.graph.metric.local_scales()[0].max() * 5 for case in cases)))
    for row, case in enumerate(cases):
        metric, embedding = case.graph.metric, case.prior.embedding
        parallel, _ = metric.local_scales()
        full = np.full(np.prod(metric.grid_shape), np.nan)
        full[embedding.node_indices] = parallel * 5
        probabilities = full.copy()
        probabilities[embedding.node_indices] = case.conditional.probabilities()[2].numpy()
        arrays = (full.reshape(metric.grid_shape), case.prior.to_grid(), case.conditional.to_grid(),
                  probabilities.reshape(metric.grid_shape))
        grid = case.model.grid.dense_grid
        y = int(grid.resolution[1]) // 2
        extent = np.asarray(grid.extent)[[0, 1, 4, 5]]
        positions = case.graph.positions[case.observations]
        for column, (axis, array) in enumerate(zip(axes[row], arrays)):
            data = array[:, y, :].T
            options = (dict(cmap=cmap, norm=norm) if column in (1, 2) else
                       dict(cmap='viridis', vmin=0, vmax=1) if column == 3 else dict(cmap='plasma', vmin=range_limits[0], vmax=range_limits[1]))
            image = axis.imshow(np.ma.masked_where(data < 0, data), origin='lower', extent=extent,
                                interpolation='nearest', **options)
            if column in (2, 3):
                axis.scatter(positions[:, 0], positions[:, 2], c=np.array(FACIES_COLORS)[case.codes],
                             edgecolors='black', s=75, linewidths=1.4)
            if case.report['geometry'] == 'faulted':
                axis.axvline(50, color='#c44343', lw=1)
            axis.set(title=f'{case.report["case"]}: {headings[column]}', xlabel='X', ylabel='Z', aspect='equal')
            if column in (0, 3):
                figure.colorbar(image, ax=axis, shrink=.65)
            if column == 0:
                quality = case.report['quality']
                status = 'pass' if quality['passes_provisional_targets'] else 'outside targets'
                axis.text(.04, .05, f'Geometry: {status}\nCovariance error P90: {quality["covariance_absolute_p90"]:.3f}',
                          transform=axis.transAxes, fontsize=9, color='#555555')
    figure.suptitle('Local continuity + conditional facies | shale / limestone / sandstone | dots: observed facies')
    return figure


def plot_volume(case, *, off_screen=False):
    from matplotlib.colors import ListedColormap
    from gempy_plugins.optional_dependencies import require_pyvista
    from .experiment_views import _plotter, _active_mesh, _outline, _cameras, _cutaway_toggle, FACIES_COLORS, INK, BACKGROUND

    pv = require_pyvista()
    title = f'GemPy - Local continuity and conditional facies ({case.report["case"]})'
    plotter = _plotter(title, off_screen)
    parallel, _ = case.graph.metric.local_scales()
    values = (parallel * 5, case.prior.values[0].numpy(), case.conditional.values[0].numpy(),
              case.conditional.probabilities()[2].numpy())
    headings = ('LOCAL CONTINUITY: range varies across model', 'PRIOR FACIES: same bedding, variable range',
                'CONDITIONED FACIES: observations honored', 'SANDSTONE PROBABILITY: conditional ensemble')
    actors = []
    positions = case.graph.positions[case.observations]
    for panel, (array, heading) in enumerate(zip(values, headings)):
        plotter.subplot(panel // 2, panel % 2)
        mesh = _active_mesh(case.model, case.prior.embedding, array, 'value')
        options = dict(scalars='value', preference='cell', lighting=False, show_scalar_bar=False)
        if panel in (1, 2):
            options.update(cmap=ListedColormap(FACIES_COLORS), clim=(-.5, 2.5), n_colors=3)
            plotter.add_legend(labels=list(zip(case.prior.names, FACIES_COLORS)), bcolor=BACKGROUND,
                               face='rectangle', loc='lower left', size=(.22, .16))
        else:
            options.update(cmap='plasma' if panel == 0 else 'viridis',
                           clim=(min(10., float(parallel.min() * 5)), max(40., float(parallel.max() * 5))) if panel == 0 else (0, 1))
        actor = plotter.add_mesh(mesh.clip(normal='y', origin=positions[0], invert=False), **options)
        actors.append((mesh, actor))
        if panel in (0, 3):
            plotter.add_scalar_bar(title='Intended parallel range' if panel == 0 else 'P(sandstone)', mapper=actor.mapper,
                                   color=INK, position_x=.20, position_y=.06, width=.60, height=.09,
                                   title_font_size=13, label_font_size=11)
        if panel in (2, 3):
            for position, code in zip(positions, case.codes):
                plotter.add_mesh(pv.Sphere(radius=1.8, center=position), color='black', lighting=False)
                # Shift toward the cut face to keep the category marker visible.
                plotter.add_mesh(pv.Sphere(radius=1.4, center=position + [0, -.8, 0]),
                                 color=FACIES_COLORS[code], lighting=False)
        if case.report['geometry'] == 'faulted':
            plane = pv.Plane(center=(50, 20, 50), direction=(1, 0, 0), i_size=80, j_size=40)
            plotter.add_mesh(plane, color='#c44343', opacity=.12, lighting=False)
        quality = case.report['quality']
        status = 'pass' if quality['passes_provisional_targets'] else 'outside provisional targets'
        subtitle = (f'Geometry: {status}; covariance P90 {quality["covariance_absolute_p90"]:.3f}' if panel == 0
                    else f'{case.report["realizations"]} draws; dots: measured facies' if panel == 3
                    else 'Dark-ringed dots: measured facies' if panel == 2 else 'Shale / limestone / sandstone')
        plotter.add_text(heading + '\n' + subtitle, position='upper_left', font_size=11, color=INK)
        _outline(plotter, mesh)
    fit_camera = _cameras(plotter)
    _cutaway_toggle(plotter, actors, cut_y=float(positions[0, 1]))
    return plotter, title, fit_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default=','.join(CASE_NAMES))
    parser.add_argument('--realizations', type=int, default=128)
    parser.add_argument('--dimensions', type=int, default=3)
    parser.add_argument('--stencil-radius', type=int, default=3)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--landmark-seed', type=int, default=0)
    parser.add_argument('--continuity-bounds', type=float, nargs=2, default=(.6, 1.4), metavar=('LEFT', 'RIGHT'))
    parser.add_argument('--anisotropy-bounds', type=float, nargs=2, default=(2., 8.), metavar=('LEFT', 'RIGHT'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--volume', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--report')
    args = parser.parse_args()
    cases = run_demo(args.cases.split(','), realizations=args.realizations, dimensions=args.dimensions,
                     stencil_radius=args.stencil_radius, seed=args.seed, device=args.device,
                     continuity_bounds=args.continuity_bounds, anisotropy_bounds=args.anisotropy_bounds,
                     landmark_seed=args.landmark_seed)
    text = json.dumps([case.report for case in cases], indent=2, allow_nan=False)
    print(text, flush=True)
    if args.report:
        with open(args.report, 'w') as stream:
            stream.write(text + '\n')
    if args.volume:
        case = next((case for case in cases if case.report['case'] == 'range_map'), cases[0])
        plotter, title, fit_camera = plot_volume(case, off_screen=args.no_show)
        if args.no_show:
            plotter.show(screenshot=args.output, interactive=False)
        else:
            plotter.show(interactive_update=True, auto_close=False)
            plotter.update(stime=200, force_redraw=True)
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
