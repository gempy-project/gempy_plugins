"""Continuous density on the combination model's conforming tetrahedral mesh.

Uses the same geometry and observation centroids as the facies example. Density
is Gaussian in g/cm^3, with prior mean 2.65 and SD 0.10. Each selected unit has
two exact synthetic measurements (2.80, 2.49) and one noisy measurement (2.72,
measurement SD 0.03). Posterior uncertainty describes the true density field,
excluding measurement noise. Values sample centroids, not cell-volume averages.

    python -m gempy_plugins.property_estimation.examples.tetrahedral_density
    python -m gempy_plugins.property_estimation.examples.tetrahedral_density \
        --no-show --output combined_density.png --report combined_density.json \
        --mesh-output combined_density.vtu

MMG3D is used by default (mmgpy); --no-remesh uses the cut mesh. GemPy downloads/
caches its combination-model data. Controls: drag/scroll, C cutaway, E edges,
Q close. Graph-relative geometry accuracy is reported independently of density
conditioning accuracy. This models density directly, not as a facies lookup.
"""
import argparse
from dataclasses import dataclass
import json

import numpy as np

from gempy_plugins.volume_mesh import to_pyvista
from .tetrahedral_combination import prepare_geometry, synthetic_observation_nodes
from ..gaussian_fields import GaussianCovariance, GaussianFieldSimulator


@dataclass
class TetrahedralDensityCase:
    model: object
    mesh: object
    graph: object
    embedding: object
    prior: object
    conditional: object
    moments: object
    observations: np.ndarray
    observation_values: np.ndarray
    noise_variance: np.ndarray
    cell_data: dict
    report: dict


def run_demo(*, cells=1500, remesh=True, realizations=64, seed=13, units=('rock2', 'rock1'),
             length_scale=60., anisotropy_ratio=4., landmarks=48, dimensions=3,
             n_frequencies=512, gradient_step=.01, model=None):
    geometry = prepare_geometry(cells=cells, remesh=remesh, units=units, length_scale=length_scale,
                                anisotropy_ratio=anisotropy_ratio, landmarks=landmarks, dimensions=dimensions,
                                gradient_step=gradient_step, model=model)
    mesh, graph, embedding = geometry.mesh, geometry.graph, geometry.embedding
    simulator = GaussianFieldSimulator(embedding, GaussianCovariance(length_scale, .10 ** 2),
                                       dtype='float64', n_frequencies=n_frequencies)
    observations = synthetic_observation_nodes(geometry)
    values = np.tile([2.80, 2.49, 2.72], len(geometry.unit_indices))
    noise = np.tile([0., 0., .03 ** 2], len(geometry.unit_indices))
    posterior = simulator.condition(observations, values, mean=2.65, noise_variance=noise)
    prior = simulator.simulate(realizations=realizations, seed=seed, mean=2.65)
    conditional = posterior.simulate(realizations=realizations, seed=seed)
    moments = posterior.moments()
    mean, variance = moments.to_grids()  # Full-cell arrays, NaN outside the active units.
    cell_data = dict(geometry.cell_data, prior_density=prior.to_cells(),
                     conditioned_density=conditional.to_cells(), posterior_mean_density=mean,
                     posterior_variance_density=variance, posterior_sd_density=np.sqrt(variance))
    observed = np.full(len(mesh.tetrahedra), np.nan)
    observed_noise = np.full(len(mesh.tetrahedra), np.nan)
    observation_cells = graph.node_indices[observations]
    observed[observation_cells], observed_noise[observation_cells] = values, noise
    cell_data.update(observed_density=observed, observation_noise_variance=observed_noise)
    untouched = ~np.isin(graph.component_ids, graph.component_ids[observations])
    exact = noise == 0
    error = float(np.max(np.abs(conditional.values.numpy()[:, observations[exact]] - values[exact])))
    change = (float(np.max(np.abs(conditional.values.numpy()[:, untouched] - prior.values.numpy()[:, untouched])))
              if untouched.any() else None)
    report = dict(geometry.report, property='density', density_units='g/cm^3', prior_mean=2.65,
                  prior_standard_deviation=.10, realizations=realizations, seed=seed,
                  observation_tetrahedra=observation_cells.tolist(),
                  observation_positions=graph.positions[observations].tolist(), observation_values=values.tolist(),
                  observation_noise_variance=noise.tolist(), exact_observations=int(exact.sum()),
                  noisy_observations=int((~exact).sum()), exact_observation_max_error=error,
                  noisy_observation_posterior_sd=np.sqrt(variance[observation_cells[~exact]]).tolist(),
                  untouched_component_cells=int(untouched.sum()), untouched_component_max_change=change,
                  conditioning=posterior.diagnostics)
    return TetrahedralDensityCase(geometry.model, mesh, graph, embedding, prior, conditional, moments,
                                  observations, values, noise, cell_data, report)


def as_pyvista(case):
    grid, boundary = to_pyvista(case.mesh)
    for name, values in case.cell_data.items():
        grid.cell_data[name] = values
    grid.field_data['density_units'] = np.array(['g/cm^3'])
    grid.field_data['unit_names'] = np.asarray(case.mesh.unit_names)
    return grid, boundary


def plot_volume(case, *, off_screen=False):
    from gempy_plugins.optional_dependencies import require_pyvista
    from .experiment_views import _plotter, _outline, _cameras, _cutaway_toggle, INK

    pv = require_pyvista()
    title = 'GemPy - Combined model: tetrahedral density conditioning'
    plotter = _plotter(title, off_screen)
    full, boundary = as_pyvista(case)
    active = full.extract_cells(case.graph.node_indices)
    origin = (0, case.report['cut_y'], 0)
    fault = boundary.extract_cells(np.isin(case.mesh.boundary_kinds, ('fault', 'fault_internal')))
    contacts = boundary.slice(normal='y', origin=origin)
    positions = case.graph.positions[case.observations]
    radius = .008 * np.linalg.norm(np.ptp(case.mesh.nodes, axis=0))
    panels = [('prior_density', 'PRIOR DENSITY: seed-matched realization'),
              ('conditioned_density', 'CONDITIONED DENSITY: exact + noisy measurements'),
              ('posterior_mean_density', 'POSTERIOR MEAN: analytical Gaussian posterior'),
              ('posterior_sd_density', 'UNCERTAINTY: posterior standard deviation')]
    quality = case.report['embedding_quality']
    status = 'pass' if quality['passes_provisional_targets'] else 'outside targets'
    actors, cell_actors, contact_actors = [], [], []
    for panel, (array, heading) in enumerate(panels):
        plotter.subplot(panel // 2, panel % 2)
        mesh = active.copy()
        mesh.set_active_scalars(array, preference='cell')
        actor = plotter.add_mesh(mesh.clip(normal='y', origin=origin, invert=False), scalars=array,
                                 preference='cell', lighting=False, show_scalar_bar=False,
                                 cmap='magma' if panel == 3 else 'viridis',
                                 clim=(0, .10) if panel == 3 else (2.35, 2.95),
                                 edge_color='#48505a', line_width=.35)
        actors.append((mesh, actor))
        cell_actors.append(actor)
        plotter.add_scalar_bar(title=f'{"Density SD" if panel == 3 else "Density"} [g/cm^3] - {panel + 1}',
                              mapper=actor.mapper, color=INK, position_x=.20, position_y=.06,
                              width=.60, height=.09, title_font_size=13, label_font_size=11)
        if fault.n_cells:
            fault_actor = plotter.add_mesh(fault.clip(normal='y', origin=origin, invert=False),
                                          color='#b63232', opacity=.22, lighting=False)
            actors.append((fault, fault_actor))
        if contacts.n_cells:
            contact_actors.append(plotter.add_mesh(contacts, color='#303a46', line_width=2,
                                                   lighting=False, render_lines_as_tubes=True))
        for position, noise in zip(positions, case.noise_variance):
            plotter.add_mesh(pv.Sphere(radius=radius, center=position), color='#202632', lighting=False)
            plotter.add_mesh(pv.Sphere(radius=.78 * radius, center=position + [0, -.55 * radius, 0]),
                             color='#e7a73d' if noise else 'white', lighting=False)
        if panel == 1:
            labels = [f'{value:.2f}' + (' (noisy)' if noise else '')
                      for value, noise in zip(case.observation_values, case.noise_variance)]
            plotter.add_point_labels(positions, labels, font_size=11, text_color=INK,
                                     point_size=0, always_visible=True, shape_opacity=.65)
        subtitle = (f'{full.n_cells:,} tets | prior 2.65 +/- 0.10 g/cm^3 | C cutaway, E edges' if panel == 0 else
                    'White: exact; gold: measurement SD 0.03 g/cm^3' if panel == 1 else
                    'Independent across units and fault blocks' if panel == 2 else
                    f'Geometry: {status}; graph-relative covariance P90 {quality["covariance_absolute_p90"]:.3f}')
        plotter.add_text(heading + '\n' + subtitle, position='upper_left', font_size=11, color=INK)
        _outline(plotter, full)
    fit_camera = _cameras(plotter, bounds=full.bounds)
    _cutaway_toggle(plotter, actors, cut_y=case.report['cut_y'])
    state = dict(edges=False, cut=True)

    def toggle_edges():
        state['edges'] = not state['edges']
        for actor in cell_actors:
            actor.prop.show_edges = state['edges']
        plotter.render()

    def toggle_contacts():
        state['cut'] = not state['cut']
        for actor in contact_actors:
            actor.visibility = state['cut']
        plotter.render()

    plotter.add_key_event('e', toggle_edges)
    plotter.add_key_event('c', toggle_contacts)
    return plotter, title, fit_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=1500)
    parser.add_argument('--no-remesh', action='store_true')
    parser.add_argument('--units', nargs='+', default=['rock2', 'rock1'])
    parser.add_argument('--realizations', type=int, default=64)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--length-scale', type=float, default=60.)
    parser.add_argument('--anisotropy-ratio', type=float, default=4.)
    parser.add_argument('--landmarks', type=int, default=48)
    parser.add_argument('--dimensions', type=int, default=3)
    parser.add_argument('--frequencies', type=int, default=512)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--report')
    parser.add_argument('--mesh-output')
    args = parser.parse_args()
    case = run_demo(cells=args.cells, remesh=not args.no_remesh, units=tuple(args.units),
                    realizations=args.realizations, seed=args.seed, length_scale=args.length_scale,
                    anisotropy_ratio=args.anisotropy_ratio, landmarks=args.landmarks,
                    dimensions=args.dimensions, n_frequencies=args.frequencies)
    text = json.dumps(case.report, indent=2, allow_nan=False)
    print(text, flush=True)
    if args.report:
        with open(args.report, 'w') as stream:
            stream.write(text + '\n')
    if args.mesh_output:
        from pathlib import Path
        path = Path(args.mesh_output).with_suffix('.vtu')
        grid, boundary = as_pyvista(case)
        grid.save(path)
        boundary.save(path.with_name(path.stem + '_boundary.vtp'))
    if not args.no_show or args.output:
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


if __name__ == '__main__':
    main()
