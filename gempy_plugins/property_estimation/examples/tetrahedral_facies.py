"""Conditional facies in GemPy's combination model: folds, fault and unconformity.

Remeshed tetrahedra conform to the geological interfaces. Facies are simulated
inside rock2 and rock1, below the younger rock3 unit that truncates the folds.
One solved interpolant supplies both the mesh and fixed-fault-side centroid
gradients (central differences in world coordinates). Six synthetic categorical
observations constrain latent intervals, not fixed Gaussian values.

    python -m gempy_plugins.property_estimation.examples.tetrahedral_facies
    python -m gempy_plugins.property_estimation.examples.tetrahedral_facies \
        --no-show --output combined_facies.png --report combined_facies.json \
        --mesh-output combined_facies.vtu

Defaults require mmgpy; --no-remesh uses the original cut tetrahedra. The GemPy
example generator downloads/caches the combination-model input data. Controls:
drag/scroll, C cutaway, E tetrahedral edges, Q close. The graph/Isomap accuracy
report remains separate from categorical observation honoring.
"""
import argparse
from dataclasses import dataclass
import json

import numpy as np

from gempy_plugins.volume_mesh import to_pyvista
from gempy_plugins.volume_mesh.background_grid import signed_volumes
from .tetrahedral_combination import make_combination_model, prepare_geometry, synthetic_observation_nodes
from ..categorical_fields import TruncatedGaussianFacies
from ..gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from ..geological_coordinates import AnisotropicGraph, GeologicalEmbedding


@dataclass
class TetrahedralFaciesCase:
    model: object
    mesh: object
    graph: AnisotropicGraph
    embedding: GeologicalEmbedding
    prior: object
    conditional: object
    observations: np.ndarray
    observed_codes: np.ndarray
    cell_data: dict
    report: dict


def run_demo(*, cells=1500, remesh=True, realizations=128, seed=13, units=('rock2', 'rock1'),
             length_scale=60., anisotropy_ratio=4., landmarks=48, dimensions=3,
             n_frequencies=512, gradient_step=.01, model=None):
    geometry = prepare_geometry(cells=cells, remesh=remesh, units=units, length_scale=length_scale,
                                anisotropy_ratio=anisotropy_ratio, landmarks=landmarks, dimensions=dimensions,
                                gradient_step=gradient_step, model=model)
    mesh, graph, embedding = geometry.mesh, geometry.graph, geometry.embedding
    unit_indices = geometry.unit_indices
    simulator = GaussianFieldSimulator(embedding, GaussianCovariance(length_scale), dtype='float64',
                                       n_frequencies=n_frequencies)
    facies = TruncatedGaussianFacies(simulator, (.35, .25, .40), names=('shale', 'siltstone', 'sandstone'))
    observations = synthetic_observation_nodes(geometry)
    codes = np.tile([2, 0, 1], len(unit_indices))
    prior = facies.simulate(realizations=realizations, seed=seed)
    conditioned = facies.condition(observations, codes, sampler='rejection')
    conditional = conditioned.simulate(realizations=realizations, seed=seed)
    probabilities = conditional.probabilities().numpy()
    prior_probabilities = prior.probabilities().numpy()
    observed = np.full(len(mesh.tetrahedra), -1, dtype=int)
    observed[graph.node_indices[observations]] = codes
    cell_data = dict(geometry.cell_data,
                     prior_facies=prior.to_cells(), conditioned_facies=conditional.to_cells(), observed_facies=observed)
    for code, name in enumerate(facies.names):
        full = np.full(len(mesh.tetrahedra), np.nan)
        full[graph.node_indices] = probabilities[code]
        cell_data['probability_' + name] = full
    untouched = ~np.isin(graph.component_ids, graph.component_ids[observations])
    # Irregular tetrahedra require volume weighting, not equal cell-count weights.
    volumes = signed_volumes(mesh.nodes, mesh.tetrahedra)[graph.node_indices]
    fractions = {}
    for unit, name in zip(unit_indices, units):
        selected = mesh.tet_units[graph.node_indices] == unit
        weights = volumes[selected] / volumes[selected].sum()
        fractions[name] = dict(prior=(prior_probabilities[:, selected] @ weights).tolist(),
                               conditional=(probabilities[:, selected] @ weights).tolist())
    report = dict(geometry.report,
        facies_names=list(facies.names), prior_proportions=list(facies.proportions),
        realizations=realizations, seed=seed, sampler='rejection',
        probability_max_monte_carlo_standard_error=float(.5 / np.sqrt(realizations)),
        observation_tetrahedra=graph.node_indices[observations].tolist(),
        observation_positions=graph.positions[observations].tolist(), observation_codes=codes.tolist(),
        observation_names=[facies.names[c] for c in codes],
        honored_fraction=float(np.mean(conditional.values.numpy()[:, observations] == codes)),
        untouched_component_cells=int(untouched.sum()),
        untouched_component_changed_codes=int(np.count_nonzero(
            conditional.values.numpy()[:, untouched] != prior.values.numpy()[:, untouched])),
        ensemble_volume_fractions_by_unit=fractions,
    )
    return TetrahedralFaciesCase(geometry.model, mesh, graph, embedding, prior, conditional,
                                 observations, codes, cell_data, report)


def as_pyvista(case):
    grid, boundary = to_pyvista(case.mesh)
    for name, values in case.cell_data.items():
        grid.cell_data[name] = values
    grid.field_data['facies_names'] = np.asarray(case.prior.names)
    grid.field_data['unit_names'] = np.asarray(case.mesh.unit_names)
    return grid, boundary


def plot_volume(case, *, off_screen=False):
    from matplotlib.colors import ListedColormap
    from gempy_plugins.optional_dependencies import require_pyvista
    from .experiment_views import _plotter, _outline, _cameras, _cutaway_toggle, FACIES_COLORS, INK, BACKGROUND

    pv = require_pyvista()
    title = 'GemPy - Combined model: tetrahedral facies, fault and unconformity'
    plotter = _plotter(title, off_screen)
    full, boundary = as_pyvista(case)
    active = full.extract_cells(case.graph.node_indices)
    cut_y = case.report['cut_y']
    origin = (0, cut_y, 0)
    fault = boundary.extract_cells(np.isin(case.mesh.boundary_kinds, ('fault', 'fault_internal')))
    section_contacts = boundary.slice(normal='y', origin=origin)
    unit_colors = ['#c9a882', '#8598b8', '#b4caa0', '#ddd5ca']
    headings = ('GEOLOGY: conforming tetrahedra', 'PRIOR FACIES: before conditioning',
                'CONDITIONED FACIES: six exact cell observations', 'SANDSTONE PROBABILITY: conditional ensemble')
    if len(case.observations) != 6:
        headings = (*headings[:2], f'CONDITIONED FACIES: {len(case.observations)} exact observations', headings[3])
    arrays = ('unit', 'prior_facies', 'conditioned_facies', 'probability_sandstone')
    actors, cell_actors, contact_actors = [], [], []
    positions = case.graph.positions[case.observations]
    radius = .008 * np.linalg.norm(np.ptp(case.mesh.nodes, axis=0))
    quality = case.report['embedding_quality']
    status = 'pass' if quality['passes_provisional_targets'] else 'outside targets'
    for panel, (array, heading) in enumerate(zip(arrays, headings)):
        plotter.subplot(panel // 2, panel % 2)
        mesh = full.copy() if panel == 0 else active.copy()
        mesh.set_active_scalars(array, preference='cell')
        options = dict(scalars=array, preference='cell', lighting=False, show_scalar_bar=False,
                       show_edges=panel == 0, edge_color='#48505a', line_width=.35)
        if panel == 0:
            options.update(cmap=ListedColormap(unit_colors), clim=(-.5, len(case.mesh.units) - .5),
                           n_colors=len(case.mesh.units))
            legend = list(zip(case.mesh.unit_names, unit_colors))
        elif panel in (1, 2):
            options.update(cmap=ListedColormap(FACIES_COLORS), clim=(-.5, 2.5), n_colors=3)
            legend = list(zip(case.prior.names, FACIES_COLORS))
        else:
            options.update(cmap='viridis', clim=(0, 1))
            legend = None
        actor = plotter.add_mesh(mesh.clip(normal='y', origin=origin, invert=False), **options)
        actors.append((mesh, actor))
        cell_actors.append(actor)
        if legend:
            plotter.add_legend(labels=legend, bcolor=BACKGROUND, face='rectangle', loc='lower left', size=(.24, .19))
        else:
            plotter.add_scalar_bar(title=f'P(sandstone) | {case.report["realizations"]} draws', mapper=actor.mapper,
                                  color=INK, position_x=.20, position_y=.06, width=.60, height=.09,
                                  title_font_size=13, label_font_size=11)
        if fault.n_cells:
            fault_actor = plotter.add_mesh(fault.clip(normal='y', origin=origin, invert=False),
                                          color='#b63232', opacity=.22, lighting=False)
            actors.append((fault, fault_actor))
        if section_contacts.n_cells:
            contact_actors.append(plotter.add_mesh(section_contacts, color='#303a46', line_width=2,
                                                   lighting=False, render_lines_as_tubes=True))
        if panel in (2, 3):
            for position, code in zip(positions, case.observed_codes):
                plotter.add_mesh(pv.Sphere(radius=radius, center=position), color='#202632', lighting=False)
                plotter.add_mesh(pv.Sphere(radius=.78 * radius, center=position + [0, -.55 * radius, 0]),
                                 color=FACIES_COLORS[code], lighting=False)
        subtitle = (f'{full.n_cells:,} tets | red: actual fault | C cutaway, E edges' if panel == 0 else
                    f'Units: {", ".join(case.report["active_units"])} | independent across boundaries' if panel == 1 else
                    'Dark-ringed dots: observed shale / siltstone / sandstone' if panel == 2 else
                    f'Geometry: {status}; graph-relative covariance P90 {quality["covariance_absolute_p90"]:.3f}')
        plotter.add_text(heading + '\n' + subtitle, position='upper_left', font_size=11, color=INK)
        _outline(plotter, full)
    fit_camera = _cameras(plotter, bounds=full.bounds)
    _cutaway_toggle(plotter, actors, cut_y=cut_y)
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
    parser.add_argument('--realizations', type=int, default=128)
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
