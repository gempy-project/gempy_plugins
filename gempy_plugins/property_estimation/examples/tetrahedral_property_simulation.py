"""GemPy -> tetrahedra -> centroid gradients -> Isomap -> conditioned porosity.

One geological solve supplies a reusable evaluator for both meshing and gradient
queries. Everything after meshing is cell-centered. No voxel gradients are used.
The offline faulted model has folded bedding and a barrier at X=50 (no imposed
throw). Three synthetic observations are placed at actual tetrahedron centroids.

Run interactively:
    python -m gempy_plugins.property_estimation.examples.tetrahedral_property_simulation

Export a reproducible headless run:
    python -m gempy_plugins.property_estimation.examples.tetrahedral_property_simulation \
        --no-show --output tetrahedral.png --report tetrahedral.json --mesh-output tetrahedral.vtu

Porosity is an example-specific logit-normal transform: the latent Gaussian has
mean logit(0.20) and SD 0.35, so 0.20 is the prior *median*. Measurements are exact
in porosity and transformed before conditioning. Output values sample centroids,
not cell-volume averages. The face-only graph and Isomap remain experimental;
their graph-relative quality is reported, not assumed to pass acceptance gates.
"""
import argparse
from dataclasses import dataclass
import json

import gempy as gp
import numpy as np
from scipy.special import expit, logit, ndtri

from gempy_plugins.volume_mesh import volume_mesh_from_evaluator, to_pyvista
from .geological_models import make_geological_model
from ..gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from ..geological_coordinates import AnisotropicGraph, GeologicalEmbedding
from ..geological_coordinates.diagnostics import embedding_quality
from ..geological_coordinates.tetrahedral import evaluate_tetrahedral_fields


@dataclass
class TetrahedralPropertyCase:
    model: object
    mesh: object
    graph: AnisotropicGraph
    embedding: GeologicalEmbedding
    cell_data: dict
    observations: np.ndarray
    observation_porosity: np.ndarray
    report: dict


def run_demo(*, geometry='faulted', cells=800, remesh=False, realizations=16, seed=13,
             landmarks=32, dimensions=3, n_frequencies=512, gradient_step=.001, model=None):
    # 1. Solve once. The small solve grid is not a property grid; gradients there
    # are deliberately disabled. Passing a computed model reuses its solution.
    if model is None:
        model = make_geological_model(geometry, resolution=(8, 4, 8), compute_gradients=False)
    evaluator = gp.prepare_field_evaluator(model)

    # 2. The public evaluator accepts world XYZ. Giving the mesher the world
    # extent makes the entire mesh construction use that same coordinate system.
    extent = np.asarray(model.grid.regular_grid.extent, dtype=float)
    mesh = volume_mesh_from_evaluator(
        evaluator, extent, cells=cells, snap=.15, remesh=remesh,
        unit_values=np.asarray(model.taped_interpolation_input.unit_values),
        number_of_surfaces_per_stack=model.input_data_descriptor.stack_structure.number_of_surfaces_per_stack,
    )
    mesh.report['coordinates'] = 'model'
    names = list(model.structural_frame.elements_names)
    mesh.unit_names = [names[i - 1] for i in mesh.unit_ids]
    mesh.surface_names = names[:-1]
    unit = mesh.unit_names.index('lower')
    active = mesh.tet_units == unit
    centroids = mesh.nodes[mesh.tetrahedra].mean(axis=1)

    # 3. A second *evaluation* pass: use each cell's stratigraphic stack and
    # fixed fault sides, including at both offsets of every central difference.
    scalar, gradients = evaluate_tetrahedral_fields(mesh, evaluator, mask=active, step=gradient_step)
    selected = np.flatnonzero(active)
    check = np.zeros(len(active), dtype=bool)
    check[selected[np.linspace(0, len(selected) - 1, min(64, len(selected)), dtype=int)]] = True
    _, half_step = evaluate_tetrahedral_fields(mesh, evaluator, mask=check, step=gradient_step / 2)
    denominator = np.linalg.norm(half_step[check], axis=1)
    relative_change = np.linalg.norm(gradients[check] - half_step[check], axis=1) / np.maximum(denominator, 1e-15)

    # 4-5. Bedding-aligned tensors, smoothly increasing local range, a shared-face
    # graph with impermeable unit/block boundaries, then component-wise Isomap.
    continuity = .8 + .4 * (centroids[:, 0] - extent[0]) / (extent[1] - extent[0])
    graph = AnisotropicGraph.from_tetrahedral_mesh(
        mesh, gradients, mask=active, anisotropy_ratio=5., continuity_scale=continuity,
    )
    graph.metric.metadata.update(gradient_source='prepared_evaluator_central_difference', gradient_step=gradient_step)
    embedding = GeologicalEmbedding.from_graph(graph, landmarks=landmarks, dimensions=dimensions, seed=0)

    # 6. Simulate and condition a Gaussian in geological coordinates. The logit
    # transform below is specific to this example, not a general property model.
    simulator = GaussianFieldSimulator(
        embedding, GaussianCovariance(length_scale=5., variance=.35 ** 2),
        dtype='float64', n_frequencies=n_frequencies,
    )
    # Select the leftmost substantial component; the other fault block receives
    # no observations. These are synthetic CELL measurements, not snapped wells.
    components = [c for c in np.unique(graph.component_ids) if np.sum(graph.component_ids == c) >= 3]
    observed_component = min(components, key=lambda c: graph.positions[graph.component_ids == c, 0].mean())
    candidates = np.flatnonzero(graph.component_ids == observed_component)
    observations = []
    for target in ([15., 20., 48.], [30., 20., 58.], [42., 20., 66.]):
        remaining = np.setdiff1d(candidates, observations)
        observations.append(remaining[np.argmin(np.linalg.norm(graph.positions[remaining] - target, axis=1))])
    observations = np.asarray(observations)
    measured_porosity = np.array([.14, .27, .23])
    posterior = simulator.condition(observations, logit(measured_porosity), mean=logit(.20))
    prior = simulator.simulate(realizations=realizations, seed=seed, mean=logit(.20))
    conditional = posterior.simulate(realizations=realizations, seed=seed)
    moments = posterior.moments()
    latent_mean, latent_variance = moments.to_grids()  # 1D full-cell storage for this embedding.
    latent_sd = np.sqrt(np.maximum(latent_variance, 0))
    low, high = expit(latent_mean - ndtri(.95) * latent_sd), expit(latent_mean + ndtri(.95) * latent_sd)
    component_array = np.full(len(active), -1, dtype=int)
    component_array[graph.node_indices] = graph.component_ids
    cell_data = dict(
        active=graph.metric.active_mask.astype(np.uint8), component=component_array,
        scalar_field=scalar, gradient=gradients, bedding_normal=graph.metric.normals,
        continuity_scale=continuity,
        prior_porosity=expit(prior.to_cells()), conditioned_porosity=expit(conditional.to_cells()),
        posterior_median_porosity=expit(latent_mean), posterior_p05_porosity=low,
        posterior_p95_porosity=high, posterior_90_interval_width=high - low,
    )
    cell_observations = np.full(len(active), np.nan)
    cell_observations[graph.node_indices[observations]] = measured_porosity
    cell_data['observed_porosity'] = cell_observations
    untouched = graph.component_ids != observed_component
    exact_error = float(np.max(np.abs(expit(conditional.observation_values.numpy()) - measured_porosity)))
    unchanged = (float(np.max(np.abs(conditional.values.numpy()[:, untouched] - prior.values.numpy()[:, untouched])))
                 if untouched.any() else None)
    report = dict(
        geometry=geometry, background_cells=cells, remeshed=remesh,
        tetrahedra=len(mesh.tetrahedra), active_tetrahedra=len(graph.node_indices),
        mesh_unit_names=mesh.unit_names, fault_blocks=len(mesh.block_sides),
        components=len(np.unique(graph.component_ids)), graph_edges=graph.adjacency.nnz // 2,
        gradient_method='central differences of fixed-weight, fixed-fault-side fields in world coordinates',
        gradient_step=gradient_step, gradient_half_step_relative_change_max=float(relative_change.max()),
        invalid_gradient_count=graph.metric.invalid_gradient_count,
        anisotropy_ratio=5., continuity_bounds=[.8, 1.2], gaussian_length_scale=5.,
        prior_median_porosity=.20, latent_standard_deviation=.35, realizations=realizations, seed=seed,
        observation_tetrahedra=graph.node_indices[observations].tolist(),
        observation_positions=graph.positions[observations].tolist(), observation_porosity=measured_porosity.tolist(),
        exact_porosity_observation_max_error=exact_error,
        untouched_component_cells=int(untouched.sum()), untouched_component_max_latent_change=unchanged,
        mesh_quality=mesh.report['quality'], closed_volumes=mesh.report['complex']['all_closed'],
        embedding_quality=embedding_quality(graph, embedding, length_scale=5., samples=16),
    )
    return TetrahedralPropertyCase(model, mesh, graph, embedding, cell_data, observations, measured_porosity, report)


def as_pyvista(case):
    """Keep the original tetrahedron order and attach full-cell property arrays."""
    grid, boundary = to_pyvista(case.mesh)
    for name, values in case.cell_data.items():
        grid.cell_data[name] = values
    return grid, boundary


def plot_volume(case, *, off_screen=False):
    from gempy_plugins.optional_dependencies import require_pyvista
    from .experiment_views import _plotter, _outline, _cameras, _cutaway_toggle, INK

    pv = require_pyvista()
    title = 'GemPy - Tetrahedral porosity: fixed-interpolant centroid gradients'
    plotter = _plotter(title, off_screen)
    full_grid, _ = as_pyvista(case)
    grid = full_grid.extract_cells(case.graph.node_indices)
    points = case.graph.positions[case.observations]
    actors = []
    panels = [('prior_porosity', 'PRIOR: bounded porosity', (.08, .35)),
              ('conditioned_porosity', 'CONDITIONED: three exact cell observations', (.08, .35)),
              ('posterior_median_porosity', 'POSTERIOR MEDIAN: logit-normal porosity', (.08, .35)),
              ('posterior_90_interval_width', 'UNCERTAINTY: 90% interval width', (0., .19))]
    quality = case.report['embedding_quality']
    status = 'PASS' if quality['passes_provisional_targets'] else 'FAIL'
    for panel, (array, heading, limits) in enumerate(panels):
        plotter.subplot(panel // 2, panel % 2)
        displayed = grid.copy()
        displayed.set_active_scalars(array, preference='cell')
        clipped = displayed.clip(normal='y', origin=(50, 20, 50), invert=False)
        actor = plotter.add_mesh(clipped, scalars=array, preference='cell', lighting=False,
                                 cmap='magma' if panel == 3 else 'viridis', clim=limits, show_scalar_bar=False)
        actors.append((displayed, actor))
        plotter.add_scalar_bar(title='90% width' if panel == 3 else f'Porosity - panel {panel + 1}',
                              mapper=actor.mapper, color=INK, position_x=.20, position_y=.06,
                              width=.60, height=.09, title_font_size=13, label_font_size=11)
        for point in points:
            plotter.add_mesh(pv.Sphere(radius=1., center=point), color='white', lighting=False)
        if panel == 1:
            plotter.add_point_labels(points, [f'{p:.2f}' for p in case.observation_porosity],
                                     font_size=12, text_color=INK, point_size=0, always_visible=True)
        if case.report['fault_blocks'] > 1:
            plane = pv.Plane(center=(50, 20, 50), direction=(1, 0, 0), i_size=80, j_size=40)
            plotter.add_mesh(plane, color='#c44343', opacity=.12, lighting=False)
        plotter.add_text(heading + f'\n{len(case.graph.node_indices):,} active tetrahedra | geometry gates: {status}',
                         position='upper_left', font_size=11, color=INK)
        _outline(plotter, grid)
    fit_camera = _cameras(plotter)
    _cutaway_toggle(plotter, actors, cut_y=20.)
    return plotter, title, fit_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', choices=('folded', 'faulted'), default='faulted')
    parser.add_argument('--cells', type=int, default=800, help='Approximate number of background cubes')
    parser.add_argument('--remesh', action='store_true', help='Improve tetrahedra with optional mmgpy')
    parser.add_argument('--realizations', type=int, default=16)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--landmarks', type=int, default=32)
    parser.add_argument('--dimensions', type=int, default=3)
    parser.add_argument('--frequencies', type=int, default=512)
    parser.add_argument('--gradient-step', type=float, default=.001)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output', help='PNG screenshot')
    parser.add_argument('--report', help='JSON diagnostics')
    parser.add_argument('--mesh-output', help='VTU tetrahedra with properties, plus boundary VTP')
    args = parser.parse_args()
    case = run_demo(geometry=args.geometry, cells=args.cells, remesh=args.remesh, realizations=args.realizations,
                    seed=args.seed, landmarks=args.landmarks, dimensions=args.dimensions,
                    n_frequencies=args.frequencies, gradient_step=args.gradient_step)
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
