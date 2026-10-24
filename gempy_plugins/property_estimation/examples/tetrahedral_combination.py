"""Shared combination-model geometry for the tetrahedral density/facies examples."""
from dataclasses import dataclass

import gempy as gp
import numpy as np

from gempy_plugins.volume_mesh import volume_mesh_from_evaluator
from ..geological_coordinates import AnisotropicGraph, GeologicalEmbedding
from ..geological_coordinates.diagnostics import embedding_quality
from ..geological_coordinates.tetrahedral import evaluate_tetrahedral_fields


@dataclass
class CombinedTetrahedralGeometry:
    model: object
    mesh: object
    graph: AnisotropicGraph
    embedding: GeologicalEmbedding
    unit_indices: list
    cell_data: dict
    report: dict


def make_combination_model():
    from gempy.core.data.enumerators import ExampleModel

    model = gp.generate_example_model(ExampleModel.COMBINATION, compute_model=False)
    model.interpolation_options.evaluation_options.mesh_extraction = False
    model.interpolation_options.evaluation_options.compute_scalar_gradient = False
    gp.compute_model(model, engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.numpy, dtype='float64', use_gpu=False,
    ))
    return model


def prepare_geometry(*, cells=1500, remesh=True, units=('rock2', 'rock1'), length_scale=60.,
                     anisotropy_ratio=4., landmarks=48, dimensions=3, gradient_step=.01, model=None):
    model = make_combination_model() if model is None else model
    evaluator = gp.prepare_field_evaluator(model)
    extent = np.asarray(model.grid.regular_grid.extent, dtype=float)
    # World-coordinate evaluator + world extent: this does not perform a solve.
    mesh = volume_mesh_from_evaluator(
        evaluator, extent, cells=cells, snap=.15, remesh=remesh,
        unit_values=np.asarray(model.taped_interpolation_input.unit_values),
        number_of_surfaces_per_stack=model.input_data_descriptor.stack_structure.number_of_surfaces_per_stack,
    )
    mesh.report['coordinates'] = 'model'
    names = list(model.structural_frame.elements_names)
    mesh.unit_names = [names[i - 1] for i in mesh.unit_ids]
    mesh.surface_names = names[:-1]
    if not units or len(set(units)) != len(units) or not set(units).issubset(mesh.unit_names):
        raise ValueError(f"Select unique mesh unit names from {mesh.unit_names}")
    unit_indices = [mesh.unit_names.index(name) for name in units]
    active = np.isin(mesh.tet_units, unit_indices)
    scalar, gradients = evaluate_tetrahedral_fields(mesh, evaluator, mask=active, step=gradient_step)
    selected = np.flatnonzero(active)
    check = np.zeros(len(active), dtype=bool)
    check[selected[np.linspace(0, len(selected) - 1, min(64, len(selected)), dtype=int)]] = True
    _, smaller_step = evaluate_tetrahedral_fields(mesh, evaluator, mask=check, step=gradient_step / 2)
    relative_change = np.linalg.norm(gradients[check] - smaller_step[check], axis=1) / np.maximum(
        np.linalg.norm(smaller_step[check], axis=1), 1e-15)
    graph = AnisotropicGraph.from_tetrahedral_mesh(mesh, gradients, mask=active, anisotropy_ratio=anisotropy_ratio)
    graph.metric.metadata.update(gradient_source='prepared_evaluator_central_difference', gradient_step=gradient_step)
    embedding = GeologicalEmbedding.from_graph(graph, landmarks=landmarks, dimensions=dimensions, seed=0)
    component = np.full(len(active), -1, dtype=int)
    component[graph.node_indices] = graph.component_ids
    cell_data = dict(active=graph.metric.active_mask.astype(np.uint8), component=component,
                     scalar_field=scalar, gradient=gradients, bedding_normal=graph.metric.normals)
    report = dict(
        geometry='combination', active_units=list(units), mesh_unit_names=mesh.unit_names,
        background_cells=cells, remeshed=remesh, tetrahedra=len(mesh.tetrahedra),
        active_tetrahedra=len(graph.node_indices), fault_blocks=len(mesh.block_sides),
        components=len(np.unique(graph.component_ids)), graph_edges=graph.adjacency.nnz // 2,
        cut_y=float((extent[2] + extent[3]) / 2),
        gradient_method='world-coordinate central differences, fixed fault sides',
        gradient_step=gradient_step, gradient_half_step_relative_change_max=float(relative_change.max()),
        invalid_gradient_count=graph.metric.invalid_gradient_count,
        anisotropy_ratio=anisotropy_ratio, gaussian_length_scale=length_scale,
        mesh_quality=mesh.report['quality'], closed_volumes=mesh.report['complex']['all_closed'],
        embedding_quality=embedding_quality(graph, embedding, length_scale=length_scale, samples=24),
    )
    return CombinedTetrahedralGeometry(model, mesh, graph, embedding, unit_indices, cell_data, report)


def synthetic_observation_nodes(geometry):
    """Three spread-out centroids in one component per unit, near the cut face.

    These define new synthetic CELL observations, not a nearest-cell mapping of
    real boreholes. Other components are deliberately left unobserved.
    """
    mesh, graph, embedding = geometry.mesh, geometry.graph, geometry.embedding
    observations = []
    node_units = mesh.tet_units[graph.node_indices]
    width = np.ptp(mesh.nodes[:, 1])
    for unit in geometry.unit_indices:
        candidates = np.flatnonzero(node_units == unit)
        components = [c for c in np.unique(graph.component_ids[candidates])
                      if np.sum(graph.component_ids[candidates] == c) >= 3]
        if not components:
            raise ValueError(f"Unit {mesh.unit_names[unit]} needs at least three connected active tetrahedra")
        # Keep substantial occurrences before selecting the leftmost component.
        sizes = {c: np.sum(graph.component_ids[candidates] == c) for c in components}
        components = [c for c in components if sizes[c] >= .1 * max(sizes.values())]
        component = min(components, key=lambda c: graph.positions[graph.component_ids == c, 0].mean())
        candidates = candidates[graph.component_ids[candidates] == component]
        near_cut = candidates[np.abs(graph.positions[candidates, 1] - geometry.report['cut_y']) < .075 * width]
        if len(near_cut) >= 3:
            candidates = near_cut
        center = np.median(graph.positions[candidates], axis=0)
        first = candidates[np.argmin(np.linalg.norm(graph.positions[candidates] - center, axis=1))]
        chosen = [first]
        for _ in range(2):
            distances = np.linalg.norm(embedding.coordinates[candidates, None]
                                       - embedding.coordinates[chosen], axis=2).min(axis=1)
            distances[np.isin(candidates, chosen)] = -1
            chosen.append(candidates[np.argmax(distances)])
        observations.extend(chosen)
    return np.asarray(observations)
