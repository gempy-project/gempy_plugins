"""Sampled distance, neighborhood and covariance validation without all-pairs storage."""
import numpy as np

from ._utils import positive_integer
from .analytical import bedding_orientation, constant_metric_coordinates


def _validation_sources(graph, embedding, samples, seed):
    if graph.fingerprint != embedding.graph_fingerprint:
        raise ValueError("Embedding and reference graph do not match")
    landmarks = (np.concatenate([p.landmark_indices for p in embedding.projections])
                 if embedding.projections else np.empty(0, dtype=int))
    candidates = np.setdiff1d(np.arange(len(graph.node_indices)), landmarks)
    held_out = bool(len(candidates))
    if not held_out:
        candidates = np.arange(len(graph.node_indices))
    sources = np.random.default_rng(seed).choice(candidates, min(samples, len(candidates)), replace=False)
    return sources, held_out


def embedding_quality(graph, embedding, *, length_scale, samples=32, neighbors=16, seed=1,
                      cartesian_normal=None):
    """Compare held-out-source graph distances against the embedding.

    Local errors cover nonzero distances up to three Gaussian length scales.
    Neighbor overlap accepts ties at the reference kth distance, avoiding an
    arbitrary penalty for equivalent neighbors on a regular lattice. The
    covariance target here is a pairwise diagnostic, not an assertion that a
    Gaussian kernel of arbitrary graph distances forms a PSD covariance matrix.
    The Cartesian comparator uses the dominant unoriented bedding normal unless
    an explicit normal is supplied. Its orientation and spread are reported.
    """
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    samples = positive_integer(samples, 'samples')
    neighbors = positive_integer(neighbors, 'neighbors')
    sources, held_out = _validation_sources(graph, embedding, samples, seed)
    distance_errors, covariance_errors, cartesian_errors, overlaps = [], [], [], []
    orientation = bedding_orientation(graph.metric)
    normal = np.asarray(orientation['normal'] if cartesian_normal is None else cartesian_normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        raise ValueError("cartesian_normal must be a finite, nonzero XYZ vector")
    normal = normal / np.linalg.norm(normal)
    parallel, perpendicular = graph.metric.local_scales()
    # A single representative constant comparator, explicitly reported. Averaging
    # tensor eigenvalues avoids silently calling a varying-scale field Cartesian.
    lateral, vertical = float(np.mean(parallel ** -2)), float(np.mean(perpendicular ** -2))
    positions = graph.positions
    for source in sources:
        reference = graph.shortest_paths([source])[0]
        approximate = np.linalg.norm(embedding.coordinates - embedding.coordinates[source], axis=1)
        same = embedding.component_ids == embedding.component_ids[source]
        approximate[~same] = np.inf
        valid = np.isfinite(reference) & (reference > length_scale * 1e-9)
        local = valid & (reference <= 3 * length_scale)
        distance_errors.extend((np.abs(approximate[local] - reference[local]) / reference[local]).tolist())
        covariance_errors.extend(np.abs(np.exp(-.5 * (approximate[local] / length_scale) ** 2)
                                        - np.exp(-.5 * (reference[local] / length_scale) ** 2)).tolist())
        delta = positions[local] - positions[source]
        cartesian_squared = lateral * np.sum(delta ** 2, axis=1) + (vertical - lateral) * (delta @ normal) ** 2
        cartesian_errors.extend(np.abs(np.exp(-.5 * cartesian_squared / length_scale ** 2)
                                      - np.exp(-.5 * (reference[local] / length_scale) ** 2)).tolist())
        indices = np.flatnonzero(valid)
        k = min(neighbors, len(indices))
        if k:
            threshold = np.partition(reference[indices], k - 1)[k - 1]
            selected = indices[np.argsort(approximate[indices], kind='stable')[:k]]
            overlaps.append(float(np.mean(reference[selected] <= threshold * (1 + 1e-6))))
    report = dict(sources=len(sources), held_out_sources=held_out, local_pairs=len(distance_errors),
                  length_scale=float(length_scale), neighbor_overlap=float(np.mean(overlaps)) if overlaps else None,
                  distance_relative_p50=float(np.percentile(distance_errors, 50)) if distance_errors else None,
                  distance_relative_p90=float(np.percentile(distance_errors, 90)) if distance_errors else None,
                  covariance_absolute_p90=float(np.percentile(covariance_errors, 90)) if covariance_errors else None,
                  cartesian_covariance_absolute_p90=float(np.percentile(cartesian_errors, 90)) if cartesian_errors else None,
                  cartesian_normal=normal.tolist(), bedding_orientation=orientation,
                  cartesian_parallel_scale=lateral ** -.5, cartesian_perpendicular_scale=vertical ** -.5,
                  spectrum=embedding.diagnostics)
    report['passes_provisional_targets'] = bool(
        distance_errors and overlaps and report['neighbor_overlap'] >= .9
        and report['distance_relative_p90'] <= .15 and report['covariance_absolute_p90'] <= .05
    )
    return report


def constant_metric_quality(graph, embedding, *, length_scale, samples=32, seed=1):
    """Separate graph and embedding errors against an analytical constant tensor.

    Both errors use identical held-out pairs within three *analytical* length
    scales. Straight-line distances ignore intra-component barriers; excess graph
    distance includes both stencil bias and any domain-confinement detours.
    Raises for spatially varying metrics rather than mislabelling a fitted tensor
    as geological ground truth.
    """
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    samples = positive_integer(samples, 'samples')
    coordinates = constant_metric_coordinates(graph.metric)
    sources, held_out = _validation_sources(graph, embedding, samples, seed)
    errors = {name: {'distance': [], 'covariance': []} for name in ('graph', 'embedding')}
    for source in sources:
        exact = np.linalg.norm(coordinates - coordinates[source], axis=1)
        local = ((exact > length_scale * 1e-9) & (exact <= 3 * length_scale)
                 & (graph.component_ids == graph.component_ids[source]))
        target = np.exp(-.5 * (exact[local] / length_scale) ** 2)
        distances = dict(graph=graph.shortest_paths([source])[0],
                         embedding=np.linalg.norm(embedding.coordinates - embedding.coordinates[source], axis=1))
        for name, distance in distances.items():
            errors[name]['distance'].extend((np.abs(distance[local] - exact[local]) / exact[local]).tolist())
            errors[name]['covariance'].extend(np.abs(np.exp(-.5 * (distance[local] / length_scale) ** 2) - target).tolist())
    report = dict(reference='unconstrained_constant_tensor', sources=len(sources), held_out_sources=held_out,
                  local_pairs=len(errors['graph']['distance']), length_scale=float(length_scale))
    for name, values in errors.items():
        report[name + '_vs_analytical'] = dict(
            distance_relative_p90=float(np.percentile(values['distance'], 90)) if values['distance'] else None,
            covariance_absolute_p90=float(np.percentile(values['covariance'], 90)) if values['covariance'] else None,
        )
    return report


def graph_refinement_quality(graph, reference, *, length_scale, samples=8, seed=1):
    """Compare two graphs at exactly coincident voxel centers, without snapping.

    The reference lattice must refine each original axis by an odd integer. Domain
    changes and reachability mismatches are reported separately from finite local
    errors. A finer/wider graph is a convergence reference, not geological truth.
    """
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    samples = positive_integer(samples, 'samples')
    shape, fine_shape = np.asarray(graph.metric.grid_shape), np.asarray(reference.metric.grid_shape)
    factors = fine_shape // shape
    if np.any(fine_shape % shape) or np.any(factors < 1) or np.any(factors % 2 != 1):
        raise ValueError("Reference grid must be an odd-integer refinement on every axis")
    ijk = np.column_stack(np.unravel_index(graph.node_indices, tuple(shape)))
    fine_ijk = ijk * factors + (factors - 1) // 2
    flat = np.ravel_multi_index(fine_ijk.T, tuple(fine_shape))
    if not np.allclose(graph.positions, reference.metric.positions[flat], rtol=0., atol=1e-8):
        raise ValueError("Reference voxel centers must coincide in physical coordinates")
    for coarse, fine in zip(graph.metric.local_scales(graph.node_indices), reference.metric.local_scales(flat)):
        if not np.allclose(coarse, fine, rtol=1e-10, atol=0):
            raise ValueError("Graph continuity scales and anisotropy ratios must match at coincident centers")
    mapping = np.full(reference.metric.active_mask.size, -1, dtype=int)
    mapping[reference.node_indices] = np.arange(len(reference.node_indices))
    mapped = mapping[flat]
    common = np.flatnonzero(mapped >= 0)
    if not len(common):
        raise ValueError("No original active nodes remain active on the reference grid")
    sources = np.random.default_rng(seed).choice(common, min(samples, len(common)), replace=False)
    relative, covariance = [], []
    lost, gained = 0, 0
    for source in sources:
        coarse = graph.shortest_paths([source])[0, common]
        fine = reference.shortest_paths([mapped[source]])[0, mapped[common]]
        lost += int(np.sum(np.isfinite(coarse) & ~np.isfinite(fine)))
        gained += int(np.sum(~np.isfinite(coarse) & np.isfinite(fine)))
        local = np.isfinite(coarse) & np.isfinite(fine) & (fine > length_scale * 1e-9) & (fine <= 3 * length_scale)
        relative.extend((np.abs(coarse[local] - fine[local]) / fine[local]).tolist())
        covariance.extend(np.abs(np.exp(-.5 * (coarse[local] / length_scale) ** 2)
                                 - np.exp(-.5 * (fine[local] / length_scale) ** 2)).tolist())
    normals = graph.metric.normals[graph.node_indices[common]]
    fine_normals = reference.metric.normals[flat[common]]
    angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(normals * fine_normals, axis=1)), 0., 1.)))
    return dict(reference='refined_graph_at_coincident_centers', refinement_factors=factors.tolist(),
                reference_grid_shape=fine_shape.tolist(), reference_stencil_radius=reference.stencil_radius,
                sources=len(sources), matched_nodes=len(common), unmatched_active_nodes=int(np.sum(mapped < 0)),
                reachability_lost_pairs=lost, reachability_gained_pairs=gained, local_pairs=len(relative),
                length_scale=float(length_scale), normal_difference_p90_degrees=float(np.percentile(angles, 90)),
                distance_relative_p90=float(np.percentile(relative, 90)) if relative else None,
                covariance_absolute_p90=float(np.percentile(covariance, 90)) if covariance else None)


def directional_variograms(graph, realizations, *, max_separation, samples=4096, seed=0):
    """Empirical short-edge semivariances along bedding and across bedding.

    Input values have shape (realizations, active_nodes). Both sets use the same
    physical maximum separation. Pair counts and mean separations are reported
    because differing sampled lengths can otherwise confound this comparison.
    """
    values = np.asarray(realizations)
    if values.ndim != 2 or values.shape[1] != len(graph.node_indices):
        raise ValueError("Expected (realizations, active_nodes) values")
    if not np.isfinite(max_separation) or max_separation <= 0:
        raise ValueError("max_separation must be finite and positive")
    samples = positive_integer(samples, 'samples')
    start, end = graph.adjacency.nonzero()
    keep = start < end
    start, end = start[keep], end[keep]
    delta = graph.positions[end] - graph.positions[start]
    distance = np.linalg.norm(delta, axis=1)
    normals = graph.metric.normals[graph.node_indices]
    projection = np.sqrt((np.sum(delta * normals[start], axis=1) ** 2
                          + np.sum(delta * normals[end], axis=1) ** 2) / 2) / distance
    rng = np.random.default_rng(seed)
    result = {}
    for name, orientation in [('parallel', projection <= .25), ('perpendicular', projection >= .9)]:
        candidates = np.flatnonzero(orientation & (distance <= max_separation))
        chosen = rng.choice(candidates, min(samples, len(candidates)), replace=False)
        result[name] = dict(pairs=len(chosen),
                            mean_separation=float(distance[chosen].mean()) if len(chosen) else None,
                            semivariance=float(.5 * np.mean((values[:, start[chosen]] - values[:, end[chosen]]) ** 2)) if len(chosen) else None)
    return result
