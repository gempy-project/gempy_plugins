"""Sampled distance, neighborhood and covariance validation without all-pairs storage."""
import numpy as np

from ._utils import positive_integer


def embedding_quality(graph, embedding, *, length_scale, samples=32, neighbors=16, seed=1,
                      cartesian_normal=(0., 0., 1.)):
    """Compare held-out-source graph distances against the embedding.

    Local errors cover nonzero distances up to three Gaussian length scales.
    Neighbor overlap accepts ties at the reference kth distance, avoiding an
    arbitrary penalty for equivalent neighbors on a regular lattice. The
    covariance target here is a pairwise diagnostic, not an assertion that a
    Gaussian kernel of arbitrary graph distances forms a PSD covariance matrix.
    """
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    samples = positive_integer(samples, 'samples')
    neighbors = positive_integer(neighbors, 'neighbors')
    if graph.fingerprint != embedding.graph_fingerprint:
        raise ValueError("Embedding and reference graph do not match")
    landmarks = np.concatenate([p.landmark_indices for p in embedding.projections])
    candidates = np.setdiff1d(np.arange(len(graph.node_indices)), landmarks)
    held_out = bool(len(candidates))
    if not held_out:
        candidates = np.arange(len(graph.node_indices))
    sources = np.random.default_rng(seed).choice(candidates, min(samples, len(candidates)), replace=False)
    distance_errors, covariance_errors, cartesian_errors, overlaps = [], [], [], []
    normal = np.asarray(cartesian_normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        raise ValueError("cartesian_normal must be a finite, nonzero XYZ vector")
    normal = normal / np.linalg.norm(normal)
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
        lateral = graph.metric.anisotropy_ratio ** -2
        cartesian_squared = lateral * np.sum(delta ** 2, axis=1) + (1 - lateral) * (delta @ normal) ** 2
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
                  spectrum=embedding.diagnostics)
    report['passes_provisional_targets'] = bool(
        distance_errors and overlaps and report['neighbor_overlap'] >= .9
        and report['distance_relative_p90'] <= .15 and report['covariance_absolute_p90'] <= .05
    )
    return report


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
