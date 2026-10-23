"""Facies proportions, domain-safe face connectivity, and bedding continuity."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _face_edges(graph):
    mapping = np.full(np.prod(graph.metric.grid_shape), -1, dtype=int)
    mapping[graph.node_indices] = np.arange(len(graph.node_indices))
    mapping = mapping.reshape(graph.metric.grid_shape)
    starts, ends = [], []
    excluded = 0
    for axis in range(3):
        left, right = [slice(None)] * 3, [slice(None)] * 3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        a, b = mapping[tuple(left)].ravel(), mapping[tuple(right)].ravel()
        active = (a >= 0) & (b >= 0)
        a, b = a[active], b[active]
        same = graph.component_ids[a] == graph.component_ids[b]
        excluded += int(np.count_nonzero(~same))
        starts.append(a[same])
        ends.append(b[same])
    return np.concatenate(starts), np.concatenate(ends), excluded


def facies_diagnostics(graph, fields):
    """Describe an ensemble using six-connected voxel faces, never wide edges.

    Connectivity and contacts exclude different geological components, even when
    their sampled facies codes happen to agree. Proportion standard errors use
    independent realizations, not a fictitious count of independent voxels.
    Counts are voxel counts (equal-volume regular grids); contact counts are not
    physical interface areas. The upper triangle stores each undirected face once.
    """
    if graph.fingerprint != fields.embedding.graph_fingerprint:
        raise ValueError("Facies and diagnostic graph do not match")
    values = fields.values.detach().cpu().numpy()
    count = len(fields.names)
    if (values.ndim != 2 or not len(values) or values.shape[1] != len(graph.node_indices)
            or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= count)):
        raise ValueError("Expected valid integer facies codes shaped (realizations, active_nodes)")
    proportions = np.stack([(values == code).mean(axis=1) for code in range(count)], axis=1)
    a, b, excluded = _face_edges(graph)
    adjacency = coo_matrix((np.ones(2 * len(a), dtype=bool), (np.r_[a, b], np.r_[b, a])),
                           shape=(values.shape[1], values.shape[1])).tocsr()
    facies = []
    for code, name in enumerate(fields.names):
        numbers, largest, sizes = [], [], []
        for realization in values:
            nodes = np.flatnonzero(realization == code)
            if not len(nodes):
                numbers.append(0)
                continue
            number, labels = connected_components(adjacency[nodes][:, nodes], directed=False)
            component_sizes = np.bincount(labels)
            numbers.append(number)
            largest.append(float(component_sizes.max() / len(nodes)))
            sizes.extend(component_sizes.tolist())
        facies.append(dict(code=code, name=name, target_proportion=fields.proportions[code],
                           proportion_mean=float(proportions[:, code].mean()),
                           proportion_standard_error=float(proportions[:, code].std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
                           components_mean=float(np.mean(numbers)),
                           component_size_p50_voxels=float(np.median(sizes)) if sizes else None,
                           largest_component_fraction_when_present=float(np.mean(largest)) if largest else None))
    contacts = np.zeros((count, count), dtype=np.int64)
    for realization in values:
        np.add.at(contacts, (np.minimum(realization[a], realization[b]), np.maximum(realization[a], realization[b])), 1)
    delta = graph.positions[b] - graph.positions[a]
    distance = np.linalg.norm(delta, axis=1)
    normals = graph.metric.normals[graph.node_indices]
    projection = np.sqrt((np.sum(delta * normals[a], axis=1) ** 2 + np.sum(delta * normals[b], axis=1) ** 2) / 2) / distance
    directional = {}
    for name, selected in [('parallel', projection <= .25), ('perpendicular', projection >= .9)]:
        directional[name] = dict(face_pairs=int(selected.sum()),
                                 mean_separation=float(distance[selected].mean()) if selected.any() else None,
                                 same_facies_probability=float(np.mean(values[:, a[selected]] == values[:, b[selected]])) if selected.any() else None)
    components = []
    for component in np.unique(graph.component_ids):
        selected = graph.component_ids == component
        components.append(dict(component=int(component), voxels=int(selected.sum()),
                               proportions=[float(np.mean(values[:, selected] == code)) for code in range(count)]))
    return dict(realizations=len(values), connectivity=6, facies=facies, domain_components=components,
                face_pairs=len(a), excluded_cross_component_faces=excluded,
                undirected_contact_counts_upper=contacts.tolist(), directional_agreement=directional)
