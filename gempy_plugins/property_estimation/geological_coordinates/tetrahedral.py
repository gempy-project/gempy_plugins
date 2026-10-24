"""Experimental cell-centered geometry on a conforming tetrahedral mesh.

The prepared evaluator must accept the same coordinates as ``mesh.nodes``.
With ``gp.prepare_field_evaluator`` these are physical/world coordinates.
No model solve, grid mutation, voxel interpolation or vertex averaging is used.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ._utils import fingerprint, positive_integer
from .metric import GeologicalMetricField


def evaluate_tetrahedral_fields(mesh, evaluator, *, mask=None, step=1e-3, chunk_size=4096):
    """Return full-cell scalar values and finite-difference gradients at centroids.

    Each cell uses its unit's stack and its explicit fault-block sides. Central
    differences hold those sides fixed at both offsets, differentiating the
    smooth one-sided field even near a fault. ``step`` is in mesh-coordinate units
    (a positive scalar or XYZ vector). This is a numerical gradient estimate,
    since GemPy's public prepared evaluator currently exposes scalars only.
    Unselected cells are NaN. Query memory is bounded by ``chunk_size``.
    """
    chunk_size = positive_integer(chunk_size, 'chunk_size')
    step = np.asarray(step, dtype=float)
    if step.shape not in ((), (3,)) or not np.isfinite(step).all() or np.any(step <= 0):
        raise ValueError("step must be a finite positive scalar or XYZ vector")
    steps = np.broadcast_to(step, (3,))
    positions = np.asarray(mesh.nodes)[mesh.tetrahedra].mean(axis=1)
    count = len(positions)
    mask = np.ones(count, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != (count,):
        raise ValueError("mask must contain one entry per tetrahedron")
    stacks = np.asarray(mesh.units, dtype=int)[mesh.tet_units, 0]
    values, gradients = np.full(count, np.nan), np.full((count, 3), np.nan)
    for stack in np.unique(stacks[mask]):
        indices = np.flatnonzero(mask & (stacks == stack))
        for start in range(0, len(indices), chunk_size):
            cells = indices[start:start + chunk_size]
            points = positions[cells]
            sides = mesh.block_sides[mesh.tet_blocks[cells]].T
            kwargs = dict(stacks=[int(stack)], fault_sides=sides)
            values[cells] = evaluator(points, **kwargs)[0]
            for axis, h in enumerate(steps):
                offset = np.eye(3)[axis] * h
                gradients[cells, axis] = (evaluator(points + offset, **kwargs)[0]
                                          - evaluator(points - offset, **kwargs)[0]) / (2 * h)
    if not np.isfinite(values[mask]).all() or not np.isfinite(gradients[mask]).all():
        raise ValueError("The prepared evaluator returned nonfinite centroid fields")
    return values, gradients


def graph_from_tetrahedral_mesh(mesh, gradients, *, mask=None, anisotropy_ratio=5., continuity_scale=1.,
                                gradient_tolerance=1e-12, dtype=np.float64):
    """Build a sparse dual graph using only same-unit, same-block shared faces.

    Each edge follows centroid A -> shared-face centroid -> centroid B, with
    each half evaluated in its cell's constant tensor. This path stays inside
    the two convex tetrahedra; a straight centroid-to-centroid segment need not.
    Face-only connectivity has directional bias and is an experimental distance
    approximation, not a continuum geodesic solver. Isomap adds a separate error.
    """
    from .graph import AnisotropicGraph

    xyz, tets = np.asarray(mesh.nodes, dtype=float), np.asarray(mesh.tetrahedra)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Mesh nodes must be finite (nodes, 3) coordinates")
    if (tets.ndim != 2 or tets.shape[1] != 4 or not len(tets)
            or not np.issubdtype(tets.dtype, np.integer) or np.any(tets < 0) or np.any(tets >= len(xyz))):
        raise ValueError("Expected nonempty valid integer tetrahedra")
    vertices = xyz[tets]
    volume6 = np.einsum('ij,ij->i', vertices[:, 1] - vertices[:, 0],
                       np.cross(vertices[:, 2] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]))
    if not np.isfinite(volume6).all() or np.any(volume6 == 0):
        raise ValueError("Degenerate tetrahedra cannot define cell-centered paths")
    if len(np.unique(np.sort(tets, axis=1), axis=0)) != len(tets):
        raise ValueError("Duplicate tetrahedra are not supported")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError("Graph weights must use float32 or float64")
    units, blocks = np.asarray(mesh.tet_units), np.asarray(mesh.tet_blocks)
    if any(a.shape != (len(tets),) or not np.issubdtype(a.dtype, np.integer) for a in (units, blocks)):
        raise ValueError("Expected one integer unit and fault-block label per tetrahedron")
    domains, labels = np.unique(np.column_stack((units, blocks)), axis=0, return_inverse=True)
    metadata = dict(sampling='tetrahedron_centroids', edge_path='centroid_face_centroid',
                    mesh_fingerprint=fingerprint({}, xyz, tets, units, blocks),
                    mesh_domains=domains.tolist())
    metric = GeologicalMetricField.from_points(
        vertices.mean(axis=1), gradients, mask=mask, domain_labels=labels,
        anisotropy_ratio=anisotropy_ratio, continuity_scale=continuity_scale,
        gradient_tolerance=gradient_tolerance, metadata=metadata,
    )
    faces = np.sort(tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3), axis=1)
    unique_faces, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Nonmanifold mesh: more than two tetrahedra share a face")
    order = np.argsort(inverse, kind='stable')
    offsets = np.concatenate(([0], np.cumsum(counts)))
    shared = np.flatnonzero(counts == 2)
    a, b = order[offsets[shared]] // 4, order[offsets[shared] + 1] // 4
    keep = metric.active_mask[a] & metric.active_mask[b] & (labels[a] == labels[b])
    a, b, shared = a[keep], b[keep], shared[keep]
    face_centers = xyz[unique_faces[shared]].mean(axis=1)
    costs = np.zeros(len(a))
    for cells in (a, b):
        delta = face_centers - metric.positions[cells]
        costs += np.sqrt(np.einsum('ni,nij,nj->n', delta, metric.tensors(cells), delta))
    costs = costs.astype(dtype)
    if not np.all(np.isfinite(costs) & (costs > 0)):
        raise ValueError("Mesh graph edges must have finite, strictly positive costs")
    nodes = np.flatnonzero(metric.active_mask)
    mapping = np.full(len(tets), -1, dtype=int)
    mapping[nodes] = np.arange(len(nodes))
    rows, columns = np.concatenate((mapping[a], mapping[b])), np.concatenate((mapping[b], mapping[a]))
    adjacency = coo_matrix((np.tile(costs, 2), (rows, columns)), shape=(len(nodes), len(nodes))).tocsr()
    _, components = connected_components(adjacency, directed=False)
    return AnisotropicGraph(metric, adjacency, nodes, components, connectivity='tetrahedral_faces')
