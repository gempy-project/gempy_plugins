"""Closed unit volumes of a cut tetrahedral mesh: the faces between different (unit, block) cells.

After the field cut every tetrahedron lies on one side of every surface, so its
unit and fault block follow from its values. The faces between different
(unit, block) cells, plus the box faces, form the closed-volume complex:
watertight and conforming by construction, with no welding. Thin units may
touch themselves along an edge or vertex, so shells are closed 2-chains (every
directed edge balanced by its reverse) rather than manifolds.
"""

import numpy as np

from .data import OUTSIDE, VolumeComplex

OUTWARD_FACES = np.array([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)])


def closed_volume_complex(cut, unit_stack, interval, tet_volumes, fault_surfaces, horizon_surfaces):
    """The closed-volume complex of a cut mesh whose tetrahedra are labelled by unit.

    ``unit_stack`` / ``interval`` (T,) are each tetrahedron's unit and
    ``tet_volumes`` its signed volume. ``fault_surfaces[f]`` is the global
    surface index of fault ``f``; ``horizon_surfaces`` lists ``(stack, level,
    surface index)`` of the other surfaces, younger stacks first. An interface
    face lies on the first surface of its two units' stacks whose sign differs
    across it. Units are ``(stack, interval)``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nodes, tets, blocks, block_sides = cut.nodes, cut.tetrahedra, cut.blocks, cut.block_sides
    unit_ids = np.stack((unit_stack, interval), axis=1)
    _, cell_unit = np.unique(unit_ids, axis=0, return_inverse=True)
    cell_unit = cell_unit.reshape(-1)
    units = [(int(g), int(k)) for g, k in np.unique(unit_ids, axis=0).tolist()]
    unit_stacks = np.array([u[0] for u in units])
    faces = tets[:, OUTWARD_FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(len(tets)), 4)
    ordered = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(ordered, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if np.any(counts > 2):
        raise ValueError('nonconforming_cut_mesh: a face is shared by more than two tetrahedra')
    order = np.argsort(inverse, kind='stable')
    first = order[np.r_[0, np.flatnonzero(np.diff(inverse[order]))+1]]
    paired = counts[inverse[first]] == 2
    second = np.full(len(first), -1)
    rest = np.setdiff1d(order, first, assume_unique=True)
    second_of = np.full(len(counts), -1)
    second_of[inverse[rest]] = rest
    second[paired] = second_of[inverse[first[paired]]]

    # Box faces.
    lone = first[~paired]
    p = nodes[faces[lone]]
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    scale = np.linalg.norm(high-low)
    on_box = np.any(np.all(np.abs(p-low) < 1e-9*scale, axis=1) | np.all(np.abs(p-high) < 1e-9*scale, axis=1),
                    axis=1)
    if not on_box.all():
        raise ValueError('open_cut_mesh: a boundary face is not on the box')
    # Faces between cells.
    a_face, b_face = first[paired], second[paired]
    ta, tb = owner[a_face], owner[b_face]
    differ = (cell_unit[ta] != cell_unit[tb]) | (blocks[ta] != blocks[tb])
    a_face, ta, tb = a_face[differ], ta[differ], tb[differ]
    across = blocks[ta] != blocks[tb]
    surface = np.full(len(ta), -1, dtype=np.int64)
    if across.any():
        fault = np.argmax(block_sides[blocks[ta[across]]] != block_sides[blocks[tb[across]]], axis=1)
        surface[across] = np.asarray(fault_surfaces, dtype=np.int64)[fault]
    stacks_a, stacks_b = unit_stacks[cell_unit[ta]], unit_stacks[cell_unit[tb]]
    values = cut.stack_values
    for stack, level, index in horizon_surfaces:
        pending = (surface < 0) & ((stacks_a == stack) | (stacks_b == stack))
        # Snapped tetrahedra can sit exactly on a level; units count it on either side.
        a, b = values[stack, ta], values[stack, tb]
        flips = pending & (((a > level) != (b > level)) | ((a >= level) != (b >= level)))
        surface[flips] = index
    if np.any(surface < 0):
        raise ValueError('unlabelled_cut_interface: no surface separates two adjacent units')
    same_unit = cell_unit[ta] == cell_unit[tb]
    kinds = ['extent']*len(lone) + np.where(across, np.where(same_unit, 'fault_internal', 'fault'),
                                            'interface').tolist()
    sides = [(units[u], OUTSIDE) for u in cell_unit[owner[lone]].tolist()] + \
        [(units[x], units[y]) for x, y in zip(cell_unit[ta].tolist(), cell_unit[tb].tolist())]
    surfaces = [-1]*len(lone) + surface.tolist()
    patches = np.concatenate((faces[lone], faces[a_face]))
    patches = np.asarray(patches, dtype=np.int64).reshape(-1, 3)
    used, compact = np.unique(patches, return_inverse=True)
    triangles = compact.reshape(-1, 3)
    vertices = nodes[used]

    # Region seeds: the centroid of the largest tetrahedron of each connected cell.
    same = paired.copy()
    same[np.flatnonzero(paired)[differ]] = False
    graph = coo_matrix((np.ones(int(same.sum())), (owner[first[same]], owner[second[same]])),
                       shape=(len(tets), len(tets)))
    _, component = connected_components(graph, directed=False)
    volumes = tet_volumes
    best = {}
    for t in np.argsort(-volumes, kind='stable').tolist():
        best.setdefault(int(component[t]), t)
    seeds = [(units[cell_unit[t]], nodes[tets[t]].mean(axis=0)) for t in best.values()]

    shells = {}
    for tri, (negative, positive) in zip(triangles.tolist(), sides):
        if negative == positive:
            continue
        if negative != OUTSIDE:
            shells.setdefault(negative, []).append(tuple(tri))
        if positive != OUTSIDE:
            shells.setdefault(positive, []).append((tri[0], tri[2], tri[1]))
    audit = _audit_shells(shells, dict(enumerate(vertices)))
    balanced = {unit: balanced_chain(rows) for unit, rows in shells.items()}
    p = vertices[triangles]
    areas = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)/2
    degenerate = int(np.sum(areas <= 1e-14*np.sum(np.ptp(vertices, axis=0)**2)))
    tet_volumes = np.bincount(cell_unit, weights=volumes, minlength=len(units))
    box = float(np.prod(high-low))
    total = sum(audit['volumes'].values())
    report = dict(method='field_cut_background_tets', volumes=audit['volumes'],
                  all_closed=all(balanced.values()) and not degenerate, degenerate_triangles=degenerate,
                  nonmanifold_units=sorted(u for u, closed in audit['closed'].items() if not closed),
                  all_positive=audit['all_positive'], open_edges=audit['open_edges'],
                  hanging_nodes=count_hanging_nodes(vertices, triangles), region_volume=box,
                  volume_error=abs(total-box), volumes_sum_to_region=bool(abs(total-box) <= 1e-9*box),
                  tet_volume_error=max(abs(tet_volumes[i]-audit['volumes'].get(u, 0.)) for i, u in enumerate(units)),
                  region_seeds=seeds, blocks=len(block_sides), cut=cut.stats,
                  patch_counts={k: kinds.count(k) for k in sorted(set(kinds))})
    return VolumeComplex(vertices=vertices, triangles=triangles, unit_sides=sides, patch_kinds=kinds,
                         patch_surfaces=surfaces, vertex_ids=used, report=report)


def connected_cells(tets, labels):
    """Connected component per tetrahedron of face-adjacent tetrahedra with equal ``labels``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.sort(tets[:, OUTWARD_FACES].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(tets)), 4)
    _, inverse = np.unique(faces, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind='stable')
    shared = np.flatnonzero(inverse[order][1:] == inverse[order][:-1])
    a, b = owner[order[shared]], owner[order[shared+1]]
    keep = labels[a] == labels[b]
    graph = coo_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(len(tets), len(tets)))
    return connected_components(graph, directed=False)[1]


def _audit_shells(shells, positions):
    """Closure (each edge exactly twice, opposite directions) and signed volume per unit."""
    report = dict(closed={}, volumes={}, degenerate={}, open_edges={})
    for unit, triangles in shells.items():
        keys = sorted({k for tri in triangles for k in tri})
        ids = {k: i for i, k in enumerate(keys)}
        faces = np.array([[ids[k] for k in tri] for tri in triangles], dtype=np.int64).reshape(-1, 3)
        xyz = np.array([positions[k] for k in keys]).reshape(-1, 3)
        directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        codes = directed[:, 0]*len(keys)+directed[:, 1]
        reverse = directed[:, 1]*len(keys)+directed[:, 0]
        unique, counts = np.unique(codes, return_counts=True)
        closed = bool(np.all(counts == 1) and np.isin(reverse, unique).all())
        a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
        area = np.linalg.norm(np.cross(b-a, c-a), axis=1)
        report['closed'][unit] = closed
        report['open_edges'][unit] = int(np.sum(~np.isin(reverse, unique)) + np.sum(counts > 1))
        report['degenerate'][unit] = int(np.sum(area < 1e-14))
        report['volumes'][unit] = float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum()/6)
    report['all_closed'] = all(report['closed'].values()) and not any(report['degenerate'].values())
    report['all_positive'] = all(v > 0 for v in report['volumes'].values())
    return report


def balanced_chain(triangles):
    """Closed 2-chain: every directed edge is used as often as its reverse.

    Weaker than ``audit_shells``' manifold closure: a unit may touch itself
    along an edge or at a vertex (thin units in a labelled tetrahedral grid).
    """
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    directed = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    forward = np.unique(directed, axis=0, return_counts=True)
    backward = np.unique(directed[:, ::-1], axis=0, return_counts=True)
    return len(forward[0]) == len(backward[0]) and np.array_equal(forward[0], backward[0]) and \
        np.array_equal(forward[1], backward[1])


def count_hanging_nodes(vertices, triangles, tolerance=1e-9):
    """Vertices lying strictly inside an edge they are not an endpoint of (geometric T-junctions).

    ``tolerance`` is relative to the bounding-box diagonal.
    """
    from scipy.spatial import cKDTree

    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return 0
    edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                              triangles[:, [2, 0]])), axis=1), axis=0)
    eps = tolerance*max(np.linalg.norm(np.ptp(vertices, axis=0)), 1e-300)
    tree = cKDTree(vertices)
    p, q = vertices[edges[:, 0]], vertices[edges[:, 1]]
    lengths = np.linalg.norm(q-p, axis=1)
    near = tree.query_ball_point((p+q)/2, lengths/2+eps)
    rows = np.repeat(np.arange(len(edges)), [len(n) for n in near])
    candidates = np.fromiter((k for n in near for k in n), dtype=np.int64, count=len(rows))
    keep = (candidates != edges[rows, 0]) & (candidates != edges[rows, 1]) & (lengths[rows] > eps)
    rows, candidates = rows[keep], candidates[keep]
    direction = (q-p)[rows]/lengths[rows, None]
    offset = vertices[candidates]-p[rows]
    along = np.einsum('ij,ij->i', offset, direction)
    distance = np.linalg.norm(np.cross(offset, direction), axis=1)
    return int(np.sum((distance < eps) & (along > eps) & (along < lengths[rows]-eps)))
