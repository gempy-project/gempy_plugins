"""Cut a background tetrahedral grid by the model's own fields (volume-first meshing).

Every fault level set is cut into the grid, then, inside each fault block,
every horizon level set. Fields are piecewise linear on the tetrahedra: actual
values at the grid nodes, linear interpolation at every inserted vertex, so one
pass per surface leaves no tetrahedron edge strictly crossing it. Horizons are
sampled one-sided per fault block (fault drift fixed to the block's side), so a
throw is never smeared across the fault step: a vertex on a fault carries one
value set per block it touches.

A level set is cut in by inserting its root on every crossed edge. A
tetrahedron with points on its edges is bisected at the point with the smallest
vertex id, repeatedly; a face is then split only by points on its own edges, in
increasing id order, so neighbours always agree and the mesh stays conforming.
Roots within ``snap`` (a fraction of their edge) of a vertex move the surface
onto that vertex (its value becomes the level): surfaces shift by at most that
much, and no needle or sliver is cut next to a vertex.
"""

import numpy as np

from .data import FieldCut

LOCAL_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
_NONE = np.iinfo(np.int64).max


class _Mesh:
    """Growing conforming tetrahedral mesh; ``attributes`` follow each tetrahedron through splits."""

    def __init__(self, nodes, tets, attributes):
        self.xyz = np.asarray(nodes, dtype=float)
        self.tets = np.asarray(tets, dtype=np.int64)
        self.attributes = np.asarray(attributes, dtype=np.int64)
        self.original = len(self.xyz)

    @property
    def big(self):
        return np.int64(len(self.xyz)+1)

    def edge_codes(self, tets=None):
        tets = self.tets if tets is None else tets
        ends = tets[:, LOCAL_EDGES]
        return np.minimum(ends[..., 0], ends[..., 1])*self.big + np.maximum(ends[..., 0], ends[..., 1])

    def insert(self, u, v, t):
        """Insert points ``(1-t) u + t v`` on edges (u < v); return their vertex ids and (u, v, t) per new vertex.

        Points on one edge closer than 1e-12 in ``t`` are one vertex.
        """
        u, v, t = np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64), np.asarray(t, dtype=float)
        if not len(u):
            return np.empty(0, dtype=np.int64), (u, v, t)
        order = np.lexsort((t, v, u))
        su, sv, st = u[order], v[order], t[order]
        new_point = np.ones(len(su), dtype=bool)
        new_point[1:] = (su[1:] != su[:-1]) | (sv[1:] != sv[:-1]) | (st[1:]-st[:-1] > 1e-12)
        point = np.cumsum(new_point)-1
        pu, pv, pt = su[new_point], sv[new_point], st[new_point]
        ids = len(self.xyz)+np.arange(len(pu))
        self.xyz = np.concatenate((self.xyz, (1-pt)[:, None]*self.xyz[pu] + pt[:, None]*self.xyz[pv]))
        # Rank of each point along its edge; round k splits (point k-1, v) at point k.
        first = np.ones(len(pu), dtype=bool)
        first[1:] = (pu[1:] != pu[:-1]) | (pv[1:] != pv[:-1])
        start = np.maximum.accumulate(np.where(first, np.arange(len(pu)), 0))
        rank = np.arange(len(pu))-start
        for k in range(int(rank.max())+1):
            rows = np.flatnonzero(rank == k)
            previous = ids[rows-1] if k else pu[rows]
            lo, hi = np.minimum(previous, pv[rows]), np.maximum(previous, pv[rows])
            self._bisect(lo*self.big + hi, ids[rows])
        result = np.empty(len(u), dtype=np.int64)
        result[order] = ids[point]
        return result, (pu, pv, pt)

    def _bisect(self, codes, vertices):
        """Split every tetrahedron at the points on its edges (one point per edge)."""
        order = np.argsort(codes)
        keys, values = codes[order], vertices[order]
        # Only tetrahedra with both ends of a split edge can be split.
        big = self.big
        ends = np.zeros(len(self.xyz), dtype=bool)
        ends[keys//big] = ends[keys % big] = True
        involved = ends[self.tets].sum(axis=1) >= 2
        tets, attributes = self.tets[involved], self.attributes[involved]
        done_tets, done_attributes = [self.tets[~involved]], [self.attributes[~involved]]
        while len(tets):
            edge_codes = self.edge_codes(tets)
            found = np.minimum(np.searchsorted(keys, edge_codes), len(keys)-1)
            hit = keys[found] == edge_codes
            split = hit.any(axis=1)
            done_tets.append(tets[~split])
            done_attributes.append(attributes[~split])
            rows = np.flatnonzero(split)
            candidates = np.where(hit[rows], values[found[rows]], _NONE)
            local = np.argmin(candidates, axis=1)
            point = candidates[np.arange(len(rows)), local]
            a, b = LOCAL_EDGES[local, 0], LOCAL_EDGES[local, 1]
            first, second = tets[rows].copy(), tets[rows].copy()
            first[np.arange(len(rows)), b] = point
            second[np.arange(len(rows)), a] = point
            tets = np.concatenate((first, second))
            attributes = np.concatenate((attributes[rows], attributes[rows]))
        self.tets = np.concatenate(done_tets)
        self.attributes = np.concatenate(done_attributes)


def _crossings(mesh, corner_values, snap, snapped):
    """Strictly crossed edges (lo, hi, t from lo) of tetrahedra with corner values (T, 4).

    Roots within ``snap`` (fraction of the edge) of a vertex first move onto
    that vertex: ``snapped(tet_rows, local_corners)`` sets those corner values
    to the level.
    Returns per crossing the edge code, its tetrahedron and t.
    """
    for attempt in range(2):
        spans = (corner_values.min(axis=1) < 0) & (corner_values.max(axis=1) > 0)
        rows = np.flatnonzero(spans)
        f = corner_values[rows][:, LOCAL_EDGES]  # (R, 6, 2)
        ends = mesh.tets[rows][:, LOCAL_EDGES]
        crossed = f[..., 0]*f[..., 1] < 0
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(crossed, f[..., 0]/(f[..., 0]-f[..., 1]), np.nan)
        near_a = crossed & (t < snap)
        near_b = crossed & (t > 1-snap) & ~near_a
        if attempt or not (near_a.any() or near_b.any()):
            break  # snapped nodes are on the level: no new near roots
        r, e = np.nonzero(near_a)
        snapped(rows[r], LOCAL_EDGES[e, 0])
        r, e = np.nonzero(near_b)
        snapped(rows[r], LOCAL_EDGES[e, 1])
        corner_values = snapped(None, None)
    r, e = np.nonzero(crossed)
    a, b = ends[r, e, 0], ends[r, e, 1]
    tt = t[r, e]
    swap = a > b
    lo, hi = np.where(swap, b, a), np.where(swap, a, b)
    return lo, hi, np.where(swap, 1-tt, tt), rows[r]


def _crossings(mesh, corner_values, snap, snapped):
    """Strictly crossed edges (lo, hi, t from lo) of tetrahedra with corner values (T, 4).

    Roots within ``snap`` (fraction of the edge) of a vertex first move onto
    that vertex: ``snapped(tet_rows, local_corners)`` sets those corner values
    to the level.
    Returns per crossing the edge code, its tetrahedron and t.
    """
    for attempt in range(2):
        spans = (corner_values.min(axis=1) < 0) & (corner_values.max(axis=1) > 0)
        rows = np.flatnonzero(spans)
        f = corner_values[rows][:, LOCAL_EDGES]  # (R, 6, 2)
        ends = mesh.tets[rows][:, LOCAL_EDGES]
        crossed = f[..., 0]*f[..., 1] < 0
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(crossed, f[..., 0]/(f[..., 0]-f[..., 1]), np.nan)
        near_a = crossed & (t < snap)
        near_b = crossed & (t > 1-snap) & ~near_a
        if attempt or not (near_a.any() or near_b.any()):
            break  # snapped nodes are on the level: no new near roots
        r, e = np.nonzero(near_a)
        snapped(rows[r], LOCAL_EDGES[e, 0])
        r, e = np.nonzero(near_b)
        snapped(rows[r], LOCAL_EDGES[e, 1])
        corner_values = snapped(None, None)
    r, e = np.nonzero(crossed)
    a, b = ends[r, e, 0], ends[r, e, 1]
    tt = t[r, e]
    swap = a > b
    lo, hi = np.where(swap, b, a), np.where(swap, a, b)
    return lo, hi, np.where(swap, 1-tt, tt), rows[r]


def cut_field_volumes(nodes, tets, *, fault_values, block_fields, stack_isovalues, fault_stacks, snap=.15):
    """Cut a conforming background tet mesh by every fault, then every horizon per fault block.

    ``fault_values(points) -> (F, M)`` are the fault stacks' raw scalars minus
    their surface level (independent of the drift). ``block_fields(points,
    stacks, sides) -> (S, M)`` are raw scalars of non-fault ``stacks`` with each
    fault's drift fixed to ``sides`` (F, M) of +1/-1. ``stack_isovalues`` holds
    every stack's levels. Roots within ``snap`` (fraction of their edge) of a
    vertex move onto that vertex.
    """
    snap = max(float(snap), 1e-9)  # never insert a point on top of a vertex
    mesh = _Mesh(nodes, tets, np.zeros(len(tets)))
    n_stacks = len(stack_isovalues)
    fault_stacks = list(fault_stacks)
    other_stacks = [g for g in range(n_stacks) if g not in fault_stacks]
    stats = dict(background_nodes=len(nodes), background_tets=len(tets), passes=[])

    # region Faults: node-valued, drift independent
    values = np.asarray(fault_values(nodes), dtype=float).T.copy()  # (V, F)
    for f in range(len(fault_stacks)):
        def snapped(rows, local, f=f):
            if rows is not None:
                values[mesh.tets[rows, local], f] = 0.
            return values[mesh.tets, f]
        lo, hi, t, _ = _crossings(mesh, values[mesh.tets, f], snap, snapped)
        key = lo*mesh.big + hi
        _, unique = np.unique(key, return_index=True)
        ids, (pu, pv, pt) = mesh.insert(lo[unique], hi[unique], t[unique])
        new = (1-pt)[:, None]*values[pu] + pt[:, None]*values[pv]
        new[:, f] = 0.
        values = np.concatenate((values, new))
        stats['passes'].append(dict(surface=('fault', fault_stacks[f]), points=len(pu), tets=len(mesh.tets)))
    corner = values[mesh.tets]  # (T, 4, F)
    totals = corner.sum(axis=1)
    # A tetrahedron snapped flat onto a fault may take either side: any
    # labelling of a conforming mesh is a valid partition.
    sides = np.where(totals >= 0, 1, -1)
    stats['flat_fault_tets'] = int(np.sum(np.any(totals == 0, axis=1)))
    block_sides, blocks = np.unique(sides, axis=0, return_inverse=True)
    mesh.attributes = blocks.reshape(-1).astype(np.int64)
    n_blocks = np.int64(len(block_sides))
    # endregion

    # region Horizons: one-sided values per (vertex, block)
    def pair_codes(tets, blocks):
        return tets*n_blocks + blocks[:, None]

    codes = np.unique(pair_codes(mesh.tets, mesh.attributes))
    pair_vertex, pair_block = codes//n_blocks, codes % n_blocks
    pair_values = np.asarray(block_fields(mesh.xyz[pair_vertex], other_stacks,
                                          block_sides[pair_block].T), dtype=float).T.copy()  # (P, G)
    stats['block_samples'] = len(codes)
    for column, g in enumerate(other_stacks):
        for level in np.asarray(stack_isovalues[g], dtype=float):
            points, rounds = 0, 0
            while True:
                # A root of one block lands on a shared fault-face edge of the
                # neighbouring block too, where it is not on that block's level;
                # splitting there can leave a crossed edge, so repeat until none.
                rows = np.searchsorted(codes, pair_codes(mesh.tets, mesh.attributes))

                def snapped(tet_rows, local, rows=rows, column=column, level=level):
                    if tet_rows is not None:
                        pair_values[rows[tet_rows, local], column] = level
                    return pair_values[rows, column]-level
                lo, hi, t, owner = _crossings(mesh, pair_values[rows, column]-level, snap, snapped)
                if not len(lo):
                    break
                rounds += 1
                if rounds > 20:
                    raise ValueError('unconverged_block_cut: horizon roots keep crossing fault faces')
                codes, pair_values, inserted = _insert_level_points(
                    mesh, lo, hi, t, owner, codes, pair_values, n_blocks, column, level)
                points += inserted
            stats['passes'].append(dict(surface=(int(g), float(level)), points=points, rounds=rounds,
                                        tets=len(mesh.tets)))
    # endregion

    rows = np.searchsorted(codes, pair_codes(mesh.tets, mesh.attributes))
    stack_values = np.zeros((n_stacks, len(mesh.tets)))
    # Tetrahedron means of one-sided values; fault stacks never decide a unit.
    stack_values[other_stacks] = pair_values[rows].mean(axis=1).T
    return FieldCut(nodes=mesh.xyz, tetrahedra=mesh.tets, blocks=mesh.attributes, block_sides=block_sides,
                    stack_values=stack_values, stats=stats)


def _insert_level_points(mesh, lo, hi, t, owner, codes, pair_values, n_blocks, column, level):
    """Insert one block-level pass's roots; new (vertex, block) values by linear interpolation."""
    block = mesh.attributes[owner]
    # Every block whose tetrahedra hold a split edge gets a value set at the new vertex.
    key = lo*mesh.big + hi
    split_keys = np.unique(key)
    edge_codes = mesh.edge_codes()
    found = np.minimum(np.searchsorted(split_keys, edge_codes), max(len(split_keys)-1, 0))
    hit = (split_keys[found] == edge_codes) if len(split_keys) else np.zeros_like(edge_codes, bool)
    r, e = np.nonzero(hit)
    touching = np.unique(np.stack((edge_codes[r, e], mesh.attributes[r]), axis=1), axis=0)
    ids, _ = mesh.insert(lo, hi, t)
    # New (vertex, block) pairs: one per inserted vertex and touching block.
    point_rows = np.unique(np.stack((ids, key), axis=1), axis=0)
    edge_of = dict(zip(key.tolist(), zip(lo.tolist(), hi.tolist())))
    new_pairs = []
    starts = np.searchsorted(touching[:, 0], point_rows[:, 1], side='left')
    stops = np.searchsorted(touching[:, 0], point_rows[:, 1], side='right')
    for (vertex, code), start, stop in zip(point_rows.tolist(), starts, stops):
        for b in touching[start:stop, 1].tolist():
            new_pairs.append((vertex, code, b))
    new_pairs = np.asarray(new_pairs, dtype=np.int64).reshape(-1, 3)
    ends = np.array([edge_of[c] for c in new_pairs[:, 1].tolist()], dtype=np.int64).reshape(-1, 2)
    position = mesh.xyz[new_pairs[:, 0]]
    a, b = mesh.xyz[ends[:, 0]], mesh.xyz[ends[:, 1]]
    fraction = np.einsum('ij,ij->i', position-a, b-a)/np.einsum('ij,ij->i', b-a, b-a)
    rows_a = np.searchsorted(codes, ends[:, 0]*n_blocks + new_pairs[:, 2])
    rows_b = np.searchsorted(codes, ends[:, 1]*n_blocks + new_pairs[:, 2])
    new_values = (1-fraction)[:, None]*pair_values[rows_a] + fraction[:, None]*pair_values[rows_b]
    # The cut surface itself is exactly at its level in the blocks that cut it.
    cut = set(zip(ids.tolist(), block.tolist()))
    on_level = np.array([(v, bl) in cut for v, bl in new_pairs[:, [0, 2]].tolist()], dtype=bool)
    new_values[on_level, column] = level
    new_codes = new_pairs[:, 0]*n_blocks + new_pairs[:, 2]
    codes = np.concatenate((codes, new_codes))
    pair_values = np.concatenate((pair_values, new_values))
    order = np.argsort(codes, kind='stable')
    codes, pair_values = codes[order], pair_values[order]
    if len(codes) > 1 and np.any(codes[1:] == codes[:-1]):
        raise ValueError('duplicate_block_vertex: a vertex received two value sets in one block')
    return codes, pair_values, len(np.unique(ids))


def _insert_level_points(mesh, lo, hi, t, owner, codes, pair_values, n_blocks, column, level):
    """Insert one block-level pass's roots; new (vertex, block) values by linear interpolation."""
    block = mesh.attributes[owner]
    # Every block whose tetrahedra hold a split edge gets a value set at the new vertex.
    key = lo*mesh.big + hi
    split_keys = np.unique(key)
    edge_codes = mesh.edge_codes()
    found = np.minimum(np.searchsorted(split_keys, edge_codes), max(len(split_keys)-1, 0))
    hit = (split_keys[found] == edge_codes) if len(split_keys) else np.zeros_like(edge_codes, bool)
    r, e = np.nonzero(hit)
    touching = np.unique(np.stack((edge_codes[r, e], mesh.attributes[r]), axis=1), axis=0)
    ids, _ = mesh.insert(lo, hi, t)
    # New (vertex, block) pairs: one per inserted vertex and touching block.
    point_rows = np.unique(np.stack((ids, key), axis=1), axis=0)
    edge_of = dict(zip(key.tolist(), zip(lo.tolist(), hi.tolist())))
    new_pairs = []
    starts = np.searchsorted(touching[:, 0], point_rows[:, 1], side='left')
    stops = np.searchsorted(touching[:, 0], point_rows[:, 1], side='right')
    for (vertex, code), start, stop in zip(point_rows.tolist(), starts, stops):
        for b in touching[start:stop, 1].tolist():
            new_pairs.append((vertex, code, b))
    new_pairs = np.asarray(new_pairs, dtype=np.int64).reshape(-1, 3)
    ends = np.array([edge_of[c] for c in new_pairs[:, 1].tolist()], dtype=np.int64).reshape(-1, 2)
    position = mesh.xyz[new_pairs[:, 0]]
    a, b = mesh.xyz[ends[:, 0]], mesh.xyz[ends[:, 1]]
    fraction = np.einsum('ij,ij->i', position-a, b-a)/np.einsum('ij,ij->i', b-a, b-a)
    rows_a = np.searchsorted(codes, ends[:, 0]*n_blocks + new_pairs[:, 2])
    rows_b = np.searchsorted(codes, ends[:, 1]*n_blocks + new_pairs[:, 2])
    new_values = (1-fraction)[:, None]*pair_values[rows_a] + fraction[:, None]*pair_values[rows_b]
    # The cut surface itself is exactly at its level in the blocks that cut it.
    cut = set(zip(ids.tolist(), block.tolist()))
    on_level = np.array([(v, bl) in cut for v, bl in new_pairs[:, [0, 2]].tolist()], dtype=bool)
    new_values[on_level, column] = level
    new_codes = new_pairs[:, 0]*n_blocks + new_pairs[:, 2]
    codes = np.concatenate((codes, new_codes))
    pair_values = np.concatenate((pair_values, new_values))
    order = np.argsort(codes, kind='stable')
    codes, pair_values = codes[order], pair_values[order]
    if len(codes) > 1 and np.any(codes[1:] == codes[:-1]):
        raise ValueError('duplicate_block_vertex: a vertex received two value sets in one block')
    return codes, pair_values, len(np.unique(ids))
