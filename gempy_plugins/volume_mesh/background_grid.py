"""Background tetrahedral grid of the model box: a Kuhn grid (six tetrahedra per cube)."""

import itertools

import numpy as np


def background_shape(extent, cells):
    """Grid shape with near-cubic cells and about ``cells`` cubes."""
    extent = np.asarray(extent, dtype=float)
    lengths = extent[1::2]-extent[::2]
    size = (np.prod(lengths)/cells)**(1/3)
    return np.maximum(1, np.rint(lengths/size)).astype(np.int64)


def kuhn_grid(extent, shape):
    """Nodes and positively oriented tetrahedra of a box grid; the six per cube share the 000-111 diagonal."""
    extent = np.asarray(extent, dtype=float)
    shape = np.asarray(shape, dtype=np.int64)
    axes = [np.linspace(extent[2*a], extent[2*a+1], shape[a]+1) for a in range(3)]
    nodes = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    stride = np.array([(shape[1]+1)*(shape[2]+1), shape[2]+1, 1])
    cubes = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), axis=-1).reshape(-1, 3)
    base = cubes @ stride
    tets = []
    for order in itertools.permutations(range(3)):
        path = [np.zeros(3, dtype=np.int64)]
        for axis in order:
            step = path[-1].copy()
            step[axis] += 1
            path.append(step)
        tets.append(np.stack([base + p @ stride for p in path], axis=1))
    tets = np.concatenate(tets)
    negative = signed_volumes(nodes, tets) < 0
    tets[negative] = tets[negative][:, [0, 2, 1, 3]]
    return nodes, tets


def signed_volumes(nodes, tets):
    p = nodes[tets]
    return np.einsum('ij,ij->i', p[:, 1]-p[:, 0], np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]))/6
