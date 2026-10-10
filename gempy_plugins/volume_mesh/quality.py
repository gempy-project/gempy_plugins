"""Tetrahedron quality measures."""

import numpy as np

_PAIRS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1))


def dihedral_angles(nodes, tets):
    """(T, 6) interior dihedral angles in degrees, one per edge."""
    p = np.asarray(nodes, dtype=float)[np.asarray(tets)]
    angles = []
    for a, b, c, d in _PAIRS:
        axis = p[:, b]-p[:, a]
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        u = p[:, c]-p[:, a]
        v = p[:, d]-p[:, a]
        u -= np.einsum('ij,ij->i', u, axis)[:, None]*axis
        v -= np.einsum('ij,ij->i', v, axis)[:, None]*axis
        cosine = np.einsum('ij,ij->i', u, v)/(np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1))
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return np.stack(angles, axis=1)


def quality_report(nodes, tets):
    """Summary of dihedral angles and the normalised volume-to-edge shape quality (1 is regular)."""
    angles = dihedral_angles(nodes, tets)
    p = np.asarray(nodes, dtype=float)[np.asarray(tets)]
    volumes = np.einsum('ij,ij->i', p[:, 1]-p[:, 0], np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]))/6
    edges = np.stack([np.linalg.norm(p[:, j]-p[:, i], axis=1) for i, j, _, _ in _PAIRS], axis=1)
    shape = 6*np.sqrt(2)*volumes/np.sqrt(np.mean(edges**2, axis=1))**3
    low, high = angles.min(axis=1), angles.max(axis=1)
    return dict(tetrahedra=len(tets), min_dihedral=float(low.min()), max_dihedral=float(high.max()),
                below_5=float(np.mean(low < 5)), below_10=float(np.mean(low < 10)),
                above_165=float(np.mean(high > 165)), shape_median=float(np.median(shape)),
                shape_min=float(shape.min()), inverted=int(np.sum(volumes <= 0)))
