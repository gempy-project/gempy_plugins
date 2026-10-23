"""Unconstrained Cartesian references for constant geological metrics."""
import numpy as np


def bedding_orientation(metric):
    """Dominant unoriented normal and angular spread, invariant to normal signs."""
    normals = metric.normals[metric.active_mask]
    _, vectors = np.linalg.eigh(normals.T @ normals / len(normals))
    normal = vectors[:, -1]
    normal *= np.sign(normal[np.argmax(np.abs(normal))])
    angles = np.degrees(np.arccos(np.clip(np.abs(normals @ normal), 0., 1.)))
    return dict(normal=normal.tolist(), angular_deviation_p90_degrees=float(np.percentile(angles, 90)),
                angular_deviation_max_degrees=float(angles.max()))


def constant_metric_coordinates(metric, *, normal_tolerance=1e-6):
    """Exact Cartesian coordinates for active nodes of a constant tensor field.

    Distances are unconstrained straight-line distances, hence lower bounds for
    paths restricted by masks/barriers. This is a reference, not an automatic
    replacement for domain-constrained Isomap. ``normal_tolerance`` is the maximum
    Euclidean difference between sign-aligned unit normals and the dominant normal.
    """
    if not np.isfinite(normal_tolerance) or normal_tolerance < 0:
        raise ValueError("normal_tolerance must be finite and nonnegative")
    normal = np.asarray(bedding_orientation(metric)['normal'])
    normals = metric.normals[metric.active_mask]
    signs = np.where(normals @ normal < 0, -1., 1.)
    deviation = np.linalg.norm(signs[:, None] * normals - normal, axis=1)
    if metric.anisotropy_ratio != 1 and np.max(deviation) > normal_tolerance:
        raise ValueError("Analytical coordinates require a constant metric")
    # M**(1/2) = I/ratio + (1 - 1/ratio) nn.T. Centering avoids large offsets.
    positions = metric.positions[metric.active_mask]
    delta = positions - positions.mean(axis=0)
    lateral = 1. / metric.anisotropy_ratio
    return lateral * delta + (1 - lateral) * (delta @ normal)[:, None] * normal
