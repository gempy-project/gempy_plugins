"""Moving-neighborhood strategies for restricting which conditioning points feed into a
kriging/simulation query.

GSTools' Krige/SRF classes always solve one global system over whatever `cond_pos`/
`cond_val` they're given -- there's no built-in local/windowed kriging. This module adds
that on top, mirroring the old kriging plugin's 'all'/'n_closest'/'range' neighborhood
modes.
"""
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np


@dataclass
class Neighborhood:
    """A moving-neighborhood strategy.

    Args:
        mode: "all" uses every conditioning point in the domain for every query (global
            kriging/simulation -- a single solve per domain, the default and by far the
            fastest option). "n_closest" uses only the `n` closest conditioning points
            to each query point. "range" uses only conditioning points within `radius`
            of each query point. Both "n_closest" and "range" solve kriging separately
            per query point, so they're much slower than "all" -- that's inherent to
            true moving-neighborhood kriging, not a shortcut worth optimizing away.
    """
    mode: Literal["all", "n_closest", "range"] = "all"
    n: Optional[int] = None
    radius: Optional[float] = None

    def __post_init__(self):
        if self.mode == "n_closest" and self.n is None:
            raise ValueError("Neighborhood(mode='n_closest') requires n.")
        if self.mode == "range" and self.radius is None:
            raise ValueError("Neighborhood(mode='range') requires radius.")


def local_neighbor_indices(query_xyz: np.ndarray, cond_xyz: np.ndarray, neighborhood: Neighborhood, model) -> List[np.ndarray]:
    """One conditioning-point index array per query point, for 'n_closest'/'range' modes.

    `model` (the same GSTools `CovModel` used for kriging in this domain) is required
    so the search can be anisotropy-aware: raw Euclidean nearest-neighbor search would
    ignore anisotropy/rotation entirely, and for a strongly anisotropic model, "closest"
    in plain distance is not the same as "closest" in the model's actual correlation
    structure (a point far away along the long axis can be more correlated than one
    nearby along the short axis). `model.isometrize(...)` transforms coordinates into a
    space where plain Euclidean distance directly corresponds to the model's own
    covariance, so a KDTree search there is correct regardless of anisotropy/rotation --
    and for an isotropic model, it reduces to a harmless uniform rescaling that leaves
    the neighbor search unchanged.

    Note this also changes what `radius` means in "range" mode: it's a distance in this
    isometrized space, not raw physical distance -- effectively "how many of the
    model's own (main-axis) correlation lengths away", which is consistent regardless
    of direction.
    """
    from scipy.spatial import cKDTree

    if neighborhood.mode == "all":
        raise ValueError("local_neighbor_indices is only for 'n_closest'/'range' modes.")

    query_xyz = np.array(model.isometrize((query_xyz[:, 0], query_xyz[:, 1], query_xyz[:, 2]))).T
    cond_xyz = np.array(model.isometrize((cond_xyz[:, 0], cond_xyz[:, 1], cond_xyz[:, 2]))).T

    tree = cKDTree(cond_xyz)

    if neighborhood.mode == "n_closest":
        n = min(neighborhood.n, len(cond_xyz))
        _, idx = tree.query(query_xyz, k=n)
        idx = np.atleast_2d(idx)
        if n == 1:
            idx = idx.reshape(-1, 1)
        return [row for row in idx]

    if neighborhood.mode == "range":
        idx_lists = tree.query_ball_point(query_xyz, r=neighborhood.radius)
        return [np.array(idx, dtype=int) for idx in idx_lists]

    raise ValueError(f"Unknown neighborhood mode: {neighborhood.mode}")
