"""Conditioning data: real property samples (XYZ + measured value) used to drive
kriging/simulation, with per-sample domain assignment."""
from typing import Optional, Sequence, Union

import numpy as np

import gempy as gp
from gempy_plugins.property_estimation.domains import DomainGroup, DomainKey, as_group


class ConditioningData:
    """XYZ + measured property value samples, with per-sample domain assignment.

    Domain assignment snaps each sample to its nearest regular-grid cell (rather than
    gempy's exact per-point interpolation) so a sample's assigned domain is always one
    of `domains.compute_domains`'s own `domain_keys` -- consistent with how the grid
    itself gets partitioned.
    """

    def __init__(self, x: Sequence[float], y: Sequence[float], z: Sequence[float], values: Sequence[float]):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.values = np.asarray(values, dtype=float)

        lengths = {len(self.x), len(self.y), len(self.z), len(self.values)}
        if len(lengths) != 1:
            raise ValueError("x, y, z, and values must all have the same length.")

        self.lith_id: Optional[np.ndarray] = None
        self.fault_id: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.values)

    @property
    def xyz(self) -> np.ndarray:
        return np.column_stack([self.x, self.y, self.z])

    def assign_domains(self, geo_model: gp.data.GeoModel, lith_array: np.ndarray, fault_array: np.ndarray) -> None:
        """Assign each sample a (lith_id, fault_id) domain via nearest-cell lookup."""
        from scipy.spatial import cKDTree

        cell_centers = geo_model.grid.regular_grid.values
        tree = cKDTree(cell_centers)
        _, nearest_idx = tree.query(self.xyz)

        self.lith_id = lith_array.ravel()[nearest_idx]
        self.fault_id = fault_array.ravel()[nearest_idx]

    def for_domain(self, domain_key: Union[DomainKey, DomainGroup]) -> "ConditioningData":
        """Return the subset of samples belonging to `domain_key`.

        Accepts either a bare `DomainKey` (single domain) or a `DomainGroup` (pools
        samples across every domain key in the group).

        Requires `assign_domains` to have been called first.
        """
        if self.lith_id is None or self.fault_id is None:
            raise RuntimeError("Call assign_domains() before for_domain().")

        mask = np.zeros(len(self), dtype=bool)
        for lith_id, fault_id in as_group(domain_key):
            mask |= (self.lith_id == lith_id) & (self.fault_id == fault_id)

        subset = ConditioningData(self.x[mask], self.y[mask], self.z[mask], self.values[mask])
        subset.lith_id = self.lith_id[mask]
        subset.fault_id = self.fault_id[mask]
        return subset
