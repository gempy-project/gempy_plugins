"""Domain-aware kriging: per-domain (or per-domain-group) GSTools kriging, reinjected
into a full-grid array."""
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

import gempy as gp
from gempy_plugins.optional_dependencies import require_gstools
from gempy_plugins.property_estimation.conditioning_data import ConditioningData
from gempy_plugins.property_estimation.domains import (
    DomainGroup, DomainKey, as_group, domain_mask, group_mask, validate_disjoint_groups,
)
from gempy_plugins.property_estimation.neighborhood import Neighborhood, local_neighbor_indices


@dataclass
class KrigingDomainConfig:
    """Kriging settings for one domain, or one group of domains processed together.

    Args:
        model: A GSTools `CovModel` instance (variogram) -- fully user-constructed and
            fit, passed straight through with no wrapping.
        krige_class: One of `gs.krige.Simple`/`Ordinary`/`Universal`. Defaults to
            `gs.krige.Ordinary` if not given.
        krige_kwargs: Extra kwargs for `krige_class` (e.g. `mean=` for Simple,
            `drift_functions=` for Universal).
        neighborhood: Moving-neighborhood strategy. Defaults to global ("all") kriging.
    """
    model: Any
    krige_class: Optional[Type] = None
    krige_kwargs: dict = field(default_factory=dict)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)


@dataclass
class PropertyField:
    """Result of a domain-aware kriging or simulation run.

    `values`/`variance` are full-regular-grid-shaped arrays, `nan` wherever a domain
    wasn't in `domain_configs` (or had no conditioning data). `variance` is left `nan`
    everywhere for simulation results, since a single stochastic realization has no
    associated estimation variance.
    """
    values: np.ndarray
    variance: np.ndarray
    domain_keys: List[DomainKey]


def run_kriging(
        geo_model: gp.data.GeoModel,
        conditioning_data: ConditioningData,
        domain_configs: Dict[Union[DomainKey, DomainGroup], KrigingDomainConfig],
) -> PropertyField:
    """Run kriging per-domain (or per-domain-group) and reassemble a full-grid property field.

    A domain is processed iff its key is present in `domain_configs`. A dict key may be
    a bare `DomainKey` (one domain) or a `DomainGroup` -- a tuple of several domain keys
    processed together, sharing one conditioning-data pool and one config (e.g. merging
    both fault blocks of a lithology across a negligible-offset fault, or merging two
    different lithologies that should share one property model). No domain key may
    appear in more than one group. Domains/groups with no matching cells in the model,
    or no conditioning data assigned to them, are skipped with a warning.
    """
    gs = require_gstools()
    validate_disjoint_groups(domain_configs.keys())

    resolution = tuple(geo_model.grid.regular_grid.resolution)
    raw_arrays = geo_model.solutions.raw_arrays
    lith_array = raw_arrays.lith_block.reshape(resolution)
    fault_array = raw_arrays.fault_block.reshape(resolution)
    cell_centers = geo_model.grid.regular_grid.values

    values = np.full(lith_array.size, np.nan)
    variance = np.full(lith_array.size, np.nan)
    populated_domains = []

    for group_key, config in domain_configs.items():
        group = as_group(group_key)
        mask = group_mask(lith_array, fault_array, group).ravel()
        if not mask.any():
            warnings.warn(f"Domain group {group} has no matching cells in this model, skipping.")
            continue

        subset = conditioning_data.for_domain(group)
        if len(subset) == 0:
            warnings.warn(f"Domain group {group} has no conditioning data assigned, skipping.")
            continue

        krige_class = config.krige_class or gs.krige.Ordinary
        query_xyz = cell_centers[mask]

        field_vals, field_var = _krige_domain(krige_class, config, subset, query_xyz)

        values[mask] = field_vals
        variance[mask] = field_var
        populated_domains.extend(k for k in group if domain_mask(lith_array, fault_array, k).any())

    return PropertyField(
        values=values.reshape(resolution),
        variance=variance.reshape(resolution),
        domain_keys=populated_domains,
    )


def _krige_domain(
        krige_class: Type,
        config: KrigingDomainConfig,
        subset: ConditioningData,
        query_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if config.neighborhood.mode == "all":
        krige = krige_class(
            config.model,
            cond_pos=subset.xyz.T,
            cond_val=subset.values,
            **config.krige_kwargs,
        )
        x, y, z = query_xyz[:, 0], query_xyz[:, 1], query_xyz[:, 2]
        return krige((x, y, z), mesh_type="unstructured")

    neighbor_idx_list = local_neighbor_indices(query_xyz, subset.xyz, config.neighborhood)
    field_vals = np.empty(len(query_xyz))
    field_var = np.empty(len(query_xyz))
    for i, (point, idx) in enumerate(zip(query_xyz, neighbor_idx_list)):
        krige = krige_class(
            config.model,
            cond_pos=subset.xyz[idx].T,
            cond_val=subset.values[idx],
            **config.krige_kwargs,
        )
        val, var = krige(
            (point[0:1], point[1:2], point[2:3]),
            mesh_type="unstructured",
        )
        field_vals[i] = val[0]
        field_var[i] = var[0]
    return field_vals, field_var
