"""Domain-aware simulation: per-domain (or per-domain-group) GSTools SRF/CondSRF,
reinjected into a full-grid array. Mirrors kriging.py's domain loop, reusing the same
PropertyField result type."""
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, Union

import numpy as np

import gempy as gp
from gempy_plugins.optional_dependencies import require_gstools
from gempy_plugins.property_estimation.conditioning_data import ConditioningData
from gempy_plugins.property_estimation.domains import (
    DomainGroup, DomainKey, as_group, domain_mask, group_mask, validate_disjoint_groups,
)
from gempy_plugins.property_estimation.kriging import PropertyField
from gempy_plugins.property_estimation.neighborhood import Neighborhood


@dataclass
class SimulationDomainConfig:
    """Simulation settings for one domain, or one group of domains processed together.

    Args:
        model: A GSTools `CovModel` instance (variogram).
        conditioned: If True (default), condition the simulation on the conditioning
            data assigned to this domain (`gs.CondSRF`). If False, generate an
            unconditioned stochastic realization from `model` alone (`gs.SRF`) -- e.g.
            for a property field with a given variogram but no real samples driving it.
        krige_class: Underlying kriging method for conditioned simulation (one of
            `gs.krige.Simple`/`Ordinary`/`Universal`). Defaults to `gs.krige.Ordinary`.
            Unused if `conditioned=False`.
        krige_kwargs: Extra kwargs for `krige_class`.
        seed: Random seed for reproducibility.
        srf_kwargs: Extra kwargs for `gs.SRF`/`gs.CondSRF` (e.g. `generator=`).
        neighborhood: Only "all" is supported for conditioned simulation -- a single
            realization needs spatially consistent conditioning across the whole
            domain, which doesn't fit a per-query-point local neighborhood the way
            plain kriging does. Anything else warns and falls back to "all".
    """
    model: Any
    conditioned: bool = True
    krige_class: Optional[Type] = None
    krige_kwargs: dict = field(default_factory=dict)
    seed: Optional[int] = None
    srf_kwargs: dict = field(default_factory=dict)
    neighborhood: Neighborhood = field(default_factory=Neighborhood)


def run_simulation(
        geo_model: gp.data.GeoModel,
        conditioning_data: ConditioningData,
        domain_configs: Dict[Union[DomainKey, DomainGroup], SimulationDomainConfig],
) -> PropertyField:
    """Run simulation per-domain (or per-domain-group) and reassemble a full-grid property field.

    A domain is processed iff its key is present in `domain_configs`. A dict key may be
    a bare `DomainKey` (one domain) or a `DomainGroup` -- a tuple of several domain keys
    processed together, sharing one conditioning-data pool and one config. No domain key
    may appear in more than one group. Domains/groups with no matching cells in the
    model, or (for conditioned simulation) no conditioning data assigned to them, are
    skipped with a warning. `variance` is left `nan` everywhere -- a single stochastic
    realization has no associated estimation variance.
    """
    gs = require_gstools()
    validate_disjoint_groups(domain_configs.keys())

    resolution = tuple(geo_model.grid.regular_grid.resolution)
    raw_arrays = geo_model.solutions.raw_arrays
    lith_array = raw_arrays.lith_block.reshape(resolution)
    fault_array = raw_arrays.fault_block.reshape(resolution)
    cell_centers = geo_model.grid.regular_grid.values

    values = np.full(lith_array.size, np.nan)
    populated_domains = []

    for group_key, config in domain_configs.items():
        group = as_group(group_key)
        mask = group_mask(lith_array, fault_array, group).ravel()
        if not mask.any():
            warnings.warn(f"Domain group {group} has no matching cells in this model, skipping.")
            continue

        query_xyz = cell_centers[mask]
        x, y, z = query_xyz[:, 0], query_xyz[:, 1], query_xyz[:, 2]

        if config.conditioned:
            subset = conditioning_data.for_domain(group)
            if len(subset) == 0:
                warnings.warn(f"Domain group {group} has no conditioning data assigned, skipping.")
                continue
            if config.neighborhood.mode != "all":
                warnings.warn(
                    f"Domain group {group}: moving-neighborhood conditioning is not "
                    "supported for simulation; using all conditioning points in the domain."
                )
            krige_class = config.krige_class or gs.krige.Ordinary
            krige = krige_class(config.model, cond_pos=subset.xyz.T, cond_val=subset.values, **config.krige_kwargs)
            srf = gs.CondSRF(krige, **config.srf_kwargs)
        else:
            srf = gs.SRF(config.model, **config.srf_kwargs)

        field_vals = srf((x, y, z), seed=config.seed, mesh_type="unstructured")

        values[mask] = field_vals
        populated_domains.extend(k for k in group if domain_mask(lith_array, fault_array, k).any())

    return PropertyField(
        values=values.reshape(resolution),
        variance=np.full(lith_array.shape, np.nan),
        domain_keys=populated_domains,
    )
