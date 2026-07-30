"""Domain partitioning for property estimation.

A "domain" is a (lithology id, fault-block id) pair -- combining both keeps property
estimation from bleeding across a fault, even where the same lithology occurs on both
sides of it (something the old kriging plugin's lithology-only masking couldn't do).
"""
from typing import Dict, List, Tuple, Union

import numpy as np

import gempy as gp

DomainKey = Tuple[int, int]
DomainGroup = Tuple[DomainKey, ...]


def as_group(key: Union[DomainKey, DomainGroup]) -> DomainGroup:
    """Normalize a bare `DomainKey` or a `DomainGroup` into a `DomainGroup`.

    A `DomainKey` is a 2-tuple of ints; a `DomainGroup` is a tuple of such 2-tuples --
    so a bare key is detected by its first element not itself being a tuple.
    """
    if isinstance(key[0], tuple):
        return tuple(key)
    return (key,)


def compute_domains(geo_model: gp.data.GeoModel) -> Tuple[np.ndarray, np.ndarray, List[DomainKey]]:
    """Compute per-cell lithology/fault-block arrays and the distinct domains present.

    Uses gempy's own ``lith_block``/``fault_block`` solution arrays directly (each a
    single consistent id per structural element/fault-block, regardless of how many
    structural series/groups the model has).

    Args:
        geo_model: A computed GemPy model (dense/regular grid only).

    Returns:
        lith_array: Per-cell lithology id, shaped like the regular grid's resolution.
        fault_array: Per-cell fault-block id, same shape.
        domain_keys: Sorted list of distinct (lith_id, fault_block_id) pairs present.
    """
    resolution = tuple(geo_model.grid.regular_grid.resolution)
    raw_arrays = geo_model.solutions.raw_arrays

    lith_array = raw_arrays.lith_block.reshape(resolution)
    fault_array = raw_arrays.fault_block.reshape(resolution)

    return lith_array, fault_array, domain_keys_from_arrays(lith_array, fault_array)


def domain_keys_from_arrays(lith_array: np.ndarray, fault_array: np.ndarray) -> List[DomainKey]:
    """Sorted list of distinct (lith_id, fault_block_id) pairs present in the arrays."""
    combined = np.stack([lith_array.ravel(), fault_array.ravel()], axis=-1)
    unique_pairs = np.unique(combined, axis=0)
    return [(int(lith_id), int(fault_id)) for lith_id, fault_id in unique_pairs]


def domain_mask(lith_array: np.ndarray, fault_array: np.ndarray, domain_key: DomainKey) -> np.ndarray:
    """Boolean mask (same shape as `lith_array`) selecting the cells in `domain_key`."""
    lith_id, fault_id = domain_key
    return (lith_array == lith_id) & (fault_array == fault_id)


def group_mask(lith_array: np.ndarray, fault_array: np.ndarray, group: Union[DomainKey, DomainGroup]) -> np.ndarray:
    """Boolean mask selecting the union of cells across every domain key in `group`.

    Accepts either a bare `DomainKey` (single domain) or a `DomainGroup` (several
    domain keys processed/populated together, e.g. merging both fault blocks of a
    lithology across a negligible-offset fault, or merging two different lithologies
    that should share one property model).
    """
    mask = np.zeros_like(lith_array, dtype=bool)
    for domain_key in as_group(group):
        mask |= domain_mask(lith_array, fault_array, domain_key)
    return mask


def validate_disjoint_groups(group_keys) -> None:
    """Raise if any `DomainKey` appears in more than one group -- ambiguous otherwise,
    since two different configs would both claim the same cells."""
    seen: Dict[DomainKey, DomainGroup] = {}
    for raw_key in group_keys:
        group = as_group(raw_key)
        for domain_key in group:
            if domain_key in seen and seen[domain_key] != group:
                raise ValueError(
                    f"Domain {domain_key} appears in more than one group "
                    f"({seen[domain_key]} and {group}) -- ambiguous which config applies."
                )
            seen[domain_key] = group


def describe_domains(geo_model: gp.data.GeoModel, domain_keys: List[DomainKey]) -> Dict[DomainKey, str]:
    """Human-readable name per domain key, e.g. (2, 0) -> "rock3 (fault block 0)".

    `lith_block` ids run from `n_faults + 1` to `n_elements`, in `elements_names` order
    -- i.e. fault elements first, then lithology elements (the same assumption
    `topology_analysis.get_lith_ids` relies on, confirmed against `lith_block`'s actual
    values).
    """
    structural_frame = geo_model.structural_frame
    n_faults = int(np.sum(structural_frame.group_is_fault))
    lith_names = structural_frame.elements_names[n_faults:]
    lith_ids = list(range(n_faults + 1, structural_frame.n_elements + 1))
    lith_id_to_name = dict(zip(lith_ids, lith_names))

    fault_ids_present = {fault_id for _, fault_id in domain_keys}
    show_fault_suffix = len(fault_ids_present) > 1

    descriptions = {}
    for domain_key in domain_keys:
        lith_id, fault_id = domain_key
        name = lith_id_to_name.get(lith_id, f"lith {lith_id}")
        if show_fault_suffix:
            name = f"{name} (fault block {fault_id})"
        descriptions[domain_key] = name
    return descriptions
