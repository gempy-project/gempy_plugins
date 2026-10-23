"""Extract gradients on an explicitly evaluated GemPy dense grid."""
import numpy as np

from .metric import GeologicalMetricField


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def metric_from_gempy(model, unit, *, anisotropy_ratio=20.0, structural_group=None,
                      fault_block=None, gradient_tolerance=1e-12):
    """Read a computed model without re-solving or modifying its grids/options.

    ``unit`` is a structural element name. Basement needs an explicit
    ``structural_group`` to supply its orientation. Fault blocks remain separate
    even when ``fault_block=None`` selects all occurrences of the unit.
    """
    solutions = model.solutions
    if solutions is None:
        raise ValueError("Compute the GemPy model before extracting a geological metric")
    grid = model.grid.dense_grid
    if grid is None or model.grid.GridTypes.DENSE not in model.grid.active_grids:
        raise ValueError("An explicitly evaluated dense regular grid is required")
    raw = solutions.raw_arrays
    if raw is None or solutions.block_solution_type.name != 'DENSE_GRID':
        raise ValueError("Dense-grid lithology and fault-block solution arrays are required")
    groups = model.structural_frame.structural_groups
    outputs = solutions.octrees_output[0].outputs
    if len(groups) != len(outputs):
        raise ValueError("Structural groups changed since computation; recompute the model")
    names = list(model.structural_frame.elements_names)
    if unit not in names:
        raise ValueError(f"Unknown geological unit {unit!r}; available units: {names}")
    if structural_group is None:
        matches = [i for i, group in enumerate(groups) if any(element.name == unit for element in group.elements)]
        if len(matches) != 1:
            raise ValueError("Specify structural_group explicitly for basement or an ambiguous unit")
        stack = matches[0]
    else:
        matches = [i for i, group in enumerate(groups) if group.name == structural_group]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous structural group {structural_group!r}")
        stack = matches[0]
        if unit != 'basement' and not any(element.name == unit for element in groups[stack].elements):
            raise ValueError("The selected unit does not belong to structural_group")
    if groups[stack].structural_relation.name in ('FAULT', 'NULL_SPACE'):
        raise ValueError("Select a stratigraphic structural group to supply bedding orientation")
    output = outputs[stack]
    if output.grid.dense_grid is None:
        raise ValueError("The solution contains no evaluated dense-grid gradients")
    fields = output.exported_fields_dense_grid
    components = (fields.gx_field, fields.gy_field, fields.gz_field)
    if any(component is None for component in components):
        raise ValueError("Enable model.interpolation_options.evaluation_options.compute_scalar_gradient = True and recompute")
    shape = tuple(int(n) for n in grid.resolution)
    gradients = np.column_stack([_numpy(component).ravel() for component in components])
    positions = np.asarray(grid.values, dtype=np.float64)
    lithology = _numpy(raw.lith_block).ravel()
    faults = _numpy(raw.fault_block).ravel()
    if gradients.shape != positions.shape or len(lithology) != len(positions) or len(faults) != len(positions):
        raise ValueError("Gradient/domain arrays do not match the dense grid; recompute with full raw arrays enabled")
    # The combined map is world -> grid-local -> interpolation coordinates.
    # For column gradients g_world = A.T @ g_engine, hence g_engine @ A for rows.
    context = getattr(model, '_field_evaluation_context', None)
    input_transform = context.input_transform if context is not None else model.input_transform
    grid_transform = context.grid_transform if context is not None else grid.transform
    pivot = np.asarray(grid.extent)[::2]
    transformed = input_transform.apply(grid_transform.apply_with_pivot(positions, pivot))
    engine_positions = _numpy(output.grid.dense_grid.values)
    if transformed.shape != engine_positions.shape or not np.allclose(transformed, engine_positions, rtol=1e-6, atol=1e-8):
        raise ValueError("The physical grid does not match the computed gradient locations; recompute the model")
    linear = input_transform.get_transform_matrix()[:3, :3] @ grid_transform.get_transform_matrix()[:3, :3]
    world_gradients = gradients @ linear
    lith_id = names.index(unit) + 1
    mask = lithology == lith_id
    if fault_block is not None:
        mask &= faults == fault_block
    if grid.mask_topo.size:
        if grid.mask_topo.size != len(positions):
            raise ValueError("Topographic mask does not match the dense grid")
        mask &= ~np.asarray(grid.mask_topo, dtype=bool).ravel()
    if not mask.any():
        raise ValueError(f"Unit {unit!r} has no active voxels in the requested fault block")
    pairs, labels = np.unique(np.column_stack((lithology, faults)), axis=0, return_inverse=True)
    metadata = dict(unit=unit, structural_group=groups[stack].name, structural_group_index=stack,
                    domain_keys=pairs[np.unique(labels[mask])].astype(int).tolist(),
                    input_linear_transform=linear.tolist(), grid_extent=np.asarray(grid.extent).tolist(),
                    gradient_source='gempy_dense_grid')
    return GeologicalMetricField.from_gradients(
        positions, world_gradients, mask.reshape(shape), anisotropy_ratio=anisotropy_ratio,
        domain_labels=labels, gradient_tolerance=gradient_tolerance, metadata=metadata,
    )
