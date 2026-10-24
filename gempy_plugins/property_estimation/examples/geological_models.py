"""Small, offline GemPy models for simulation examples and numerical benchmarks."""
import numpy as np
import gempy as gp


MODEL_GEOMETRIES = ('horizontal', 'dipping', 'folded', 'syncline', 'gentle_fold',
                    'tight_fold', 'thin_layer', 'faulted')


def make_geological_model(geometry='folded', resolution=(24, 10, 24), *, compute_gradients=True,
                          fold_amplitude=None, layer_thickness=None):
    """Build two interpolated horizons from explicit contacts and orientations.

    All models use the actual GemPy interpolator. The input formula only generates
    observations; the resulting scalar field and gradients come from GemPy.
    ``gentle_fold``/``tight_fold`` vary curvature; ``thin_layer`` has six units of
    vertical separation instead of thirty. Amplitude and separation can also be
    supplied explicitly (physical units, not normal-to-bedding thickness).
    """
    if geometry not in MODEL_GEOMETRIES:
        raise ValueError("Unknown example geometry")
    amplitude = {'horizontal': 0., 'dipping': 0., 'folded': 18., 'syncline': -18., 'faulted': 18.,
                 'gentle_fold': 8., 'tight_fold': 30., 'thin_layer': 18.}[geometry]
    amplitude = amplitude if fold_amplitude is None else float(fold_amplitude)
    thickness = (6. if geometry == 'thin_layer' else 30.) if layer_thickness is None else float(layer_thickness)
    if not np.isfinite(amplitude) or not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("Fold amplitude must be finite and layer thickness finite and positive")
    dip = .3 if geometry == 'dipping' else 0.
    elements = []
    for name, offset, color in [('upper', 28. + thickness, '#e6c06e'), ('lower', 28., '#748da6')]:
        x, y = np.meshgrid([10., 30., 50., 70., 90.], [8., 32.], indexing='ij')
        x, y = x.ravel(), y.ravel()
        z = offset + amplitude * (1 - ((x - 50) / 50) ** 2) + dip * (x - 50)
        ox = np.array([10., 50., 90.])
        oz = offset + amplitude * (1 - ((ox - 50) / 50) ** 2) + dip * (ox - 50)
        slope = -2 * amplitude * (ox - 50) / 2500 + dip
        normal = np.column_stack((-slope, np.zeros(3), np.ones(3)))
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        elements.append(gp.data.StructuralElement(
            name=name, color=color,
            surface_points=gp.data.SurfacePointsTable.from_arrays(x=x, y=y, z=z, names=name),
            orientations=gp.data.OrientationsTable.from_arrays(
                x=ox, y=np.full(3, 20.), z=oz,
                G_x=normal[:, 0], G_y=normal[:, 1], G_z=normal[:, 2], names=name,
            ),
        ))
    group = gp.data.StructuralGroup(name='stratigraphy', elements=elements,
                                    structural_relation=gp.data.StackRelationType.ERODE)
    frame = gp.data.StructuralFrame(structural_groups=[group], color_gen=gp.data.ColorsGenerator())
    if geometry == 'faulted':
        fault = gp.data.StructuralElement(
            name='fault', color='#333333',
            surface_points=gp.data.SurfacePointsTable.from_arrays(
                x=np.full(4, 50.), y=np.array([4., 36., 4., 36.]), z=np.array([10., 10., 90., 90.]), names='fault'),
            orientations=gp.data.OrientationsTable.from_arrays(
                x=np.array([50.]), y=np.array([20.]), z=np.array([50.]),
                G_x=np.array([1.]), G_y=np.array([0.]), G_z=np.array([0.]), names='fault'),
        )
        frame.insert_group(0, gp.data.StructuralGroup(
            name='fault_series', elements=[fault], structural_relation=gp.data.StackRelationType.FAULT,
            fault_relations=gp.data.FaultsRelationSpecialCase.OFFSET_ALL,
        ))
    model = gp.create_geomodel(project_name=f'property_{geometry}', extent=[0, 100, 0, 40, 0, 100],
                              resolution=list(resolution), structural_frame=frame)
    model.update_transform(gp.data.GlobalAnisotropy.NONE)
    options = model.interpolation_options.evaluation_options
    options.mesh_extraction = False
    options.compute_scalar_gradient = compute_gradients
    gp.compute_model(model, engine_config=gp.data.GemPyEngineConfig(
        backend=gp.data.AvailableBackends.numpy, dtype='float64', use_gpu=False,
    ))
    return model
