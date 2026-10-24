Property Estimation
----------------------

Populates a computed structural model's regular grid with a property (porosity, ore
grade, etc.) via GSTools kriging or simulation, respecting the model's lithology and
fault-block boundaries.

Geological-coordinate simulation (experimental)
-----------------------------------------------

``geological_coordinates`` builds reusable locally varying anisotropic distances
and Landmark Isomap coordinates from GemPy voxel gradients. ``gaussian_fields``
simulates Gaussian continuous properties in those coordinates using chunked
PyTorch spectral features, on CPU or CUDA. ``categorical_fields`` turns those fields
into ordered facies, with exact categorical observations handled through joint
latent-interval conditioning and domain-safe connectivity diagnostics.
Continuous fields support exact/noisy conditioning, posterior mean and variance,
and domain-safe off-grid observation queries using the cached coordinates.
Local range multipliers and anisotropy ratios can be supplied as voxel maps to
vary continuity across a model while retaining bedding-aligned orientations.
See ``GEOLOGICAL_SIMULATION.md`` for the API, analytical/refinement checks,
multi-geometry validation results, and runnable continuous/facies examples.

The mesh-native example reuses one solved GemPy evaluator to generate tetrahedra,
estimate world-coordinate gradients at their centroids, and simulate conditioned
porosity on a unit/fault-safe cell graph::

    python -m gempy_plugins.property_estimation.examples.tetrahedral_property_simulation

It provides a linked PyVista comparison and optional PNG, JSON and property-bearing
VTU exports. Gradients use fixed-fault-side central differences; the face-only
graph/Isomap geometry remains experimental and its accuracy report is displayed.

For categorical facies on the combination model (folds, fault and unconformity)::

    python -m gempy_plugins.property_estimation.examples.tetrahedral_facies

This remeshes with MMG3D by default (``mmgpy``; use ``--no-remesh`` to omit it),
conditions six synthetic centroid observations, and compares the geological mesh,
prior/conditional facies and sandstone probabilities. Press ``c`` for the cutaway
and ``e`` for tetrahedral edges. Facies occupy ``rock2`` and ``rock1`` by default.

The matching continuous-density example uses the same combination-model geometry::

    python -m gempy_plugins.property_estimation.examples.tetrahedral_density

It compares prior/conditioned density, posterior mean and posterior standard
deviation in g/cm^3, with four exact and two noisy synthetic cell measurements.

Install the optional dependencies with::

    pip install -e ".[geological-simulation,geological-simulation-viz]"
