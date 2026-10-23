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
into unconditional ordered facies, with domain-safe connectivity diagnostics.
See ``GEOLOGICAL_SIMULATION.md`` for the API, analytical/refinement checks,
multi-geometry validation results, and runnable continuous/facies examples.

Install the optional dependencies with::

    pip install -e ".[geological-simulation,geological-simulation-viz]"
