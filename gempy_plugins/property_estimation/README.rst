Property Estimation
----------------------

Populates a computed structural model's regular grid with a property (porosity, ore
grade, etc.) via GSTools kriging or simulation, respecting the model's lithology and
fault-block boundaries.

Geological-coordinate simulation (experimental M1–M3)
---------------------------------------------------

``geological_coordinates`` builds reusable locally varying anisotropic distances
and Landmark Isomap coordinates from GemPy voxel gradients. ``gaussian_fields``
simulates Gaussian continuous properties in those coordinates using chunked
PyTorch spectral features, on CPU or CUDA. See ``GEOLOGICAL_SIMULATION.md`` for the
API, numerical conventions, validation results, and runnable examples.

Install the optional dependencies with::

    pip install -e ".[geological-simulation,geological-simulation-viz]"
