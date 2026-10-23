# Geological property simulation — M1–M3

Experimental voxel-gradient anisotropy, sparse geological graphs, cached Landmark
Isomap, and unconditional Gaussian spectral simulation. Geometry and stochastic
simulation are separate modules. The Gaussian backend supports CPU and CUDA;
graph construction, Dijkstra, and landmark MDS currently use NumPy/SciPy.

## Quick start

```python
from gempy_plugins.property_estimation.examples.geological_models import make_geological_model
from gempy_plugins.property_estimation.geological_coordinates import GeologicalEmbedding
from gempy_plugins.property_estimation.gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from gempy_plugins.property_estimation.plotting import plot_property_field

model = make_geological_model("folded")  # Offline, actual GemPy interpolation
geometry = GeologicalEmbedding.from_gempy(
    model, unit="lower", anisotropy_ratio=5,
    landmarks=32, dimensions=6, stencil_radius=3,
)
geometry.save("limestone_geometry.npz")

simulator = GaussianFieldSimulator(
    geometry, GaussianCovariance(length_scale=5, variance=0.01),
    n_frequencies=512, basis_seed=0, device="cpu",  # or "cuda"
)
density = simulator.simulate_continuous(realizations=8, mean=2.65, seed=13)
plot_property_field(model, density.to_property_field(realization=0))
```

For an existing model, enable
`model.interpolation_options.evaluation_options.compute_scalar_gradient = True`
**before** `gp.compute_model(model)`. Use an explicitly evaluated dense regular
grid and enable full raw solution arrays. The adapter reads the selected
structural group's gradient components, verifies grid alignment, and applies the
physical-coordinate gradient chain rule. It does not recompute or modify the model.
Recompute after modifying the structural model; extraction uses its solved state.
Basement needs `structural_group="..."` to identify the field supplying orientation.
`fault_block=...` optionally selects one fault-block ID.

## Numerical conventions

- Grid and active-node mappings use NumPy C order.
- `anisotropy_ratio = lateral_scale / perpendicular_scale`, at least one.
- The metric is `(I - nn.T) / ratio**2 + nn.T`. Covariance scale is separate.
- For `GaussianCovariance(a, variance)`, `C(h) = variance * exp(-h² / (2a²))`.
  With constant orientation, perpendicular length scale is `a`, lateral scale is
  `ratio*a`. This convention is not necessarily GSTools' Gaussian scale convention.
- Edge cost is `sqrt(delta.T @ ((M_A + M_B)/2) @ delta)`. It approximates the
  midpoint tensor from voxel gradients; no off-grid field queries are needed.
- Normals are unoriented: changing their sign leaves costs unchanged. Nonfinite
  and near-zero gradients are excluded and counted. The threshold is relative to
  the largest finite gradient magnitude in the selected unit.
- Default connectivity is 26 with radius 1. Connectivity 6/18 and wider primitive
  lattice stencils are available. Every voxel touched by an edge must be active
  and in the same domain, including corner/edge contacts. Subvoxel geological
  barriers cannot be recovered from a voxel mask; resolve them on the input grid.
- Lithology and fault-block boundaries are impermeable. Each connected component
  receives its own landmarks, MDS projection, and independent Gaussian coefficients.
  Cross-component coordinate distances are undefined; finite covariance is zero.
- MDS double-centers squared landmark distances and uses positive eigenpairs.
  Requested coordinate width is retained with zero padding for low-rank components.
  Effective rank, discarded positive spectrum, and negative spectrum are reported.
- Landmark distances are computed one source at a time. SciPy's temporary result
  is float64, even with float32 graph weights. Stored landmark distances can be
  float32 or float64. Eigensolves and centering use float64; coordinates use float32.

## Validate geometry before selecting simulation settings

```python
from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField, AnisotropicGraph
from gempy_plugins.property_estimation.geological_coordinates.diagnostics import embedding_quality

metric = GeologicalMetricField.from_gempy(model, "lower", anisotropy_ratio=5)
graph = AnisotropicGraph.from_metric(metric, stencil_radius=3)
geometry = GeologicalEmbedding.from_graph(graph, landmarks=32, dimensions=6)
quality = embedding_quality(graph, geometry, length_scale=5)
print(quality)
```

The quality report samples held-out source nodes where available. It reports local
relative distance error up to three Gaussian length scales, covariance error
normalized to unit variance, tie-aware nearest-neighbor overlap, and a fixed
Cartesian-anisotropy comparison (normal `[0, 0, 1]` by default). Its provisional
gate is overlap >= 0.9, P90 relative distance error <= 0.15, and P90 covariance
error <= 0.05. These are engineering criteria for the sampled model and range.
Covariance evaluated on arbitrary graph distances is a pairwise validation target;
positive definiteness of a full graph-distance covariance matrix is not assumed.

### Initial folded-model results

Actual GemPy anticline, grid `24 × 10 × 24`, target unit `lower`, Gaussian scale 5,
32 farthest-point landmarks, 32 held-out validation sources, fixed seeds:

| Ratio | Stencil radius | Dimensions | Neighbor overlap | P90 distance error | P90 covariance error | Gate |
|---:|---:|---:|---:|---:|---:|:---|
| 20 | 1 | 6 | 0.645 | 0.295 | 0.134 | fail |
| 20 | 1 | 8 | 0.748 | 0.328 | 0.151 | fail |
| 20 | 3 | 6 | 0.756 | 0.151 | 0.070 | fail |
| 5 | 2 | 3 | 0.918 | 0.075 | 0.041 | pass |
| 5 | 3 | 6 | 0.928 | 0.078 | 0.041 | pass |

The smallest passing tested configuration above is radius 2 / 3 dimensions at
5:1. The six-dimensional demonstration uses radius 3 / 5:1. Increasing dimensions
alone can increase local distortion. **The original 20:1/26-neighbor default is
an experimental baseline, not a validated production setting.** Strong anisotropy
still needs further stencil/grid/landmark experiments. These results establish a
working folded-model demonstration, not general acceptance across geometries.
In the initial 12-case horizontal/dipping/folded matrix with six dimensions,
only the folded 5:1 / radius-3 case passed all three targets. At 5:1 / radius 3,
horizontal and dipping P90 covariance errors were 0.121 and 0.068 respectively.
Their parameter selection also needs further work; a wider graph alone does not
guarantee an adequate embedding.

Run the reproducible matrix (JSON includes timing, array sizes, spectrum and errors):

```bash
python -m gempy_plugins.property_estimation.benchmarks.geological_simulation --output benchmark.json
python -m gempy_plugins.property_estimation.benchmarks.geological_simulation \
  --models folded --ratios 5,20 --stencil-radii 1,2,3 \
  --landmarks 16,32,64 --dimensions 3,4,6,8 --output embedding_matrix.json
python -m gempy_plugins.property_estimation.benchmarks.geological_simulation \
  --models folded --ratios 5 --stencil-radii 3 --simulate \
  --features 256,512,1024,2048 --realizations 100 --device cuda --output spectral.json
```

Horizontal/dipping cases also report constant-tensor graph error separately from
embedding error. Process peak RSS is a whole-process high-water mark, not an
isolated per-stage memory measurement. CUDA runs report PyTorch peak allocated
memory. Spectral timing includes one warm-up realization and reports the median
of three subsequent ensemble timings, with the raw timings retained. CPU tests
have been executed; CUDA parity is tested automatically when
CUDA is available. No GPU speedup claim follows from the CPU results.

## Streaming, reproducibility and covariance consistency

`n_frequencies=F` produces **2F real sine/cosine columns**. The fixed basis uses
Gaussian frequencies; each realization draws Gaussian coefficients, giving exact
Gaussian marginals conditional on that basis and constant marginal variance.
The finite covariance is `variance/F * sum(cos(omega @ (q-q')))`. It approximates
the target Gaussian covariance. `simulator.finite_covariance(node_indices)` exposes
the actual model for future consistent conditioning.

```python
for batch in simulator.iter_simulate(
    realizations=100, realization_batch_size=8, point_chunk_size=8192, seed=42,
):
    # A Torch tensor, (batch_realizations, active_nodes), on CPU by default.
    consume(batch.values, batch.realization_start)
```

`simulate` materializes the full ensemble. `iter_simulate` bounds output storage
by the realization batch. GPU temporary basis storage scales with the point chunk,
not the entire grid. `output_device="cuda"` optionally retains results on GPU.
Coordinates are snapshotted on the host. Coefficients use independent reproducible
streams keyed by seed, component, and realization; changing batch/chunk sizes does
not change random draws. Floating-point reductions may differ slightly across
chunk shapes or devices.

One million points and 512 frequencies require about **4.1 GB** to cache both
sine and cosine columns in float32. Streaming is the default. Explicitly opt in
with `simulator.cache_basis(max_bytes=...)`; this budget covers persistent basis
storage. Allow additional memory for chunk temporaries. Use
`clear_basis_cache()` to release it. The basis fingerprint includes coordinates,
component IDs, covariance parameters, and actual frequencies.

## Cache reuse

```python
geometry.save("geometry.npz")
reloaded = GeologicalEmbedding.load("geometry.npz", expected_graph_fingerprint=graph.fingerprint)
```

The pickle-free, checksummed cache stores coordinates, mappings, per-component
landmarks/eigenvectors/centering factors, source metadata, and version information.
Supply `expected_graph_fingerprint` to reject a different grid/domain/orientation/
anisotropy/stencil. Loading without it deliberately loads a standalone saved
geometry. Treat embeddings as immutable after construction. Covariance range,
variance, and property mean do not change the embedding; the anisotropy ratio does.

`LandmarkProjection.transform` accepts distances to its component's landmarks.
Continuous physical-coordinate queries are a subsequent extension and must avoid
interpolation across incompatible domains.

## Visualization and tests

```bash
python -m gempy_plugins.property_estimation.examples.folded_geological_simulation
python -m gempy_plugins.property_estimation.examples.folded_geological_simulation \
  --no-show --output folded_density.png
python -m gempy_plugins.property_estimation.examples.folded_geological_simulation --volume
python -m pytest gempy_plugins/property_estimation/tests -q
```

The new numerical/integration tests use offline models. Existing GSTools tests
also cover compatibility and use their existing external geological-data fixture.
Running the full test directory additionally requires `pytest` and `gstools`.
New coverage includes normal sign invariance, physical-gradient transforms,
constant-tensor distances, wide-stencil barriers, disconnected components,
correct MDS extension, cache compatibility, finite-feature ensemble statistics,
CPU/CUDA parity, faulted GemPy domains, and bedding-directional variograms.

## Following milestones

M4 adds observation-coordinate queries and continuous conditioning against the
finite-feature covariance, with noise and rank handling. M5 adds categorical
latent-region conditioning. Fixed-path SGS, Matérn spectra, plurigaussian facies,
and large-model GPU profiling remain subsequent milestones. A finite basis has
rank at most 2F per component, which must be considered when honoring many exact
observations.

## References

- Boisvert & Deutsch (2011), *Programs for kriging and sequential Gaussian simulation
  with locally varying anisotropy using non-Euclidean distances*.
  DOI: [10.1016/j.cageo.2010.03.021](https://doi.org/10.1016/j.cageo.2010.03.021).
- Nussbaumer et al. (2018), *Accelerating Sequential Gaussian Simulation with a
  constant path*. DOI: [10.1016/j.cageo.2017.12.006](https://doi.org/10.1016/j.cageo.2017.12.006).
- Li et al. (2024), *Direct kriging: A direct optimization based model with locally
  varying anisotropy*. DOI: [10.1016/j.jhydrol.2024.131553](https://doi.org/10.1016/j.jhydrol.2024.131553).
