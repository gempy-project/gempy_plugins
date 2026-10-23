# Geological property simulation — coordinates, Gaussian fields, and facies

Experimental voxel-gradient anisotropy, sparse geological graphs, cached Landmark
Isomap, unconditional Gaussian spectral simulation, and truncated-Gaussian facies. Geometry and stochastic
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
    landmarks=32, dimensions=3, stencil_radius=3,
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
Cartesian-anisotropy comparison (dominant unoriented bedding normal by default;
pass `cartesian_normal=(0, 0, 1)` to reproduce the original horizontal comparator).
The comparator uses the principal eigenvector of the mean normal outer product,
so normal sign flips cannot cancel it. Its orientation and angular spread are
included in the report. Its provisional
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

Horizontal/dipping cases now report graph and embedding error against an analytical
constant tensor on identical local pairs. This replaces the original whole-box
`constant_tensor_graph_relative_p90` statistic; the new results live under
`constant_metric_reference`. Process peak RSS is a whole-process high-water mark, not an
isolated per-stage memory measurement. CUDA runs report PyTorch peak allocated
memory. Spectral timing includes one warm-up realization and reports the median
of three subsequent ensemble timings, with the raw timings retained. CPU tests
have been executed; CUDA parity is tested automatically when
CUDA is available. No GPU speedup claim follows from the CPU results.

### Analytical diagnosis and broader validation

For constant tensors, `q = M**(1/2) @ x` is an exact unconstrained coordinate
transform. `GeologicalEmbedding.from_constant_metric(graph)` supplies this reference
to the existing spectral simulator and cache API. It checks that the metric really
is constant (or isotropic) and preserves independent geological components. Within
a component it ignores holes/nonconvex boundaries, so its distances are lower bounds
for constrained paths. It is not substituted automatically for Isomap.

`constant_metric_quality(graph, embedding, length_scale=5)` separates graph error
from embedding error against that analytical reference. Both use the same held-out
pairs within three analytical length scales. The original `embedding_quality`
continues to use graph-local pairs, so its pair set can differ.

The horizontal/dipping GemPy normals agree with the expected constant orientations.
The failures are attributable to graph-direction restrictions and non-Euclidean
graph distances that MDS does not preserve locally. Extra positive MDS modes can
represent lattice-distance distortion rather than useful physical dimensions.
For example, horizontal radius 3 / seed 0 covariance error against the graph is
0.068 in 3D, 0.121 in 6D, and 0.152 in 8D. More dimensions are not a universal fix.

Follow-up grid `24 × 10 × 24`, ratio 5, scale 5, 32 landmarks, dimensions 3,
landmark seeds 0/1/2 (ranges across seeds):

| Model | Radius | Neighbor overlap | P90 graph-relative distance error | P90 covariance error vs graph | Passing seeds |
|---|---:|---:|---:|---:|---:|
| Horizontal | 5 | 0.975–0.990 | 0.052–0.056 | 0.028–0.031 | 3/3 |
| Dipping | 5 | 0.932–0.980 | 0.033–0.054 | 0.018–0.027 | 3/3 |
| Anticline (`folded`) | 3 | 0.932–0.936 | 0.057–0.059 | 0.030–0.031 | 3/3 |
| Syncline | 3 | 0.932–0.939 | 0.055–0.066 | 0.030–0.035 | 3/3 |
| Gentle fold | 3 | 0.953–0.957 | 0.060–0.064 | 0.034–0.037 | 3/3 |
| Tight fold | 3 | 0.873–0.881 | 0.083–0.091 | 0.046–0.050 | 0/3 |
| Thin layer | 3 | 0.713–0.750 | 0.259–0.261 | 0.136–0.146 | 0/3 |
| Fault-separated fold | 3 | 0.908–0.934 | 0.061–0.080 | 0.036–0.047 | 3/3 |

The horizontal/dipping radius-5 3D embeddings also have P90 covariance errors of
0.030–0.032 and 0.017–0.021 **against the analytical metric**, respectively.
At seed 0, their graph storage is 6.02 MB and 5.94 MB. These settings are
exploratory selections, not a general replacement for validating each geometry.
All 18 tested ratio-20 curved/faulted/thin cases (six models × three seeds,
radius 3 / dimensions 3) failed at least one gate.

Resolution/range experiments used `24 × 10 × 24` and `48 × 20 × 48`, scales
2.5/5/10, dimensions 3/6, and seeds 0/1 for folded, thin-layer, and fault-separated
models. The 3D anticline passed all those settings. Thin-layer distance/covariance
errors improved with refinement, but neighborhood overlap remained below 0.9;
even its finer 6D settings reached only 0.807–0.828. The thin unit has six physical
units of vertical separation and is only one or two voxels thick on the coarse grid.

There is also a separate graph-convergence check, `graph_refinement_quality`.
Odd refinement factors preserve the original voxel centers exactly, avoiding
nearest-voxel snapping error. Comparing radius-3 `24 × 10 × 24` graphs with
radius-5 `72 × 30 × 72` graphs, ratio/scale 5 and eight source nodes:

| Model | Matched nodes | Local pairs | P90 distance error | P90 covariance error |
|---|---:|---:|---:|---:|
| Anticline | 1,724 | 8,916 | 0.073 | 0.035 |
| Thin layer | 336 | 2,147 | 0.574 | 0.256 |

No active-node or reachability mismatches occurred in these runs. Normal differences
were approximately 1e-6 degrees. The thin-layer problem therefore already appears
in the graph, before embedding. The finer anticline graph has 46,660 active nodes
and occupies 323 MB; wide-stencil scaling needs attention. This is a convergence
reference, not exact continuum truth, and these error percentiles cannot be added
to embedding or spectral percentiles to claim an end-to-end bound.

Reproduce these experiments:

```bash
python -m gempy_plugins.property_estimation.benchmarks.geological_simulation \
  --models horizontal,dipping --ratios 5 --stencil-radii 1,3,5 \
  --dimensions 3,4,6,8 --seeds 0,1,2 --summary --output analytical_geometry.json
python -m gempy_plugins.property_estimation.benchmarks.geological_simulation \
  --models folded,syncline,gentle_fold,tight_fold,thin_layer,faulted \
  --ratios 5,20 --stencil-radii 3 --dimensions 3 --seeds 0,1,2 \
  --facies --features 512 --realizations 64 --summary --output expanded_geometry.json
python -m gempy_plugins.property_estimation.benchmarks.geological_simulation \
  --models folded,thin_layer,faulted --ratios 5 --stencil-radii 3 --dimensions 3,6 \
  --seeds 0,1 --resolutions 24,10,24 48,20,48 --length-scales 2.5,5,10 \
  --summary --output resolution_geometry.json
python -m gempy_plugins.property_estimation.benchmarks.graph_refinement \
  --model folded --output folded_refinement.json
python -m gempy_plugins.property_estimation.benchmarks.graph_refinement \
  --model thin_layer --output thin_refinement.json
```

### Unconditional truncated-Gaussian facies

```python
from gempy_plugins.property_estimation.categorical_fields import TruncatedGaussianFacies

facies_simulator = TruncatedGaussianFacies(
    simulator, proportions=(0.3, 0.4, 0.3), names=('shale', 'limestone', 'sandstone'),
)
facies = facies_simulator.simulate(realizations=64, seed=13)
codes = facies.to_grid(realization=0)  # 0/1/2, with -1 outside the selected unit
field = facies.to_property_field()    # floating codes, NaN outside the unit
```

The wrapper fixes the latent mean to zero and scales thresholds by the simulator's
standard deviation. It targets Gaussian quantiles at cumulative proportions,
without ranking realizations or imposing exact quotas. `iter_simulate` supports the
same seed, chunking, batch, and output-device options as the Gaussian backend.
Codes/names are property categories, independent of GemPy structural-element IDs.

`facies_diagnostics(graph, facies)` reports ensemble proportions with standard errors
computed over independent realizations, facies cluster sizes/counts, contact counts,
and along/across-bedding agreement. Morphology uses **six-connected voxel faces**,
not long graph edges. Different geological components never connect, including
touching fault blocks with identical facies codes. Upper-triangular contact counts
count each undirected face once and are counts, not physical interface areas.

The default demo uses 1,024 frequencies and 64 realizations. On the anticline,
observed proportions were 0.289/0.429/0.282 (targets 0.3/0.4/0.3); same-facies
probability was 0.898 along bedding and 0.605 across bedding, at mean face separations
4.05 and 4.17. The fault-separated example excludes 70 cross-block faces from
connectivity/contact diagnostics. Failed geometry gates remain visible in both
the JSON and plot titles. These statistics describe a demonstration, not a calibrated
sedimentary process or an acceptance gate for categorical realism.

One latent field imposes ordered categories along continuous paths. Voxel sampling
can miss an intermediate category and produce nonadjacent face contacts. Flexible
contact rules require plurigaussian or other categorical models. Conditional facies
are not implemented by this unconditional truncation.

```bash
python -m gempy_plugins.property_estimation.examples.truncated_gaussian_facies \
  --no-show --output facies.png --report facies.json
```

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
python -m gempy_plugins.property_estimation.examples.experiment_views --view facies
python -m gempy_plugins.property_estimation.examples.experiment_views --view distances
python -m pytest gempy_plugins/property_estimation/tests -q
```

The interactive experiment views show a matched continuous/facies realization,
independent fault blocks, the failing thin layer, and original/improved horizontal
covariance-error maps against the analytical metric. Run the two views in separate
processes for simultaneous native PyVista windows. Cameras are linked within each
window; drag to rotate, scroll to zoom, press `c` to toggle cutaway, and `q` to close.
Use `--no-show --screenshot view.png` to render a view headlessly.

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
latent-region conditioning. Unconditional facies are now available as above.
Fixed-path SGS, a conventional Cartesian FFT baseline, Matérn spectra, plurigaussian
facies, and large-model GPU profiling remain subsequent milestones. The current
backend uses random Fourier features at irregular embedded coordinates, not a
regular-grid FFT. A finite basis has
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
