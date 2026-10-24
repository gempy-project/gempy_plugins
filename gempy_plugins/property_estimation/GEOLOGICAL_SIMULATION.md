# Geological property simulation — coordinates, Gaussian fields, and facies

Experimental voxel-gradient anisotropy, sparse geological graphs, cached Landmark
Isomap, spatially varying continuity scales, conditional/unconditional Gaussian
spectral simulation, and interval-conditioned truncated-Gaussian facies. Geometry and stochastic
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
- Both `anisotropy_ratio` and `continuity_scale` accept scalars or full-grid arrays
  (grid-shaped or flat C order). The ratio is >= 1; continuity scales are > 0.
- The metric is `((I - nn.T) / ratio**2 + nn.T) / continuity_scale**2`.
  `continuity_scale=1` retains the original convention.
- For `GaussianCovariance(a, variance)`, `C(h) = variance * exp(-h² / (2a²))`.
  Intended local perpendicular length scale is `a*continuity_scale`, lateral scale
  is `ratio*a*continuity_scale`. Changing these scales changes continuity, not
  marginal variance. This is not necessarily GSTools' Gaussian scale convention.
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
For scale maps, the fixed Cartesian comparator uses the mean parallel/perpendicular
tensor eigenvalues. Its two resulting scales are reported explicitly; it is a
representative constant comparator, not an analytical solution of the varying metric.

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
use the interval-conditioning API below.

```bash
python -m gempy_plugins.property_estimation.examples.truncated_gaussian_facies \
  --no-show --output facies.png --report facies.json
```

### Conditional facies through latent intervals

```python
facies_model = TruncatedGaussianFacies(
    simulator, (.3, .4, .3), names=("shale", "limestone", "sandstone"),
)
posterior = facies_model.condition(
    observations=[100, 200, 300],  # active-node indices, or geometry.query(graph, xyz)
    codes=[2, 0, 2],              # exact observed category codes, in name order
    sampler="rejection",
)
fields = posterior.simulate(realizations=128, seed=13)
probabilities = fields.probabilities()  # (categories, active_nodes), empirical posterior
print(fields.observation_codes)        # requested categories, in original input order
print(fields.latent_observation_values)
```

Each category constrains the latent Gaussian observation to its threshold interval.
The algorithm samples the **joint** truncated Gaussian at observation locations,
then uses the finite-feature Gaussian conditional simulator for the rest of the
field. Observed latent values vary across realizations; fixed interval midpoints
would produce a different, over-constrained posterior. Prior latent mean is zero;
thresholds scale with the configured standard deviation, preserving category
probabilities when only latent variance changes. There is no category-noise model.

The default `sampler="rejection"` produces independent conditional draws, subject
to numerical precision, and is useful for modest observation sets. Each component
is sampled separately. `max_proposals=100000` bounds proposals per component per
realization. An exhausted budget raises an error; the sampler never silently
substitutes an approximate result. Rare/incompatible category patterns can exhaust
this budget. Consistent duplicate locations are merged; conflicting category
observations at the same embedded location/component are rejected.

For harder problems, `sampler="gibbs", burn_in=256` uses a separate independently
initialized truncated-normal Gibbs chain per realization. This requires full-rank,
numerically positive-definite observation covariance after duplicate removal.
Log-tail inverse-CDF sampling handles extreme truncated intervals. Gibbs results
are **approximate at finite burn-in**: increasing burn-in and comparing independent
runs is necessary for difficult, strongly correlated constraints. Its burn-in and
sampler are recorded in metadata; there is no automatic convergence certification.
Rejection also supports singular finite bases when the constraints are feasible.

Both samplers reuse the continuous coefficient-space update, preserve independent
components, support realization/point batching, and keep random streams stable
across batch sizes. Observation sampling currently runs on CPU; field evaluation
can use CUDA. Float64 is recommended. A numerical threshold-crossing failure raises
an error instead of overwriting category codes to disguise a latent inconsistency.

`fields.proportions` remain the **prior** probabilities. Posterior proportions can
change after conditioning. `probabilities()` estimates them from the ensemble;
with 128 independent draws its worst-case pointwise Monte Carlo standard error is
about 0.044. Full ensembles are optional: `iter_simulate` permits streamed counts.

### Spatially varying continuity experiments

Keep orientation supplied by the same GemPy gradients and give the metric a range map:

```python
import numpy as np
from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField, AnisotropicGraph

x = model.grid.dense_grid.values[:, 0]
t = np.clip((x - 20) / 60, 0, 1)
blend = t*t*(3 - 2*t)
metric = GeologicalMetricField.from_gempy(
    model, "lower", anisotropy_ratio=5,
    continuity_scale=.6 + .8*blend,
)
graph = AnisotropicGraph.from_metric(metric, stencil_radius=3)
geometry = GeologicalEmbedding.from_graph(graph, landmarks=32, dimensions=3)
simulator = GaussianFieldSimulator(geometry, GaussianCovariance(5), dtype="float64", n_frequencies=1024)
```

This changes intended local ranges from **15 parallel / 3 perpendicular** on the
left to **35 parallel / 7 perpendicular** on the right, retaining 5:1 anisotropy.
Alternatively `anisotropy_ratio=2 + 6*blend, continuity_scale=1` changes only the
parallel range, from 10 to 40, leaving perpendicular range 5. These maps enter
endpoint-averaged tensors, graph distances, and Isomap before simulation. Paths can
prefer higher-continuity regions; pairwise distances are not simply a local range
average. Complete maps enter graph/cache fingerprints, while embedding metadata
stores their extrema. Changing either map requires rebuilding the graph/embedding.

The demonstration compares uniform, range-map, ratio-map, and faulted-range cases
on the same `24 x 10 x 24` geometries, with 1,024 frequencies and 128 draws. Regional
statistics use equal **8-unit Y lags**, tangent to bedding, and *unconditional*
ensembles so conditioning does not confound the continuity comparison:

| Case | Intended local correlation left/right | Finite-feature correlation left/right | Same-facies agreement left/right |
|---|---:|---:|---:|
| Uniform 5:1 | .949 / .949 | .941 / .943 | .810 / .811 |
| Range multiplier .6–1.4 | .866 / .973 | .854 / .967 | .700 / .854 |
| Ratio 2–8 | .733 / .979 | .707 / .976 | .591 / .880 |
| Faulted range map | .866 / .973 | .856 / .973 | .701 / .870 |

All categorical observations were honored in every conditional draw. In the faulted
case, all 862 unobserved-block nodes retained their seed-matched prior codes exactly.
The intended/embedded/finite-feature correlations are reported separately in JSON.

**Geometry accuracy remains provisional.** At radius 3 / 3D / landmark seed 0,
the range-map, ratio-map, and faulted-range cases had neighbor overlaps
.877/.879/.863 and graph-relative covariance P90 errors .074/.049/.062. All missed
at least one gate, despite reproducing the regional continuity contrast. A 36-case
follow-up across those three cases, radii 3/5, dimensions 3/4/6, and landmark seeds
0/1 did not pass all targets either. Gentler range multipliers .8–1.2 and .7–1.3
(radius 3 / 3D / seeds 0/1/2) also missed a gate. These results do not establish
generally accurate local metrics in an Isomap embedding; the plots retain this
status. A separate one-dimensional test verifies variable-scale graph distances,
Isomap coordinates, covariance contrast, and unchanged point variance exactly or
to numerical tolerance.

```bash
python -m gempy_plugins.property_estimation.examples.local_continuity_facies \
  --no-show --output local_facies.png --report local_facies.json
python -m gempy_plugins.property_estimation.examples.local_continuity_facies --cases range_map --volume
python -m gempy_plugins.property_estimation.examples.local_continuity_facies \
  --cases range_map --continuity-bounds .8 1.2 --landmark-seed 1 \
  --dimensions 4 --stencil-radius 5 --no-show --report gentler_map.json
```

The section comparison shows intended parallel range, prior facies, conditioned
facies, and conditional sandstone probability. The PyVista view uses the same four
panels and linked-camera/cutaway controls. `--anisotropy-bounds LEFT RIGHT` controls
the ratio-map endpoints independently. `--seed` controls realizations;
`--landmark-seed` controls the embedding.

## Continuous conditioning

Prepare a reusable Gaussian posterior from measurements in property units:

```python
# Indices refer to active embedding nodes, not flattened full-grid indices.
density = GaussianFieldSimulator(
    geometry, GaussianCovariance(length_scale=5, variance=0.1**2),
    n_frequencies=1024, dtype="float64",
)
posterior = density.condition(
    observations=[100, 200, 300], values=[2.80, 2.49, 2.72], mean=2.65,
    noise_variance=[0., 0., 0.03**2],
)
draws = posterior.simulate(realizations=64, seed=13)
print(draws.observation_values)  # latent values at observation locations
moments = posterior.moments()   # analytical mean and marginal latent variance
field = moments.to_property_field()  # both values and variance, NaN outside the unit
```

`mean` is a known constant prior mean (simple Gaussian conditioning). Noise entries
are **variances**, not standard deviations, in squared property units. Zero denotes
an exact observation; positive entries describe independent Gaussian measurement
errors. The posterior's latent draws do not need to equal noisy measurements.
`moments().variance` excludes measurement noise. Mean and noise are fixed when the
posterior is prepared. Continuous conditioning does not impose categorical facies
observations or bounded physical properties.

For physical locations, keep the graph used to build the embedding:

```python
from gempy_plugins.property_estimation.geological_coordinates import GeologicalMetricField, AnisotropicGraph

metric = GeologicalMetricField.from_gempy(model, "lower", anisotropy_ratio=5)
graph = AnisotropicGraph.from_metric(metric, stencil_radius=3)
geometry = GeologicalEmbedding.from_graph(graph, landmarks=32, dimensions=3)
observations = geometry.query(graph, xyz)  # finite (observations, 3) physical array
density = GaussianFieldSimulator(geometry, GaussianCovariance(5, 0.1**2))
posterior = density.condition(observations, measured_density, mean=2.65)
```

Queries interpolate **geological coordinates**, then evaluate the spectral basis
there; they do not interpolate sampled voxel values or snap measurements to centers.
Every positive-weight trilinear corner must be active and in the same geological
component. Queries outside voxel-center bounds or whose support touches holes,
different units, or fault blocks are rejected. Exact center queries reproduce the
stored coordinates. This is a conservative voxel-resolution extension, not a new
shortest-path solve or a claim of subvoxel geometric accuracy. Points carry an
embedding fingerprint to reject reuse with incompatible coordinates. The same
restriction applies to the analytical reference's query interpolation.

The conditional model uses the **actual finite-feature covariance**. For each
component, with observation basis `B`, independent noise covariance `R`, and
`G = B B.T + R`, coefficient draws use the Matheron update
`w_cond = w + B.T G^+ (y - mean - B w - epsilon)`. Here `w` and measurement noise
`epsilon` are independent Gaussian draws. Using the same seed as the unconditional
simulator preserves the prior realization, and unobserved components remain exactly
unchanged for identical chunk/batch settings. Cross-component influence is zero.

Preparation uses float64 eigensolves and a relative eigenvalue cutoff
`rank_tolerance=1e-12` (with a machine-precision floor). There is no hidden jitter.
Consistent duplicate exact observations are allowed; values inconsistent with the
retained covariance range raise an error rather than being silently relaxed.
`posterior.diagnostics` reports retained ranks and condition numbers per observed
component. A finite basis has rank at most `2F` per component. In poorly conditioned
problems, increase feature count or review noise/rank settings; float64 simulation
is useful for tight honoring tolerances. All honoring is subject to numerical
precision and the chosen rank cutoff.

`posterior.iter_simulate` supports the same realization batching, point chunking,
seed, and output-device options as unconditional simulation. Each batch includes
`observation_values` in input order, including off-grid observations. The observation
system is factored once and reused across draws. Its memory cost includes observation
basis/factors and a dense observation covariance per component; no grid-by-observation
covariance matrix is stored. Large observation sets still require further scaling work.

The example uses successful 5:1 / radius-3 / 3D settings with two exact off-grid
density observations and one noisy observation. The fault example places all three
on one side so the unchanged opposite block is directly visible:

```bash
python -m gempy_plugins.property_estimation.examples.conditioned_geological_simulation \
  --no-show --output conditioning.png --report conditioning.json
python -m gempy_plugins.property_estimation.examples.conditioned_geological_simulation --volume
python -m gempy_plugins.property_estimation.examples.conditioned_geological_simulation --models folded --volume
```

Sections and the linked PyVista view compare a seed-matched prior draw, conditioned
draw, posterior mean, and posterior standard deviation. PyVista controls are the
same as the experiment views: drag/scroll, `c` cutaway, `q` close. The red fault
plane marks a barrier with no imposed displacement. Example measurements are
synthetic; the JSON reports exact-observation residuals and untouched-block changes.

In the default 64-realization CPU/float64 run, both models honored their exact
off-grid observations with maximum error `4.4e-16`. All 862 nodes in the unobserved
fault block were unchanged, with maximum seed-matched difference exactly zero.
The noisy observation retained latent ensemble standard deviations of 0.020
(folded) and 0.019 (faulted). Separate ensemble tests compare means and full
covariances with dense Gaussian posterior formulas, including mixed exact/noisy data.

## Mesh-native tetrahedral example

`examples/tetrahedral_property_simulation.py` implements the mesh-first path:

1. Solve the offline folded/faulted GemPy model once, with solve-grid gradients
   disabled, and prepare `gp.prepare_field_evaluator(model)`.
2. Pass that **same evaluator** to `volume_mesh_from_evaluator`, with a matching
   world-coordinate extent. This lower-level entry point reuses the solution;
   the high-level `compute_volume_mesh` currently performs its own solve.
3. Evaluate each selected tetrahedron's scalar field and gradient at its centroid,
   using the unit's structural stack and the mesh's explicit fault-block sides.
4. Construct bedding-aligned tensors with ratio 5 and a smooth X-dependent local
   continuity multiplier from 0.8 to 1.2.
5. Build a shared-face cell graph and component-wise Landmark Isomap coordinates.
6. Simulate a latent Gaussian, condition three synthetic centroid measurements,
   transform to porosity, and attach the arrays as tetrahedral **cell data**.

```bash
# Interactive PyVista: prior, conditioned draw, posterior median and uncertainty.
python -m gempy_plugins.property_estimation.examples.tetrahedral_property_simulation

# Headless PNG, JSON diagnostics, property-bearing VTU and boundary VTP.
python -m gempy_plugins.property_estimation.examples.tetrahedral_property_simulation \
  --no-show --output tetrahedral.png --report tetrahedral.json \
  --mesh-output tetrahedral.vtu

# Optional MMG3D improvement (requires mmgpy); --cells counts background cubes.
python -m gempy_plugins.property_estimation.examples.tetrahedral_property_simulation \
  --cells 2000 --remesh
```

`--geometry folded` removes the fault barrier. The default `faulted` model has
folded bedding and a barrier at X=50, with no imposed throw. The viewer supports
linked drag/scroll, `c` for cutaway and `q` to close. `--no-show` without `--output`
runs the numerical example without creating a plotter. The VTU retains all mesh
cells, unit/block labels, selected-cell scalars/gradients, bedding normals, local
continuity, prior/conditional porosity, posterior median and 5th/95th percentiles,
and synthetic observations. Unsimulated property cells are NaN.

### Gradient and graph details

The public prepared evaluator currently returns scalar fields only. The plugin's
`geological_coordinates.tetrahedral.evaluate_tetrahedral_fields` therefore uses
**central differences of the fixed-weight scalar evaluator**, with default
`step=0.001` in world-coordinate units. Both offsets retain the cell's fault-side
labels, so the derivative follows its smooth one-sided field rather than crossing
the fault jump. Differencing in physical coordinates also incorporates GemPy's
coordinate transforms. A sampled half-step comparison is included in every run.
This is a numerical derivative, not a newly exposed analytic engine-gradient API.

`AnisotropicGraph.from_tetrahedral_mesh(mesh, gradients, mask=..., ...)` supports
scalar or full-cell anisotropy/continuity maps. Only face-sharing cells with the
same unit **and** fault block are joined. Each edge follows centroid A → shared
face centroid → centroid B; its two halves use their respective cell tensors.
This stays inside the two tetrahedra even when their union is nonconvex. Invalid
gradients exclude cells. Fault/interface vertices are never averaged together.

The resulting embedding stores `grid_shape=(number_of_tetrahedra,)`; its
`node_indices` map active nodes back to the original cell order. The same spectral
simulator and exact conditioning code are reused. `GaussianRealizations.to_cells()`
returns a full-cell vector with NaNs outside the active domain. Observations in
this example are deliberately **at cell centroids**, not wells snapped to them;
the existing voxel-only `embedding.query` rejects tetrahedral embeddings.
Centroid samples are not cell-volume averages.

### Porosity and verification

The example applies `porosity = expit(latent)` with prior latent mean `logit(0.20)`
and standard deviation 0.35. Thus 0.20 is the prior **median**, not its arithmetic
mean. Exact porosity observations 0.14, 0.27 and 0.23 are converted with `logit`
before conditioning. Posterior medians and 90% intervals are obtained by
transforming the corresponding Gaussian quantiles; the fourth panel shows the
interval width. This bounded transform is example-specific.

The default CPU run (`--cells 800`, 16 realizations, 512 frequencies, 32 landmarks,
3 dimensions) produced 8,854 tetrahedra and 3,065 active reservoir cells. The
half-step gradient comparison changed gradients by at most `2.9e-10` relative;
exact porosity residuals were at most `1.4e-16`. All 1,559 cells in the unobserved
fault block retained their seed-matched latent prior exactly. Tests also verify
that meshing/evaluation never invoke another solve or mutate the model grid,
physical-coordinate derivative transforms, face-path costs, unit/fault isolation,
embedding cache round trips and tetrahedral VTU export.

**Geometry remains provisional:** the default neighbor overlap was 0.809, local
distance relative P90 0.340 and normalized covariance error P90 0.139, failing the
existing targets. These compare Isomap to the cell graph, not to continuum
geodesics. The low-degree face graph has additional directional/path bias; a
better mesh-native distance approximation and refinement study remain necessary
before making quantitative continuity claims. The plot and report retain this
status even though conditioning and fault isolation work correctly.

### Conditional facies in the combination model

`examples/tetrahedral_facies.py` uses GemPy's **COMBINATION** example: folded
older units, a younger unconformity, and a displacing fault. MMG3D improves the
conforming mesh by default; `mmgpy` is required unless `--no-remesh` is supplied.
The model generator downloads/caches its input data through GemPy.

```bash
python -m gempy_plugins.property_estimation.examples.tetrahedral_facies
python -m gempy_plugins.property_estimation.examples.tetrahedral_facies \
  --no-show --output combined_facies.png --report combined_facies.json \
  --mesh-output combined_facies.vtu
```

Four linked panels show:

1. All structural units and the conforming tetrahedral mesh.
2. Prior shale/siltstone/sandstone facies in `rock2` and `rock1`.
3. A seed-matched realization conditioned on six synthetic centroid observations.
4. Conditional sandstone probability estimated from 128 realizations.

The red overlay is the **actual meshed fault**, rather than an illustrative
plane. Cut-face lines show structural contacts, including the unconformity.
Drag/scroll to navigate, `c` toggles the cutaway, `e` toggles cell edges, and `q`
closes the window. `--units rock2` restricts the simulated domain; `--cells`,
`--realizations`, `--length-scale`, and `--anisotropy-ratio` expose the principal
resolution/simulation settings. Units and fault blocks remain independent.

The prior facies proportions are 0.35/0.25/0.40, with a Gaussian length scale of
60 and bedding anisotropy 4:1. Three well-separated synthetic cell observations
are selected in one component of each active unit. Exact joint latent-interval
**rejection sampling** conditions the categories. No fixed latent midpoints,
post-hoc code replacement, or nearest-cell assignment of real boreholes is used.
Other components retain their seed-matched prior realizations. Proportions are
ensemble targets, not exact quotas, and posterior probabilities are Monte Carlo
estimates (maximum binomial standard error about 0.044 for 128 draws).

The VTU preserves full cell indexing, structural labels, prior/conditional facies,
observations, gradients/normals, and probabilities of **all three** facies, with
facies/unit name tables in field data. Inactive categorical cells have code -1;
inactive probabilities are NaN. `FaciesRealizations.to_cells()` provides this
cell-indexed categorical export. Reported ensemble facies fractions are weighted
by tetrahedron **volume**, so fine mesh regions do not count disproportionately.

The default remeshed run (`--cells 1500`, 48 landmarks, 3 dimensions, 512 frequencies)
produced 97,404 tetrahedra, of which 53,852 were active, in six connected components.
Every categorical observation was honored in all 128 draws; all 16,273 cells in
unobserved components were unchanged. The gradient half-step relative change was
at most `2.3e-7`, and the mesh contained no inverted tetrahedra.

The geometry gate still **does not pass**: neighbor overlap 0.703, local distance
relative P90 0.195, normalized covariance P90 0.111. These are graph-relative
embedding errors, in addition to the face graph's discretization error. The
conforming mesh resolves the geological boundaries, but it does not by itself
validate stochastic correlation distances. The probability panel displays the
geometry status.

### Continuous density in the combination model

`examples/tetrahedral_density.py` uses the same geometry, active units (`rock2`,
`rock1`), observation centroids and correlation settings as the combination-model
facies example. Both share the mesh/gradient/Isomap preparation in
`examples/tetrahedral_combination.py`. Density is modeled directly as a Gaussian
field in **g/cm³**, with mean **2.65** and standard deviation **0.10**; it is not
obtained by assigning constants to facies codes.

```bash
python -m gempy_plugins.property_estimation.examples.tetrahedral_density
python -m gempy_plugins.property_estimation.examples.tetrahedral_density \
  --no-show --output combined_density.png --report combined_density.json \
  --mesh-output combined_density.vtu
```

Each active unit receives three synthetic centroid measurements: exact densities
2.80 and 2.49, and density 2.72 with Gaussian measurement SD 0.03 g/cm³. Thus the
default has **four exact and two noisy observations**. The conditioner receives
measurement **variances** (`0`, `0`, `0.03²`) and uses the actual finite spectral
covariance. Noisy measurements retain posterior uncertainty and are not forced
to match every realization. Unobserved geological components retain their
seed-matched prior draws exactly.

The four linked views show the prior draw, conditioned draw, **analytical**
posterior mean, and **analytical** posterior standard deviation of the true
density field. The latter excludes measurement noise; it is not an ensemble
standard-deviation estimate. White markers denote exact data, gold markers noisy
data. The actual meshed fault and cut-face contacts provide structural context.
`c` toggles the cutaway, `e` the tetrahedral edges, and `q` closes the view.

Defaults retain 64 realizations, 512 frequencies, 48 landmarks, 3 embedding
dimensions, length scale 60, and bedding anisotropy 4:1. `--units`, `--cells`,
`--no-remesh`, `--realizations`, and the geometry/spectral options work as in the
facies example. The VTU contains both density draws, posterior mean/variance/SD,
observations/noise variances, structural labels and centroid gradients. Property
values are NaN outside the active domain; density units are stored in field data.
Density and its SD are in g/cm³, and variances in (g/cm³)².

Tests verify exact honoring, positive uncertainty at noisy observations, posterior
variance reduction, unchanged unobserved components, original-solve/grid reuse,
and cell-aligned VTU export. The same **provisional graph/Isomap geometry** limits
apply as in the facies example and are shown in the uncertainty panel. Density
values are point samples at cell centroids, not volume-averaged cell densities.

## Streaming, reproducibility and covariance consistency

`n_frequencies=F` produces **2F real sine/cosine columns**. The fixed basis uses
Gaussian frequencies; each realization draws Gaussian coefficients, giving exact
Gaussian marginals conditional on that basis and constant marginal variance.
The finite covariance is `variance/F * sum(cos(omega @ (q-q')))`. It approximates
the target Gaussian covariance. `simulator.finite_covariance(node_indices)` exposes
the actual model used for consistent conditioning.

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
geometry. Treat embeddings as immutable after construction. The simulator's global
covariance range, variance, and property mean do not change the embedding; local
continuity scales and anisotropy ratios do.

`LandmarkProjection.transform` accepts distances to its component's landmarks.
`geometry.query(graph, positions)` provides the conservative, domain-safe physical
coordinate interpolation described above.

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
Conditioning coverage checks dense posterior moments and ensemble covariance,
exact/noisy observations, duplicate/rank-limited constraints, chunk/cache parity,
off-grid queries, and zero influence across geological components.

## Following milestones

Continuous conditioning, conservative observation-coordinate queries, local range
maps, and categorical latent-interval conditioning are implemented. Categorical
sampling scalability and MCMC diagnostics remain further work.
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
