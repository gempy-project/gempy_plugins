"""Compare graph stencils, landmark embeddings, and optional spectral ensembles.

Defaults deliberately include both the failing 20:1/26-neighbor baseline and
the passing 5:1/wider-stencil folded example. JSON output records all outcomes.
"""
import argparse
from itertools import product
import json
import platform
import resource
from time import perf_counter

import numpy as np
import scipy

from ..examples.geological_models import make_geological_model
from ..geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding
from ..geological_coordinates.diagnostics import embedding_quality, constant_metric_quality
from ..geological_coordinates.analytical import bedding_orientation


def run_benchmark(*, models=('horizontal', 'dipping', 'folded'), resolution=(24, 10, 24),
                  ratios=(5., 20.), stencil_radii=(1, 3), landmarks=(32,), dimensions=(6,),
                  features=(512,), realizations=8, simulate=False, device='cpu', length_scale=5.,
                  distance_dtype='float32', seeds=(0,), resolutions=None, length_scales=None,
                  facies=False):
    reports = []
    resolutions = (resolution,) if resolutions is None else resolutions
    length_scales = (length_scale,) if length_scales is None else length_scales
    for geometry, resolution in product(models, resolutions):
        start = perf_counter()
        model = make_geological_model(geometry, resolution=resolution)
        model_seconds = perf_counter() - start
        for ratio, radius in product(ratios, stencil_radii):
            start = perf_counter()
            metric = GeologicalMetricField.from_gempy(model, 'lower', anisotropy_ratio=ratio)
            graph = AnisotropicGraph.from_metric(metric, stencil_radius=radius)
            graph_seconds = perf_counter() - start
            orientation = bedding_orientation(metric)
            is_constant = ratio == 1 or orientation['angular_deviation_max_degrees'] < np.degrees(1e-6)
            analytical_spectral = {}
            for count, width, seed in product(landmarks, dimensions, seeds):
                start = perf_counter()
                embedding = GeologicalEmbedding.from_graph(graph, landmarks=count, dimensions=width,
                                                            seed=seed, distance_dtype=distance_dtype)
                embedding_seconds = perf_counter() - start
                for scale in length_scales:
                    report = dict(geometry=geometry, resolution=list(resolution), anisotropy_ratio=ratio,
                                  stencil_radius=radius, landmarks=count, dimensions=width, landmark_seed=seed,
                                  length_scale=scale, components=len(embedding.projections),
                                  active_nodes=len(embedding.coordinates), directed_edges=graph.adjacency.nnz,
                                  distance_dtype=distance_dtype, geometry_seconds=model_seconds,
                                  graph_seconds=graph_seconds, embedding_seconds=embedding_seconds,
                                  graph_csr_bytes=sum(a.nbytes for a in (graph.adjacency.data, graph.adjacency.indices, graph.adjacency.indptr)),
                                  coordinate_bytes=embedding.coordinates.nbytes,
                                  quality=embedding_quality(graph, embedding, length_scale=scale))
                    if is_constant:
                        report['constant_metric_reference'] = constant_metric_quality(graph, embedding, length_scale=scale)
                    if simulate:
                        report['spectral'] = [_spectral_benchmark(embedding, scale, n, realizations, device) for n in features]
                        if is_constant:
                            if scale not in analytical_spectral:
                                analytical = GeologicalEmbedding.from_constant_metric(graph)
                                analytical_spectral[scale] = [_spectral_benchmark(analytical, scale, n, realizations, device) for n in features]
                            report['analytical_spectral'] = analytical_spectral[scale]
                    if facies:
                        report['facies'] = [_facies_benchmark(graph, embedding, scale, n, realizations, device) for n in features]
                    reports.append(report)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                 platform=platform.platform(), device=device),
                process_peak_rss_bytes=int(peak * (1 if platform.system() == 'Darwin' else 1024)),
                cases=reports)


def _spectral_benchmark(embedding, length_scale, frequencies, realizations, device):
    from ..gaussian_fields import GaussianCovariance, GaussianFieldSimulator
    import torch
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    simulator = GaussianFieldSimulator(embedding, GaussianCovariance(length_scale), n_frequencies=frequencies, device=device)
    if cuda:
        torch.cuda.synchronize(device)
    setup_seconds = perf_counter() - start
    for batch in simulator.iter_simulate(realizations=1, seed=1):
        del batch
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for repeat in range(3):
        start = perf_counter()
        for batch in simulator.iter_simulate(realizations=realizations, seed=repeat):
            del batch
        if cuda:
            torch.cuda.synchronize(device)
        timings.append(perf_counter() - start)
    seconds = float(np.median(timings))
    nodes = np.random.default_rng(1).choice(len(embedding.coordinates), min(64, len(embedding.coordinates)), replace=False)
    delta = embedding.coordinates[nodes, None, :] - embedding.coordinates[None, nodes, :]
    target = simulator.covariance(np.linalg.norm(delta, axis=-1))
    target *= embedding.component_ids[nodes, None] == embedding.component_ids[None, nodes]
    error = np.abs(simulator.finite_covariance(nodes).cpu().numpy() - target)
    return dict(frequencies=frequencies, real_basis_columns=2 * frequencies, realizations=realizations,
                setup_seconds=setup_seconds, ensemble_seconds_median=seconds,
                ensemble_seconds_repeats=timings, seconds_per_realization=seconds / realizations,
                uncached_basis_bytes=simulator.basis_cache_bytes, covariance_absolute_p90=float(np.percentile(error, 90)),
                peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if cuda else None,
                torch=torch.__version__)


def _facies_benchmark(graph, embedding, length_scale, frequencies, realizations, device):
    from ..gaussian_fields import GaussianCovariance, GaussianFieldSimulator
    from ..categorical_fields import TruncatedGaussianFacies, facies_diagnostics
    simulator = GaussianFieldSimulator(embedding, GaussianCovariance(length_scale),
                                       n_frequencies=frequencies, device=device)
    fields = TruncatedGaussianFacies(simulator, (.3, .4, .3), names=('shale', 'limestone', 'sandstone')).simulate(
        realizations=realizations, seed=13)
    return dict(frequencies=frequencies, **facies_diagnostics(graph, fields))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default='horizontal,dipping,folded')
    parser.add_argument('--resolution', default='24,10,24')
    parser.add_argument('--resolutions', nargs='+', help='Multiple grids, e.g. 24,10,24 48,20,48')
    parser.add_argument('--ratios', default='5,20')
    parser.add_argument('--stencil-radii', default='1,3')
    parser.add_argument('--landmarks', default='32')
    parser.add_argument('--dimensions', default='6')
    parser.add_argument('--seeds', default='0', help='Comma-separated landmark seeds')
    parser.add_argument('--features', default='512')
    parser.add_argument('--realizations', type=int, default=8)
    parser.add_argument('--length-scale', type=float, default=5.)
    parser.add_argument('--length-scales', help='Comma-separated covariance scales')
    parser.add_argument('--distance-dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--facies', action='store_true', help='Also report truncated-Gaussian facies statistics')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', help='Optional machine-readable JSON result path')
    parser.add_argument('--summary', action='store_true', help='Print a compact table instead of full JSON')
    args = parser.parse_args()
    report = run_benchmark(
        models=args.models.split(','), resolution=tuple(map(int, args.resolution.split(','))),
        ratios=tuple(map(float, args.ratios.split(','))), stencil_radii=tuple(map(int, args.stencil_radii.split(','))),
        landmarks=tuple(map(int, args.landmarks.split(','))), dimensions=tuple(map(int, args.dimensions.split(','))),
        features=tuple(map(int, args.features.split(','))), realizations=args.realizations,
        length_scale=args.length_scale, distance_dtype=args.distance_dtype, simulate=args.simulate, device=args.device,
        seeds=tuple(map(int, args.seeds.split(','))),
        resolutions=[tuple(map(int, value.split(','))) for value in args.resolutions] if args.resolutions else None,
        length_scales=tuple(map(float, args.length_scales.split(','))) if args.length_scales else None,
        facies=args.facies,
    )
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(text + '\n')
    if args.summary:
        print('geometry grid ratio radius dims seed scale overlap distance_p90 covariance_p90 gate analytical_graph_cov analytical_embedding_cov')
        for case in report['cases']:
            quality = case['quality']
            reference = case.get('constant_metric_reference', {})
            print(case['geometry'], 'x'.join(map(str, case['resolution'])), case['anisotropy_ratio'],
                  case['stencil_radius'], case['dimensions'], case['landmark_seed'], case['length_scale'],
                  *[f'{quality[key]:.4f}' if quality[key] is not None else '-' for key in
                    ('neighbor_overlap', 'distance_relative_p90', 'covariance_absolute_p90')],
                  'pass' if quality['passes_provisional_targets'] else 'fail',
                  *[f"{reference[key]['covariance_absolute_p90']:.4f}" if reference.get(key, {}).get('covariance_absolute_p90') is not None else '-'
                    for key in ('graph_vs_analytical', 'embedding_vs_analytical')])
    else:
        print(text)


if __name__ == '__main__':
    main()
