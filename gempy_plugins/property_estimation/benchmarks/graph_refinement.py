"""Compare coarse and refined geological graphs at identical physical locations.

An odd refinement factor preserves voxel-center locations. No nearest-voxel
snapping is used, so discretization effects are not mixed with endpoint shifts.
"""
import argparse
import json
from time import perf_counter

from ..examples.geological_models import make_geological_model
from ..geological_coordinates import GeologicalMetricField, AnisotropicGraph
from ..geological_coordinates._utils import positive_integer
from ..geological_coordinates.diagnostics import graph_refinement_quality


def run_benchmark(*, geometry='folded', resolution=(24, 10, 24), refinement=3, anisotropy_ratio=5.,
                  stencil_radius=3, reference_radius=5, length_scale=5., samples=8):
    refinement = positive_integer(refinement, 'refinement')
    if refinement % 2 != 1:
        raise ValueError("refinement must be odd to preserve voxel centers")
    start = perf_counter()
    model = make_geological_model(geometry, resolution=resolution)
    metric = GeologicalMetricField.from_gempy(model, 'lower', anisotropy_ratio=anisotropy_ratio)
    graph = AnisotropicGraph.from_metric(metric, stencil_radius=stencil_radius)
    reference_model = make_geological_model(geometry, resolution=tuple(size * refinement for size in resolution))
    reference_metric = GeologicalMetricField.from_gempy(reference_model, 'lower', anisotropy_ratio=anisotropy_ratio)
    reference = AnisotropicGraph.from_metric(reference_metric, stencil_radius=reference_radius)
    quality = graph_refinement_quality(graph, reference, length_scale=length_scale, samples=samples)
    return dict(geometry=geometry, resolution=list(resolution), anisotropy_ratio=anisotropy_ratio,
                stencil_radius=stencil_radius, active_nodes=len(graph.node_indices),
                reference_active_nodes=len(reference.node_indices),
                reference_graph_csr_bytes=sum(a.nbytes for a in (reference.adjacency.data, reference.adjacency.indices, reference.adjacency.indptr)),
                elapsed_seconds=perf_counter() - start, quality=quality)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='folded')
    parser.add_argument('--resolution', default='24,10,24')
    parser.add_argument('--refinement', type=int, default=3)
    parser.add_argument('--ratio', type=float, default=5.)
    parser.add_argument('--stencil-radius', type=int, default=3)
    parser.add_argument('--reference-radius', type=int, default=5)
    parser.add_argument('--length-scale', type=float, default=5.)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--output')
    args = parser.parse_args()
    report = run_benchmark(geometry=args.model, resolution=tuple(map(int, args.resolution.split(','))),
                           refinement=args.refinement, anisotropy_ratio=args.ratio,
                           stencil_radius=args.stencil_radius, reference_radius=args.reference_radius,
                           length_scale=args.length_scale, samples=args.samples)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
