"""Unconditional facies in offline GemPy folds, faults, and thin layers.

Run with --no-show --output facies.png --report facies.json for headless output.
Every model retains its geometry-quality result, including failures.
"""
import argparse
import json

import numpy as np

from .geological_models import make_geological_model
from ..geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding
from ..geological_coordinates.diagnostics import embedding_quality
from ..gaussian_fields import GaussianCovariance, GaussianFieldSimulator
from ..categorical_fields import TruncatedGaussianFacies, facies_diagnostics


def run_demo(models=('folded', 'syncline', 'faulted', 'thin_layer'), *, resolution=(24, 10, 24),
             realizations=64, dimensions=3, stencil_radius=3, device='cpu'):
    cases = []
    for name in models:
        model = make_geological_model(name, resolution=resolution)
        metric = GeologicalMetricField.from_gempy(model, 'lower', anisotropy_ratio=5)
        graph = AnisotropicGraph.from_metric(metric, stencil_radius=stencil_radius)
        embedding = GeologicalEmbedding.from_graph(graph, landmarks=32, dimensions=dimensions)
        simulator = GaussianFieldSimulator(embedding, GaussianCovariance(length_scale=5),
                                           n_frequencies=1024, device=device)
        fields = TruncatedGaussianFacies(simulator, (.3, .4, .3), names=('shale', 'limestone', 'sandstone')).simulate(
            realizations=realizations, seed=13)
        report = dict(geometry=name, active_nodes=len(embedding.coordinates), dimensions=dimensions,
                      stencil_radius=stencil_radius, quality=embedding_quality(graph, embedding, length_scale=5),
                      facies=facies_diagnostics(graph, fields))
        cases.append((model, fields, report))
    return cases


def plot_sections(cases):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    columns = min(2, min(len(fields.values) for _, fields, _ in cases))
    figure, axes = plt.subplots(len(cases), columns, figsize=(5 * columns, 3.8 * len(cases)),
                                squeeze=False, constrained_layout=True)
    cmap = ListedColormap(['#616b7b', '#86b6b0', '#edc66e'])
    norm = BoundaryNorm(np.arange(4) - .5, 3)
    for row, (model, fields, report) in enumerate(cases):
        grid = model.grid.dense_grid
        y = int(grid.resolution[1]) // 2
        extent = np.asarray(grid.extent)[[0, 1, 4, 5]]
        fault = np.asarray(model.solutions.raw_arrays.fault_block).reshape(tuple(grid.resolution))[:, y, :].T
        fault_levels = np.unique(fault)[:-1] + .5
        for realization, axis in enumerate(axes[row]):
            codes = fields.to_grid(realization)[:, y, :].T
            image = axis.imshow(np.ma.masked_less(codes, 0), origin='lower', extent=extent,
                                cmap=cmap, norm=norm, interpolation='nearest')
            axis.contour(grid.x_coord, grid.z_coord, (codes >= 0).astype(float), levels=[.5], colors='black', linewidths=.7)
            if len(fault_levels):
                axis.contour(grid.x_coord, grid.z_coord, fault, levels=fault_levels, colors='#c44343', linewidths=1.2)
            gate = 'pass' if report['quality']['passes_provisional_targets'] else 'FAIL'
            axis.set(title=f"{report['geometry']} · realization {realization + 1} · geometry {gate}",
                     xlabel='X', ylabel='Z', aspect='equal')
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), ticks=[0, 1, 2], shrink=.6)
    colorbar.ax.set_yticklabels(['Shale (30%)', 'Limestone (40%)', 'Sandstone (30%)'])
    figure.suptitle('Unconditional truncated-Gaussian facies · 5:1 bedding anisotropy')
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default='folded,syncline,faulted,thin_layer')
    parser.add_argument('--resolution', default='24,10,24')
    parser.add_argument('--realizations', type=int, default=64)
    parser.add_argument('--dimensions', type=int, default=3)
    parser.add_argument('--stencil-radius', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--report')
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = run_demo(models=args.models.split(','), resolution=tuple(map(int, args.resolution.split(','))),
                     realizations=args.realizations, dimensions=args.dimensions, stencil_radius=args.stencil_radius,
                     device=args.device)
    text = json.dumps([report for _, _, report in cases], indent=2, allow_nan=False)
    print(text)
    if args.report:
        with open(args.report, 'w') as stream:
            stream.write(text + '\n')
    figure = plot_sections(cases)
    if args.output:
        figure.savefig(args.output, dpi=160)
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == '__main__':
    main()
