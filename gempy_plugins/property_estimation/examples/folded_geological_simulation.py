"""Offline M1–M3 demonstration: Gaussian density inside a GemPy anticline.

Run with ``python -m gempy_plugins.property_estimation.examples.folded_geological_simulation``.
Use ``--no-show --output section.png`` for a headless section image, or
``--volume`` to also inspect the realization with the existing PyVista viewer.
"""
import argparse
import json
from time import perf_counter

import numpy as np

from .geological_models import make_geological_model
from ..geological_coordinates import GeologicalMetricField, AnisotropicGraph, GeologicalEmbedding
from ..geological_coordinates.diagnostics import embedding_quality, directional_variograms
from ..gaussian_fields import GaussianCovariance, GaussianFieldSimulator


def run_demo(device='cpu'):
    start = perf_counter()
    model = make_geological_model('folded')
    metric = GeologicalMetricField.from_gempy(model, 'lower', anisotropy_ratio=5)
    graph = AnisotropicGraph.from_metric(metric, stencil_radius=3)
    embedding = GeologicalEmbedding.from_graph(graph, landmarks=32, dimensions=6)
    quality = embedding_quality(graph, embedding, length_scale=5)
    simulator = GaussianFieldSimulator(embedding, GaussianCovariance(length_scale=5, variance=.01),
                                       n_frequencies=512, basis_seed=0, device=device)
    fields = simulator.simulate_continuous(realizations=8, seed=13, mean=2.65)
    report = dict(quality=quality, active_nodes=len(embedding.coordinates),
                  directional_variograms=directional_variograms(graph, fields.values.cpu().numpy(), max_separation=5),
                  elapsed_seconds=perf_counter() - start, device=device)
    return model, embedding, fields, report


def plot_sections(model, fields):
    import matplotlib.pyplot as plt

    grid = model.grid.dense_grid
    shape = tuple(grid.resolution)
    y = shape[1] // 2
    scalars = model.solutions.octrees_output[0].outputs[0].exported_fields_dense_grid.scalar_field.reshape(shape)
    levels = np.sort(model.solutions.octrees_output[0].outputs[0].scalar_field_at_sp)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for realization, axis in enumerate(axes):
        image = axis.imshow(fields.to_grid(realization)[:, y, :].T, origin='lower', extent=[0, 100, 0, 100],
                            cmap='viridis', vmin=2.35, vmax=2.95, interpolation='nearest')
        axis.contour(grid.x_coord, grid.z_coord, scalars[:, y, :].T, levels=levels, colors='black', linewidths=1)
        axis.set(title=f'Density realization {realization + 1}', xlabel='X', ylabel='Z', aspect='equal')
    figure.colorbar(image, ax=axes, label='Density (example units)')
    figure.suptitle('GemPy anticline · shared 6D embedding · 5:1 bedding anisotropy')
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output', help='Optional section image path')
    parser.add_argument('--volume', action='store_true')
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model, embedding, fields, report = run_demo(args.device)
    print(json.dumps(report, indent=2))
    figure = plot_sections(model, fields)
    if args.output:
        figure.savefig(args.output, dpi=160)
    if not args.no_show:
        plt.show()
    plt.close(figure)
    if args.volume:
        from ..plotting import plot_property_field
        plotter = plot_property_field(model, fields.to_property_field(), show=not args.no_show)
        plotter.close()


if __name__ == '__main__':
    main()
