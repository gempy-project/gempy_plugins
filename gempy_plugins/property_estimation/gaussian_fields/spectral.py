"""Chunked paired-feature Gaussian fields on arbitrary embedded coordinates."""
from dataclasses import asdict

import numpy as np

from gempy_plugins.optional_dependencies import require_torch
from ..geological_coordinates._utils import fingerprint, positive_integer
from .covariance import GaussianCovariance
from .results import GaussianRealizations


class GaussianFieldSimulator:
    """Spectral backend with fixed frequencies and independent component draws.

    ``n_frequencies=F`` means 2F real basis columns, not F columns. The default
    streams point chunks and writes results to CPU. ``iter_simulate`` also bounds
    output memory by yielding realization batches. Random draws are keyed by
    (seed, component, realization), independently of chunk sizes and device.

    Geometry and covariance are fixed for the lifetime of a simulator. Create a
    new instance after changing either. No engine backend/global RNG is changed.
    """

    def __init__(self, embedding, covariance=None, *, backend='spectral', device='cpu',
                 dtype='float32', n_frequencies=512, basis_seed=0):
        if backend != 'spectral':
            raise ValueError("The implemented backend is 'spectral'")
        self.torch = require_torch()
        self.embedding = embedding
        self.covariance = covariance or GaussianCovariance()
        if not isinstance(self.covariance, GaussianCovariance):
            raise TypeError("This backend currently supports GaussianCovariance")
        if dtype not in ('float32', 'float64'):
            raise ValueError("dtype must be 'float32' or 'float64'")
        self.dtype = getattr(self.torch, dtype)
        self.device = self.torch.device(device)
        self.n_frequencies = positive_integer(n_frequencies, 'n_frequencies')
        if embedding.coordinates.ndim != 2 or not np.isfinite(embedding.coordinates).all():
            raise ValueError("Embedding coordinates must be a finite (nodes, dimensions) array")
        self._coordinates = np.array(embedding.coordinates, dtype=dtype, copy=True)
        self._component_ids = np.array(embedding.component_ids, copy=True)
        if self._component_ids.shape != (len(self._coordinates),) or not np.issubdtype(self._component_ids.dtype, np.integer) or np.any(self._component_ids < 0):
            raise ValueError("Embedding requires one nonnegative integer component ID per node")
        frequencies = self.covariance.sample_frequencies(self.n_frequencies, self._coordinates.shape[1], basis_seed)
        self._frequencies = self.torch.tensor(frequencies, device=self.device, dtype=self.dtype)
        self.basis_fingerprint = fingerprint(dict(covariance=asdict(self.covariance)),
                                             self._coordinates, self._component_ids, frequencies)
        self._basis_cache = None

    def _basis(self, indices):
        if self._basis_cache is not None:
            index = self.torch.as_tensor(indices, dtype=self.torch.long, device=self.device)
            return self._basis_cache[index]
        q = self.torch.as_tensor(self._coordinates[indices], device=self.device, dtype=self.dtype)
        phase = q @ self._frequencies.T
        basis = self.torch.cat((self.torch.cos(phase), self.torch.sin(phase)), dim=1)
        return basis * np.sqrt(self.covariance.variance / self.n_frequencies)

    @property
    def basis_cache_bytes(self):
        return len(self._coordinates) * 2 * self.n_frequencies * self._frequencies.element_size()

    def cache_basis(self, *, max_bytes, point_chunk_size=8192):
        """Opt in to a full device-resident basis, under an explicit byte budget.

        The budget covers the persistent basis; evaluation also needs temporary
        storage for one point chunk. Frequencies/coordinates are fixed snapshots.
        """
        max_bytes = positive_integer(max_bytes, 'max_bytes')
        point_chunk_size = positive_integer(point_chunk_size, 'point_chunk_size')
        if self.basis_cache_bytes > max_bytes:
            raise ValueError(f"Basis needs {self.basis_cache_bytes} bytes, exceeding max_bytes={max_bytes}")
        if self._basis_cache is not None:
            return
        basis = self.torch.empty((len(self._coordinates), 2 * self.n_frequencies), dtype=self.dtype, device=self.device)
        with self.torch.no_grad():
            for start in range(0, len(basis), point_chunk_size):
                stop = min(start + point_chunk_size, len(basis))
                basis[start:stop] = self._basis(np.arange(start, stop))
        self._basis_cache = basis

    def clear_basis_cache(self):
        self._basis_cache = None

    def finite_covariance(self, left, right=None):
        """Actual covariance of this finite-feature simulator at node indices.

        Returns a device tensor. Cross-component covariance is exactly zero.
        This is the covariance to use for future conditioning; it is generally
        different from the infinite-feature target Gaussian covariance.
        """
        left = self._validate_indices(left)
        right = left if right is None else self._validate_indices(right)
        with self.torch.no_grad():
            covariance = self._basis(left) @ self._basis(right).T
            same = self._component_ids[left, None] == self._component_ids[None, right]
            return covariance * self.torch.as_tensor(same, device=self.device)

    def _validate_indices(self, indices):
        indices = np.atleast_1d(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= len(self._coordinates)):
            raise ValueError("Expected valid integer active-node indices")
        return indices

    def iter_simulate(self, *, realizations=1, seed=0, mean=0.0, point_chunk_size=8192,
                      realization_batch_size=16, output_device='cpu'):
        """Yield batches without keeping the entire ensemble or basis in memory."""
        realizations = positive_integer(realizations, 'realizations')
        point_chunk_size = positive_integer(point_chunk_size, 'point_chunk_size')
        realization_batch_size = positive_integer(realization_batch_size, 'realization_batch_size')
        if not np.isfinite(mean):
            raise ValueError("mean must be finite")
        # Validate the seed even if a malformed/empty embedding has no components.
        np.random.SeedSequence(seed)
        components = [(int(component), np.flatnonzero(self._component_ids == component))
                      for component in np.unique(self._component_ids)]
        for first in range(0, realizations, realization_batch_size):
            stop = min(first + realization_batch_size, realizations)
            values = self.torch.empty((stop - first, len(self._coordinates)), dtype=self.dtype, device=output_device)
            with self.torch.no_grad():
                for component, nodes in components:
                    coefficients = np.stack([
                        np.random.default_rng(np.random.SeedSequence([seed, component, realization])).normal(size=2 * self.n_frequencies)
                        for realization in range(first, stop)
                    ])
                    coefficients = self.torch.as_tensor(coefficients, dtype=self.dtype, device=self.device)
                    for start in range(0, len(nodes), point_chunk_size):
                        indices = nodes[start:start + point_chunk_size]
                        chunk = coefficients @ self._basis(indices).T + mean
                        output_indices = self.torch.as_tensor(indices, device=values.device, dtype=self.torch.long)
                        values[:, output_indices] = chunk.to(values.device)
            yield GaussianRealizations(values, self.embedding, first,
                                       dict(seed=int(seed), mean=float(mean), covariance=asdict(self.covariance),
                                            basis_fingerprint=self.basis_fingerprint))

    def simulate(self, *, realizations=1, output_device='cpu', **kwargs):
        """Materialize an ensemble; use iter_simulate for bounded output memory."""
        realizations = positive_integer(realizations, 'realizations')
        values = self.torch.empty((realizations, len(self._coordinates)), dtype=self.dtype, device=output_device)
        metadata = {}
        for batch in self.iter_simulate(realizations=realizations, output_device=output_device, **kwargs):
            start = batch.realization_start
            values[start:start + len(batch.values)] = batch.values
            metadata = batch.metadata
        return GaussianRealizations(values, self.embedding, metadata=metadata)

    def simulate_continuous(self, **kwargs):
        """Gaussian continuous properties with constant mean and covariance variance."""
        return self.simulate(**kwargs)
