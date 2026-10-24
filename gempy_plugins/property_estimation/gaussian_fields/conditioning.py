"""Component-wise Gaussian conditioning of the actual finite spectral model."""
from dataclasses import asdict, dataclass

import numpy as np

from ..geological_coordinates._utils import fingerprint, positive_integer
from .results import ConditionalGaussianRealizations, GaussianMoments


@dataclass
class _ConditioningBlock:
    indices: np.ndarray
    basis: object
    whitener: object
    projected_basis: object
    mean_coefficients: object
    noise_std: object


class ConditionalGaussianField:
    """Reusable simple-Gaussian posterior with known constant prior mean.

    Uses a Matheron update in coefficient space, independently per geological
    component. Solves use float64 and an eigenvalue-truncated pseudoinverse, with
    no hidden jitter. Inconsistent exact constraints are rejected, including
    conflicting duplicates and observations outside the finite basis's range.
    Storage depends on observations and features, not grid nodes times observations.
    The source simulator and embedding must remain immutable.
    """

    def __init__(self, simulator, observations, values, *, mean=0., noise_variance=0., rank_tolerance=1e-12):
        self.simulator = simulator
        self.embedding = simulator.embedding
        self.torch = torch = simulator.torch
        if not np.isscalar(mean) or not np.isfinite(mean):
            raise ValueError("mean must be a finite scalar")
        if not np.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1:
            raise ValueError("rank_tolerance must lie strictly between zero and one")
        self.mean = float(mean)
        basis, coordinates, components = simulator._observation_basis(observations)
        values = np.asarray(values, dtype=float)
        if values.shape != (len(coordinates),) or not np.isfinite(values).all():
            raise ValueError("values must contain one finite value per observation")
        noise = np.asarray(noise_variance, dtype=float)
        if noise.ndim == 0:
            noise = np.full(len(values), float(noise))
        if noise.shape != values.shape or not np.isfinite(noise).all() or np.any(noise < 0):
            raise ValueError("noise_variance must be finite, nonnegative, and scalar or one per observation")
        self.observation_values = values.copy()
        self.noise_variance = noise.copy()
        self._centered = torch.as_tensor(values - self.mean, dtype=torch.float64, device=simulator.device)
        self._blocks = {}
        self.diagnostics = []
        with torch.no_grad():
            for component in np.unique(components):
                indices = np.flatnonzero(components == component)
                index = torch.as_tensor(indices, device=simulator.device)
                phi = basis[index].to(torch.float64)
                noise_block = torch.as_tensor(noise[indices], dtype=torch.float64, device=simulator.device)
                covariance = phi @ phi.T + torch.diag(noise_block)
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                relative = max(float(rank_tolerance), len(indices) * torch.finfo(torch.float64).eps)
                retained = eigenvalues > eigenvalues[-1] * relative
                vectors, eigenvalues = eigenvectors[:, retained], eigenvalues[retained]
                centered = self._centered[index]
                unsupported = centered - vectors @ (vectors.T @ centered)
                tolerance = 1e-8 * max(np.sqrt(simulator.covariance.variance), float(torch.linalg.vector_norm(centered)))
                if float(torch.linalg.vector_norm(unsupported)) > tolerance:
                    raise ValueError("Observations are inconsistent with the retained finite-feature covariance rank; "
                                     "check conflicting exact values, increase frequencies, or revise noise/rank_tolerance")
                whitener = vectors / torch.sqrt(eigenvalues)
                projected = whitener.T @ phi
                mean_coefficients = projected.T @ (whitener.T @ centered)
                self._blocks[int(component)] = _ConditioningBlock(
                    indices, phi, whitener, projected, mean_coefficients, torch.sqrt(noise_block))
                self.diagnostics.append(dict(component=int(component), observations=len(indices),
                                             exact_observations=int(np.sum(noise[indices] == 0)),
                                             retained_rank=int(retained.sum()),
                                             retained_condition_number=float(eigenvalues[-1] / eigenvalues[0]) if len(eigenvalues) else None))
        self._metadata = dict(method='finite_feature_gaussian_conditioning', mean=self.mean,
                              covariance=asdict(simulator.covariance), basis_fingerprint=simulator.basis_fingerprint,
                              observations=len(values), components=self.diagnostics,
                              conditioning_fingerprint=fingerprint(
                                  dict(basis=simulator.basis_fingerprint, mean=self.mean, rank_tolerance=rank_tolerance),
                                  coordinates, components, values, noise))

    def _coefficients(self, seed, component, first, stop):
        simulator, torch = self.simulator, self.torch
        coefficients = simulator._draw_coefficients(seed, component, first, stop)
        block = self._blocks.get(component)
        if block is None:
            return coefficients
        # A separate RNG stream is essential: observed noise is independent of
        # the unconditional coefficients, while prior draws remain seed-matched.
        noise = np.stack([
            np.random.default_rng(np.random.SeedSequence([seed, component, realization, 1])).normal(size=len(block.indices))
            for realization in range(first, stop)
        ])
        noise = torch.as_tensor(noise, dtype=torch.float64, device=simulator.device) * block.noise_std
        coefficients = coefficients.to(torch.float64)
        index = torch.as_tensor(block.indices, device=simulator.device)
        residual = self._centered[index] - coefficients @ block.basis.T - noise
        coefficients += (residual @ block.whitener) @ block.projected_basis
        return coefficients.to(simulator.dtype)

    def iter_simulate(self, *, realizations=1, seed=0, point_chunk_size=8192,
                      realization_batch_size=16, output_device='cpu'):
        """Yield grid and observation draws; mean/noise are fixed at preparation."""
        realizations = positive_integer(realizations, 'realizations')
        point_chunk_size = positive_integer(point_chunk_size, 'point_chunk_size')
        realization_batch_size = positive_integer(realization_batch_size, 'realization_batch_size')
        np.random.SeedSequence(seed)
        simulator, torch = self.simulator, self.torch
        components = [(int(component), np.flatnonzero(simulator._component_ids == component))
                      for component in np.unique(simulator._component_ids)]
        for first in range(0, realizations, realization_batch_size):
            stop = min(first + realization_batch_size, realizations)
            values = torch.empty((stop - first, len(simulator._coordinates)), dtype=simulator.dtype, device=output_device)
            observed = torch.empty((stop - first, len(self.observation_values)), dtype=simulator.dtype, device=output_device)
            with torch.no_grad():
                for component, nodes in components:
                    coefficients = self._coefficients(seed, component, first, stop)
                    for start in range(0, len(nodes), point_chunk_size):
                        indices = nodes[start:start + point_chunk_size]
                        chunk = coefficients @ simulator._basis(indices).T + self.mean
                        index = torch.as_tensor(indices, device=values.device)
                        values[:, index] = chunk.to(values.device)
                    if component in self._blocks:
                        block = self._blocks[component]
                        chunk = coefficients @ block.basis.to(simulator.dtype).T + self.mean
                        index = torch.as_tensor(block.indices, device=observed.device)
                        observed[:, index] = chunk.to(observed.device)
            yield ConditionalGaussianRealizations(values, self.embedding, first,
                                                   dict(self._metadata, seed=int(seed)), observed)

    def simulate(self, *, realizations=1, output_device='cpu', **kwargs):
        """Materialize conditional draws, retaining values at observation locations."""
        realizations = positive_integer(realizations, 'realizations')
        values = self.torch.empty((realizations, len(self.simulator._coordinates)),
                                  dtype=self.simulator.dtype, device=output_device)
        observed = self.torch.empty((realizations, len(self.observation_values)),
                                    dtype=self.simulator.dtype, device=output_device)
        metadata = {}
        for batch in self.iter_simulate(realizations=realizations, output_device=output_device, **kwargs):
            start = batch.realization_start
            values[start:start + len(batch.values)] = batch.values
            observed[start:start + len(batch.values)] = batch.observation_values
            metadata = batch.metadata
        return ConditionalGaussianRealizations(values, self.embedding, metadata=metadata, observation_values=observed)

    def moments(self, *, point_chunk_size=8192, output_device='cpu'):
        """Analytical posterior mean/variance of the latent field, without sampling.

        Variance excludes measurement error. Positive-noise observations retain
        uncertainty; exact observations have zero variance to numerical precision.
        """
        point_chunk_size = positive_integer(point_chunk_size, 'point_chunk_size')
        simulator, torch = self.simulator, self.torch
        mean = torch.empty(len(simulator._coordinates), dtype=simulator.dtype, device=output_device)
        variance = torch.empty_like(mean)
        with torch.no_grad():
            for component in np.unique(simulator._component_ids):
                nodes = np.flatnonzero(simulator._component_ids == component)
                block = self._blocks.get(int(component))
                for start in range(0, len(nodes), point_chunk_size):
                    indices = nodes[start:start + point_chunk_size]
                    basis = simulator._basis(indices).to(torch.float64)
                    mu = torch.full((len(indices),), self.mean, dtype=torch.float64, device=simulator.device)
                    var = (basis ** 2).sum(dim=1)
                    if block is not None:
                        mu += basis @ block.mean_coefficients
                        var -= ((basis @ block.projected_basis.T) ** 2).sum(dim=1)
                    index = torch.as_tensor(indices, device=mean.device)
                    mean[index] = mu.to(device=mean.device, dtype=mean.dtype)
                    variance[index] = var.clamp_min(0).to(device=variance.device, dtype=variance.dtype)
        return GaussianMoments(mean, variance, self.embedding, dict(self._metadata))
