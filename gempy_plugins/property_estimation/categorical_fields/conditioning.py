"""Interval-conditioned Gaussian latent fields for ordered categorical observations."""
import numpy as np
from scipy.linalg import cho_solve
from scipy.special import log_ndtr, ndtr, ndtri, ndtri_exp

from ..gaussian_fields.conditioning import ConditionalGaussianField
from ..geological_coordinates import GeologicalPoints
from ..geological_coordinates._utils import fingerprint, positive_integer
from .results import ConditionalFaciesRealizations


def _truncated_normal(uniforms, lower, upper, mean, std):
    """Inverse-CDF normal draws with log-tail arithmetic for extreme intervals."""
    u = np.clip(uniforms, np.nextafter(0., 1.), np.nextafter(1., 0.))
    u, a, b, mean, std = np.broadcast_arrays(u, (lower - mean) / std, (upper - mean) / std, mean, std)
    result = np.empty_like(u)
    right, left = a >= 0, b <= 0
    middle = ~(right | left)
    result[right] = -ndtri_exp(np.logaddexp(np.log1p(-u[right]) + log_ndtr(-a[right]),
                                          np.log(u[right]) + log_ndtr(-b[right])))
    result[left] = ndtri_exp(np.logaddexp(np.log1p(-u[left]) + log_ndtr(a[left]),
                                        np.log(u[left]) + log_ndtr(b[left])))
    result[middle] = ndtri(ndtr(a[middle]) + u[middle] * (ndtr(b[middle]) - ndtr(a[middle])))
    return mean + std * result


class _IntervalGaussianField(ConditionalGaussianField):
    """Reuse continuous coefficient-space conditioning with a random latent target."""

    def __init__(self, simulator, observations, lower, upper, *, sampler, max_proposals, burn_in, rank_tolerance):
        super().__init__(simulator, observations, np.zeros(len(lower)), rank_tolerance=rank_tolerance)
        self.sampler, self.max_proposals, self.burn_in = sampler, max_proposals, burn_in
        self._intervals = {}
        for component, block in self._blocks.items():
            phi = block.basis.cpu().numpy()
            covariance = phi @ phi.T
            try:
                factor = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError:
                if sampler == 'gibbs':
                    raise ValueError("Gibbs sampling requires numerically positive-definite, full-rank observation covariance") from None
                factor = phi  # Exact rejection also supports singular finite bases.
            if block.whitener.shape[1] != len(block.indices):
                factor = phi
            precision = None
            if sampler == 'gibbs':
                if block.whitener.shape[1] != len(block.indices):
                    raise ValueError("Gibbs sampling requires full-rank observation covariance; "
                                     "increase frequencies, remove redundant observations, or use rejection")
                precision = cho_solve((factor, True), np.eye(len(block.indices)))
            self._intervals[component] = (lower[block.indices], upper[block.indices], factor, precision,
                                           np.sqrt(np.diag(covariance)))

    def _sample_targets(self, seed, component, first, stop):
        lower, upper, factor, precision, std = self._intervals[component]
        count = len(lower)
        generators = [np.random.default_rng(np.random.SeedSequence([seed, component, realization, 2]))
                      for realization in range(first, stop)]
        if self.sampler == 'rejection':
            targets = np.empty((stop - first, count))
            for row, rng in enumerate(generators):
                attempted = 0
                while attempted < self.max_proposals:
                    size = min(128, self.max_proposals - attempted)
                    candidates = rng.normal(size=(size, factor.shape[1])) @ factor.T
                    accepted = np.flatnonzero(np.all((candidates >= lower) & (candidates < upper), axis=1))
                    if len(accepted):
                        targets[row] = candidates[accepted[0]]
                        break
                    attempted += size
                else:
                    raise RuntimeError("Facies rejection sampling exhausted max_proposals; constraints may be rare "
                                       "or inconsistent. Increase the budget or explicitly choose sampler='gibbs'.")
            return targets
        # Independent chains make random draws invariant to batch size. Finite
        # burn-in still introduces potential bias and is reported in metadata.
        uniforms = np.stack([rng.random((self.burn_in + 1, count)) for rng in generators])
        targets = _truncated_normal(uniforms[:, 0], lower, upper, np.zeros((stop - first, count)), std)
        conditional_std = 1 / np.sqrt(precision.diagonal())
        for sweep in range(self.burn_in):
            for index in range(count):
                mean = targets[:, index] - (targets @ precision[:, index]) / precision[index, index]
                targets[:, index] = _truncated_normal(uniforms[:, sweep + 1, index], lower[index], upper[index],
                                                      mean, conditional_std[index])
        return targets

    def _coefficients(self, seed, component, first, stop):
        simulator, torch = self.simulator, self.torch
        coefficients = simulator._draw_coefficients(seed, component, first, stop)
        block = self._blocks.get(component)
        if block is None:
            return coefficients
        targets = torch.as_tensor(self._sample_targets(seed, component, first, stop),
                                  dtype=torch.float64, device=simulator.device)
        coefficients = coefficients.to(torch.float64)
        residual = targets - coefficients @ block.basis.T
        return (coefficients + (residual @ block.whitener) @ block.projected_basis).to(simulator.dtype)


class ConditionalTruncatedGaussianFacies:
    """Exact categorical observations as joint latent Gaussian interval constraints.

    Rejection is the default reference sampler for modest observation sets. It
    returns independent conditional draws, or fails explicitly at its budget.
    Gibbs is optional and approximate at finite burn-in: one independent chain
    per realization, with full-rank observation covariance required. Neither
    method substitutes fixed interval midpoints or enforces global facies quotas.
    """

    def __init__(self, facies, observations, codes, *, sampler='rejection', max_proposals=100000,
                 burn_in=256, rank_tolerance=1e-12):
        if sampler not in ('rejection', 'gibbs'):
            raise ValueError("sampler must be 'rejection' or 'gibbs'")
        max_proposals = positive_integer(max_proposals, 'max_proposals')
        burn_in = positive_integer(burn_in, 'burn_in')
        self.facies, self.simulator = facies, facies.simulator
        _, coordinates, components = self.simulator._observation_basis(observations)
        codes = np.asarray(codes)
        if (codes.shape != (len(coordinates),) or not np.issubdtype(codes.dtype, np.integer)
                or np.any(codes < 0) or np.any(codes >= len(facies.names))):
            raise ValueError("codes must contain one integer category code per observation")
        # Merge coincident observations in the simulator's actual precision.
        coordinates = coordinates.astype(self.simulator._coordinates.dtype)
        unique, inverse, lookup = [], [], {}
        for index, (q, component) in enumerate(zip(coordinates, components)):
            key = (int(component), *q.tolist())
            if key in lookup:
                match = lookup[key]
                if codes[unique[match]] != codes[index]:
                    raise ValueError("Conflicting facies codes at the same embedded location and component")
            else:
                match = len(unique)
                lookup[key] = match
                unique.append(index)
            inverse.append(match)
        self._inverse = np.asarray(inverse)
        self.observation_codes = codes.copy()
        points = GeologicalPoints(coordinates[unique], components[unique], self.simulator.embedding.fingerprint)
        thresholds = facies.thresholds * np.sqrt(self.simulator.covariance.variance)
        bounds = np.r_[-np.inf, thresholds, np.inf]
        self._latent = _IntervalGaussianField(
            self.simulator, points, bounds[codes[unique]], bounds[codes[unique] + 1],
            sampler=sampler, max_proposals=max_proposals, burn_in=burn_in, rank_tolerance=rank_tolerance)
        self.diagnostics = self._latent.diagnostics
        self._metadata = dict(method='conditional_truncated_gaussian', sampler=sampler,
                              burn_in=burn_in if sampler == 'gibbs' else None,
                              independent_chains=sampler == 'gibbs', max_proposals=max_proposals,
                              original_observations=len(codes), unique_observations=len(unique),
                              standardized_thresholds=facies.thresholds.tolist(),
                              conditioning_fingerprint=fingerprint(
                                  dict(basis=self.simulator.basis_fingerprint, proportions=facies.proportions),
                                  coordinates, components, codes))

    def iter_simulate(self, **kwargs):
        """Yield conditional codes, honoring observations at their actual locations."""
        torch = self.simulator.torch
        for latent in self._latent.iter_simulate(**kwargs):
            thresholds = torch.as_tensor(self.facies.thresholds * np.sqrt(self.simulator.covariance.variance),
                                          dtype=latent.values.dtype, device=latent.values.device)
            codes = torch.bucketize(latent.values.contiguous(), thresholds, right=True)
            inverse = torch.as_tensor(self._inverse, device=latent.values.device)
            observed = latent.observation_values[:, inverse]
            observed_codes = torch.bucketize(observed.contiguous(), thresholds, right=True)
            expected = torch.as_tensor(self.observation_codes, device=observed_codes.device)
            if not torch.all(observed_codes == expected):
                raise RuntimeError("Numerical conditioning error crossed a facies threshold; "
                                   "use float64 and review observation covariance rank")
            yield ConditionalFaciesRealizations(
                codes, latent.embedding, self.facies.names, self.facies.proportions, latent.realization_start,
                dict(latent.metadata, **self._metadata), observed_codes, observed)

    def simulate(self, *, realizations=1, output_device='cpu', **kwargs):
        """Materialize a conditional ensemble; probabilities() estimates its marginals."""
        realizations = positive_integer(realizations, 'realizations')
        torch = self.simulator.torch
        values = torch.empty((realizations, len(self.simulator.embedding.coordinates)), dtype=torch.long, device=output_device)
        observed_codes = torch.empty((realizations, len(self.observation_codes)), dtype=torch.long, device=output_device)
        observed_latent = torch.empty(observed_codes.shape, dtype=self.simulator.dtype, device=output_device)
        metadata = {}
        for batch in self.iter_simulate(realizations=realizations, output_device=output_device, **kwargs):
            start, stop = batch.realization_start, batch.realization_start + len(batch.values)
            values[start:stop] = batch.values
            observed_codes[start:stop] = batch.observation_codes
            observed_latent[start:stop] = batch.latent_observation_values
            metadata = batch.metadata
        return ConditionalFaciesRealizations(values, self.simulator.embedding, self.facies.names, self.facies.proportions,
                                              metadata=metadata, observation_codes=observed_codes,
                                              latent_observation_values=observed_latent)
