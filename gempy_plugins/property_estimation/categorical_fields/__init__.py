"""Unconditional categorical properties from Gaussian latent fields."""
from .truncated import TruncatedGaussianFacies
from .results import FaciesRealizations
from .diagnostics import facies_diagnostics

__all__ = ['TruncatedGaussianFacies', 'FaciesRealizations', 'facies_diagnostics']
