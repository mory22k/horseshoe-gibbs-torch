"""PyTorch implementation of Gibbs samplers for Bayesian linear regression with horseshoe prior.

The horseshoe prior is a sparsity-inducing prior useful for high-dimensional problems where
many coefficients are expected to be zero or near-zero. This package implements efficient
Gibbs sampling methods for Bayesian sparse regression with horseshoe priors using PyTorch.

Classes:
    HorseshoeGibbsSampler: Gibbs sampler for Bayesian linear regression with horseshoe prior
    FastMultivariateGaussianMixtureBCM: Sampler for Gaussian posterior using BCM algorithm
    FastMultivariateGaussianMixtureRue: Sampler for Gaussian posterior using Rue's algorithm

References:
    [1] C. M. Carvalho, N. G. Polson, and J. G. Scott, The horseshoe estimator for sparse signals, Biometrika 97, 465 (2010).
    [2] A. Bhattacharya, A. Chakraborty, and B. K. Mallick, Fast sampling with Gaussian scale-mixture priors in high-dimensional regression, Biometrika 103, 985 (2016).
    [3] H. Rue, Fast sampling of Gaussian Markov random fields, J. R. Stat. Soc. Series B Stat. Methodol. 63, 325 (2001).
    [4] E. Makalic and D. F. Schmidt, A simple sampler for the horseshoe estimator, IEEE Signal Process. Lett. 23, 179 (2016).
"""

from . import fast_mvg
from .horseshoe_ms import HorseshoeGibbsSampler

__all__ = [
    "HorseshoeGibbsSampler",
    "fast_mvg",
]
