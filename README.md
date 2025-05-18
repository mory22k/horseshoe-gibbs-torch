# horseshoe-gibbs-torch

A PyTorch implementation of Gibbs samplers for Bayesian linear regression with horseshoe prior.

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![PyTorch Version](https://img.shields.io/badge/torch-2.7.0%2B-orange)](https://pytorch.org/)

## Overview

The horseshoe prior is a sparsity-inducing prior useful for high-dimensional linear regression problems where many coefficients are expected to be zero. This package implements efficient Gibbs sampling methods for Bayesian sparse regression with horseshoe priors using PyTorch.

## References

1. C. M. Carvalho, N. G. Polson, and J. G. Scott, "The horseshoe estimator for sparse signals," Biometrika **97**, 465 (2010).
2. A. Bhattacharya, A. Chakraborty, and B. K. Mallick, "Fast sampling with Gaussian scale-mixture priors in high-dimensional regression," Biometrika **103**, 985 (2016).
3. H. Rue, "Fast sampling of Gaussian Markov random fields," J. R. Stat. Soc. Series B Stat. Methodol. **63**, 325 (2001).
4. E. Makalic and D. F. Schmidt, "A simple sampler for the horseshoe estimator," IEEE Signal Process. Lett. **23**, 179 (2016).

## Quick Start

```python
import torch
from horseshoe_gibbs_torch import HorseshoeGibbsSampler

# Create synthetic sparse regression problem
n = 300  # Number of data samples
p = 200  # Number of features

X = torch.randn(n, p)
w = torch.randn(p) * torch.bernoulli(torch.ones(p) * 0.1)  # Sparse coefficients
y = X @ w

# Run the horseshoe sampler
w_sample = HorseshoeGibbsSampler(X, y).sample(num_mcs=100)

# Calculate RMSE
rmse = ((w_sample - w)**2).mean().sqrt()
```

## API Documentation

### HorseshoeGibbsSampler

```python
HorseshoeGibbsSampler(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    sigma2: Optional[torch.Tensor] = None,
    tau2: Optional[torch.Tensor] = None,
    lamb2: Optional[torch.Tensor] = None,
    a_sigma2: float = 1.0,
    b_sigma2: float = 1.0
)
```

Gibbs sampler for Bayesian linear regression with horseshoe prior.

### FastMultivariateGaussianMixtureBCM

```python
FastMultivariateGaussianMixtureBCM(
    X: torch.Tensor,
    y: torch.Tensor,
    D_diag: torch.Tensor,
    sigma2: torch.Tensor,
    device: Optional[torch.device] = None
)
```

Fast sampler for Gaussian posterior using the Bhattacharya-Chakraborty-Mallick algorithm. Fast when `p >> n`.

### FastMultivariateGaussianMixtureRue

```python
FastMultivariateGaussianMixtureRue(
    XtX: torch.Tensor,
    Xty: torch.Tensor,
    D_diag: torch.Tensor,
    sigma2: torch.Tensor,
    device: Optional[torch.device] = None
)
```

Fast sampler for Gaussian posterior using Rue's algorithm.

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/mory22k/horseshoe-gibbs-torch.git
cd horseshoe-gibbs-torch

# If you are using mise, trust and install the dependencies
mise trust
mise install

# Set up development environment
uv sync
```

### Development Tools

This project uses:
- `mise` for development environment management
- `task` for running common development tasks
- `uv` for python package management
- `ruff` for linting and formatting
- `mypy` for type checking

```bash
# Format code
task format

# Check code style
task check

# Format code style
task format

# Fix autofixable issues
task fix

# Prepare and commit
task commit
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
