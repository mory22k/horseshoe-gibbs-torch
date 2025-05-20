"""Horseshoe Gibbs Sampler for Bayesian linear regression with horseshoe prior."""

from logging import getLogger
from typing import Optional
from warnings import warn

import torch
from torch.distributions import InverseGamma
from tqdm import tqdm

from .base_sampler import BaseSampler
from .fast_mvg import (
    FastMultivariateGaussianMixtureBCM,
    FastMultivariateGaussianMixtureRue,
)

logger = getLogger(__name__)

EPSILON = 1e-15


def _sample_beta(
    X: torch.Tensor,
    y: torch.Tensor,
    XtX: torch.Tensor,
    XtY: torch.Tensor,
    sigma2: torch.Tensor,
    tau2: torch.Tensor,
    lamb2: torch.Tensor,
) -> torch.Tensor:
    n, p = X.shape
    if n < p:
        beta_new = FastMultivariateGaussianMixtureBCM(
            X,
            y,
            tau2 * lamb2,
            sigma2,
        ).sample()
    else:
        beta_new = FastMultivariateGaussianMixtureRue(
            XtX,
            XtY,
            tau2 * lamb2,
            sigma2,
        ).sample()
    return beta_new


def _sample_sigma2(
    X: torch.Tensor,
    y: torch.Tensor,
    tau2: torch.Tensor,
    lamb2: torch.Tensor,
    a_sigma: float,
    b_sigma: float,
) -> torch.Tensor:
    n = X.shape[0]

    concentration = a_sigma + n / 2

    XD = lamb2 * tau2 * X
    XDX_In = XD @ X.T + torch.eye(n)
    try:
        XDX_In_inv_y = torch.linalg.solve(XDX_In, y)
    except RuntimeError:
        logger.warning(
            "Solving linear system failed while sampling sigma2. Using lstsq instead."
        )
        XDX_In_inv_y = torch.linalg.lstsq(XDX_In, y, rcond=None)[0]
    rate = b_sigma + y @ XDX_In_inv_y / 2
    sigma2_new: torch.Tensor = InverseGamma(concentration, rate).sample()
    return sigma2_new


def _sample_lamb2(
    beta: torch.Tensor,
    sigma2: torch.Tensor,
    tau2: torch.Tensor,
    current_lamb2: torch.Tensor,
) -> torch.Tensor:
    concentration = 1.0
    try:
        rate = 1.0 + 1.0 / current_lamb2
    except ZeroDivisionError:
        rate = 1.0 + 1.0 / (current_lamb2 + EPSILON)
    nu: torch.Tensor = InverseGamma(concentration, rate).sample()

    concentration = 1.0
    rate = 1.0 / nu + beta**2 / sigma2 / tau2 / 2
    lamb2_new: torch.Tensor = InverseGamma(concentration, rate).sample()
    return lamb2_new


def _sample_tau2(
    beta: torch.Tensor,
    sigma2: torch.Tensor,
    current_tau2: torch.Tensor,
    lamb2: torch.Tensor,
) -> torch.Tensor:
    p = beta.shape[0]

    concentration = 1.0
    try:
        rate: float = 1.0 + 1.0 / current_tau2
    except ZeroDivisionError:
        rate = 1.0 + 1.0 / (current_tau2 + EPSILON)
    xi: torch.Tensor = InverseGamma(concentration, rate).sample()

    concentration = (p + 1.0) / 2
    rate = 1.0 / xi + torch.sum(beta**2 / lamb2) / sigma2 / 2
    tau2_new: torch.Tensor = InverseGamma(concentration, rate).sample()
    return tau2_new


def _markov_transition(
    X: torch.Tensor,
    y: torch.Tensor,
    XtX: torch.Tensor,
    XtY: torch.Tensor,
    beta: torch.Tensor,
    sigma2: torch.Tensor,
    tau2: torch.Tensor,
    lamb2: torch.Tensor,
    a_sigma: float,
    b_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    is_failed = False
    try:
        tau2_new = _sample_tau2(
            beta,
            sigma2,
            tau2,
            lamb2,
        )
    except ValueError:
        tau2_new = tau2
        logger.warning("Sampling tau2 failed. Using the current value instead.")

    try:
        sigma2_new = _sample_sigma2(
            X,
            y,
            tau2_new,
            lamb2,
            a_sigma=a_sigma,
            b_sigma=b_sigma,
        )
    except ValueError:
        sigma2_new = sigma2
        is_failed = True
        logger.warning("Sampling sigma2 failed. Using the current value instead.")

    beta_new = _sample_beta(
        X,
        y,
        XtX,
        XtY,
        sigma2_new,
        tau2_new,
        lamb2,
    )

    try:
        lamb2_new = _sample_lamb2(
            beta_new,
            sigma2_new,
            tau2_new,
            lamb2,
        )
    except ValueError:
        lamb2_new = lamb2
        is_failed = True
        logger.warning("Sampling lamb2 failed. Using the current value instead.")

    return beta_new, sigma2_new, tau2_new, lamb2_new, is_failed


class HorseshoeGibbsSampler(BaseSampler):
    """Horseshoe Gibbs Sampler for Bayesian linear regression with horseshoe prior."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        sigma2: Optional[torch.Tensor] = None,
        tau2: Optional[torch.Tensor] = None,
        lamb2: Optional[torch.Tensor] = None,
        a_sigma2: float = 1.0,
        b_sigma2: float = 1.0,
    ) -> None:
        """HorseshoeGibbsSampler(torch.Tensor, torch.Tensor) -> None.

        Args:
            train_X (torch.Tensor): Training data features of shape (batch_size, feature_dim).
            train_Y (torch.Tensor): Training data labels of shape (batch_size, feature_dim).
            weight (torch.Tensor): Initial values of the regression coefficients of shape (feature_dim,).
            sigma2 (torch.Tensor): Initial value of noise variance of shape (1,).
            tau2 (torch.Tensor): Initial value of global shrinkage parameter of shape (1,).
            lamb2 (torch.Tensor): Initial values of local shrinkage parameters of shape (feature_dim,).
            a_sigma2 (float): Shape parameter for the inverse gamma prior of sigma2.
                Defaults to 1.0
            b_sigma2 (float): Scale parameter for the inverse gamma prior of sigma2.
                Defaults to 1.0.
        """
        n, p = train_X.shape
        if weight is None:
            weight = torch.zeros(p, dtype=torch.float32)
        if sigma2 is None:
            sigma2 = torch.tensor(1.0, dtype=torch.float32)
        if tau2 is None:
            tau2 = torch.tensor(1.0, dtype=torch.float32)
        if lamb2 is None:
            lamb2 = torch.ones(p, dtype=torch.float32)

        if train_X.dtype != torch.float32:
            warn(
                f"train_X is not of type torch.float32, but of type {train_X.dtype}. This may cause slow sampling or errors.",
            )
        if train_Y.dtype != torch.float32:
            warn(
                f"train_Y is not of type torch.float32, but of type {train_Y.dtype}. This may cause slow sampling or errors.",
            )

        self.train_X = train_X
        self.train_Y = train_Y
        self.params = {
            "weight": weight,
            "sigma2": sigma2,
            "tau2": tau2,
            "lamb2": lamb2,
        }
        self.hyper_params = {
            "a_sigma2": a_sigma2,
            "b_sigma2": b_sigma2,
        }

        self.XtX = train_X.T @ train_X
        self.XtY = train_X.T @ train_Y

    def sample(
        self,
        num_mcs: int = 1000,
        update_parameters: bool = True,
        show_progress_bar: bool = True,
        continue_after_failure: bool = True,
    ) -> torch.Tensor:
        """HorseshoeGibbsSampler.sample() -> torch.Tensor.

        Args:
            num_mcs (int): Number of Monte Carlo steps.
            update_parameters (bool): Whether to update the initial parameters of the sampler.
                Defaults to True.
            show_progress_bar (bool): Whether to show a progress bar.
                Defaults to True.
            continue_after_failure (bool): Whether to continue sampling after a failure.
                Defaults to True.

        Returns:
            torch.Tensor: Samples of the parameters.
        """
        weight = self.params["weight"]
        sigma2 = self.params["sigma2"]
        tau2 = self.params["tau2"]
        lamb2 = self.params["lamb2"]
        a_sigma2 = self.hyper_params["a_sigma2"]
        b_sigma2 = self.hyper_params["b_sigma2"]

        for _ in tqdm(
            range(num_mcs),
            desc="Sampling",
            unit="iteration",
            disable=not show_progress_bar,
        ):
            weight, sigma2, tau2, lamb2, is_failed = _markov_transition(
                self.train_X,
                self.train_Y,
                self.XtX,
                self.XtY,
                weight,
                sigma2,
                tau2,
                lamb2,
                a_sigma2,
                b_sigma2,
            )
            if is_failed and not continue_after_failure:
                raise RuntimeError("Sampling failed..")

        if update_parameters:
            self.params["weight"] = weight
            self.params["sigma2"] = sigma2
            self.params["tau2"] = tau2
            self.params["lamb2"] = lamb2

        return weight
