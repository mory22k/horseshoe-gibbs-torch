"""Fast multivariate Gaussian posterior sampling."""

import logging
from math import prod
from typing import Optional, Tuple

import torch
from torch.types import _size

logger = logging.getLogger(__name__)

EPSILON = 1e-15


class FastMultivariateGaussianMixtureBCM:
    r"""Sampler for Gaussian posterior of parameters using [1].

    This class implements fast sampling of theta from
        p(theta | X, y, D, sigma),
    where
        p(theta | X, y, D, sigma) = N(A^{-1} X^T y, sigma^2 A^{-1}),
        A = X^T X + diag(D)^{-1}.

    Args:
        X (torch.Tensor): Design matrix of shape (n, p).
        y (torch.Tensor): Target vector of shape (n,).
        D_diag (torch.Tensor): Diagonal of prior precision matrix D, shape (p,).
        sigma (torch.Tensor): Noise standard deviation (scalar).
        show_warn (bool): Whether to log a warning if linear solve fails.

    References:
        [1] A. Bhattacharya, A. Chakraborty, and B. K. Mallick, Fast sampling with Gaussian scale-mixture priors in high-dimensional regression, Biometrika 103, 985 (2016).
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        D_diag: torch.Tensor,
        sigma2: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the FastMultivariateGaussianMixtureBCM sampler.

        Args:
            X (torch.Tensor): Design matrix of shape (n, p).
            y (torch.Tensor): Target vector of shape (n,).
            D_diag (torch.Tensor): Diagonal of prior precision matrix D, shape (p,).
            sigma2 (torch.Tensor): Noise variance (scalar).
            device (Optional[torch.device], optional): Device to use for computations. If None, uses the device of X. Defaults to None.
        """
        if device is None:
            device = X.device
        else:
            device = torch.device(device)
        self.device = device

        self.X = X.to(self.device)
        self.y = y.to(self.device)
        self.D_diag = D_diag.to(self.device)
        self.sigma = sigma2.to(self.device).sqrt()
        self.n, self.p = X.shape
        self.XD = D_diag * self.X
        self.lhs = self.XD @ self.X.T + torch.eye(self.n, device=self.X.device)

    def _sample_single(self) -> torch.Tensor:
        """Draw a sample from the Gaussian posterior.

        Returns:
            torch.Tensor: Sampled parameter vector of shape (p,).
        """
        u = (
            torch.randn(size=(self.p,), dtype=torch.float32, device=self.device)
            * self.sigma
            * torch.sqrt(self.D_diag)
        )
        v = torch.randn(self.n, dtype=torch.float32, device=self.X.device) * self.sigma

        lhs = self.lhs
        rhs = self.y - self.X @ u - v
        try:
            sol = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            logger.warning(
                "Solving linear system failed in FastMultivariateGaussianMixtureBCM. "
                "Using least squares instead."
            )
            sol = torch.linalg.lstsq(lhs, rhs).solution

        theta: torch.Tensor = u + self.XD.T @ sol
        return theta

    def sample(self, size: Tuple[int, ...] = ()) -> torch.Tensor:
        """Draw samples from the Gaussian posterior.

        Args:
            size (Tuple[int, ...], optional): Desired output shape. Defaults to ().

        Returns:
            torch.Tensor: Sampled parameter vector of shape (size + (p,)).
        """
        if not size:
            return self._sample_single()

        sample_size = prod(size)

        # u: (sample_size, p)
        # v: (sample_size, n)
        u = (
            torch.randn(sample_size, self.p, dtype=torch.float32, device=self.device)
            * self.sigma
            * torch.sqrt(self.D_diag)
        )
        v = (
            torch.randn(sample_size, self.n, dtype=torch.float32, device=self.device)
            * self.sigma
        )

        # lhs: (n, n)
        # rhs: (sample_size, n)
        rhs = (
            self.y.unsqueeze(0)  # (1, n)
            - (self.X @ u.T).T  # (sample_size, n)
            - v  # (sample_size, n)
        )

        # sol: (sample_size, n)
        try:
            sol = torch.linalg.solve(self.lhs, rhs.T).T
        except RuntimeError:
            logger.warning(
                "torch.linalg.solve failed in batched sample; falling back to lstsq."
            )
            sol = torch.linalg.lstsq(self.lhs, rhs).solution.T

        # theta: (sample_size, p)
        theta: torch.Tensor = (
            u  # (sample_size, p)
            + (self.XD.T @ sol.T).T  # (sample_size, p)
        )
        return theta.view(size + (self.p,))


class FastMultivariateGaussianMixtureRue:
    """Sampler for Gaussian posterior using Rue's algorithm [1].

    This class implements fast sampling of theta from
        p(theta | X, y, D, sigma),
    where
        p(theta | X, y, D, sigma) = N(A^{-1} X^T y, sigma^2 A^{-1}),
        A = X^T X + diag(D)^{-1}.

    Args:
        XtX (torch.Tensor): X^T X matrix of shape (p, p).
        Xty (torch.Tensor): X^T y vector of shape (p,).
        D_diag (torch.Tensor): Diagonal of prior precision matrix D, shape (p,).
        sigma2 (torch.Tensor): Noise variance (scalar).

    References:
        [1] H. Rue, Fast sampling of Gaussian Markov random fields, J. R. Stat. Soc. Series B Stat. Methodol. 63, 325 (2001).
    """

    def __init__(
        self,
        XtX: torch.Tensor,
        Xty: torch.Tensor,
        D_diag: torch.Tensor,
        sigma2: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the FastMultivariateGaussianMixtureRue sampler.

        Args:
            XtX (torch.Tensor): X^T X matrix of shape (p, p).
            Xty (torch.Tensor): X^T y vector of shape (p,).
            D_diag (torch.Tensor): Diagonal of prior precision matrix D, shape (p,).
            sigma2 (torch.Tensor): Noise variance (scalar).
            device (Optional[torch.device], optional): Device to use for computations. If None, uses the device of XtX. Defaults to None.
        """
        self.p = XtX.shape[0]

        if device is None:
            device = XtX.device
        else:
            device = torch.device(device)
        self.device = device

        XtX = XtX.to(self.device)
        Xty = Xty.to(self.device)
        D_diag = D_diag.to(self.device)
        sigma2 = sigma2.to(self.device)

        XtX_ = XtX / sigma2
        Xty_ = Xty / sigma2
        D_diag_ = D_diag * sigma2
        D_diag_inv_ = 1.0 / D_diag_
        D_inv_ = torch.diag(D_diag_inv_)
        try:
            L = torch.linalg.cholesky(XtX_ + D_inv_)
        except RuntimeError:
            _M = XtX_ + D_inv_
            _S = (_M + _M.T) / 2.0
            _max_eig_S = torch.max(torch.linalg.eigvals(_S))
            L = torch.linalg.cholesky(
                _S + _max_eig_S * EPSILON * torch.eye(_S.shape[0])
            )
        self.L: torch.Tensor = L
        self.v: torch.Tensor = torch.linalg.solve(self.L, Xty_)
        self.m: torch.Tensor = torch.linalg.solve(self.L.T, self.v)

    def _sample_single(self) -> torch.Tensor:
        """Draw a sample from the Gaussian posterior.

        Returns:
            torch.Tensor: Sampled parameter vector of shape (p,).
        """
        z: torch.Tensor = torch.randn(self.p, device=self.device)
        w: torch.Tensor = torch.linalg.solve(self.L.T, z)

        theta: torch.Tensor = self.m + w
        return theta

    def sample(self, size: Tuple[int, ...] = ()) -> torch.Tensor:
        """Draw a sample from the Gaussian posterior.

        Args:
            size (Tuple[int, ...], optional): Desired output shape. Defaults to ().

        Returns:
            torch.Tensor: Sampled parameter vector of shape (p,).
        """
        if not size:
            return self._sample_single()

        sample_size = prod(size)

        # z: (sample_size, p)
        # w: (sample_size, p)
        z: torch.Tensor = torch.randn(
            sample_size, self.p, dtype=torch.float32, device=self.device
        )
        w: torch.Tensor = torch.linalg.solve(self.L.T, z.T).T

        # m: (1, p)
        # theta: (sample_size, p)
        theta: torch.Tensor = self.m.unsqueeze(0) + w
        return theta.view(size + (self.p,))
