"""Abstract base class for sampling methods."""

import abc

import torch


@abc.abstractmethod
class BaseSampler(abc.ABC):
    """Abstract base class for sampling methods."""

    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        """Samples the parameters of the model."""
        pass
