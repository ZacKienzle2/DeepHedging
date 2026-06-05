"""Vanilla European payoffs."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EuropeanCall:
    """European call option payoff ``max(S_T - K, 0)``.

    Attributes:
        strike: Strike price K.
    """

    strike: float

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the call payoff from the terminal prices.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        return torch.clamp(paths[-1] - self.strike, min=0.0)


@dataclass(frozen=True)
class EuropeanPut:
    """European put option payoff ``max(K - S_T, 0)``.

    Attributes:
        strike: Strike price K.
    """

    strike: float

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the put payoff from the terminal prices.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        return torch.clamp(self.strike - paths[-1], min=0.0)
