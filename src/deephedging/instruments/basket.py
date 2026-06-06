"""Multi-asset basket payoffs."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BasketCall:
    """Call on the weighted arithmetic average of the terminal prices.

    Attributes:
        strike: Strike price K.
        weights: Per-asset weights; need not sum to one.
    """

    strike: float
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights must contain at least one asset")

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the basket call payoff from the terminal prices.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths, n_assets)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        weights = paths.new_tensor(self.weights)
        basket = (paths[-1] * weights).sum(dim=-1)
        return torch.clamp(basket - self.strike, min=0.0)


@dataclass(frozen=True)
class GeometricBasketCall:
    """Call on the geometric average of the terminal prices.

    The geometric average of lognormal assets is itself lognormal, so
    this contract admits a Black-Scholes-type closed form under
    correlated GBM and serves as the multi-asset golden reference.

    Attributes:
        strike: Strike price K.
    """

    strike: float

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the geometric basket call payoff.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths, n_assets)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        geometric = torch.exp(torch.log(paths[-1]).mean(dim=-1))
        return torch.clamp(geometric - self.strike, min=0.0)
