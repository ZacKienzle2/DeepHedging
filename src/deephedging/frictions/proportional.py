"""Proportional transaction costs."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProportionalCost:
    """Cost proportional to traded notional: ``rate * price * |trade|``.

    Attributes:
        rate: Proportional cost rate, e.g. 1e-3 for 10 basis points.
    """

    rate: float

    def __post_init__(self) -> None:
        if self.rate < 0.0:
            raise ValueError(f"rate must be non-negative, got {self.rate}")

    def __call__(self, trade: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """Computes proportional cost of a trade.

        Args:
            trade: Signed position change.
            price: Execution price.

        Returns:
            ``rate * price * |trade|`` elementwise.
        """
        return self.rate * price * trade.abs()


@dataclass(frozen=True)
class NoCost:
    """Frictionless execution."""

    def __call__(self, trade: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """Returns zero cost with the broadcast shape of the inputs.

        Args:
            trade: Signed position change.
            price: Execution price.

        Returns:
            Zeros with the broadcast shape of ``trade`` and ``price``.
        """
        return torch.zeros_like(trade * price)
