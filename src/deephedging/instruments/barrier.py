"""Barrier option payoffs."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UpAndOutCall:
    """Up-and-out call: vanilla call payoff knocked out at a barrier.

    Pays ``max(S_T - K, 0)`` if the path never reaches the barrier and zero
    otherwise. The barrier is monitored discretely on the rebalancing grid,
    so the payoff is the discretely-monitored contract; continuously
    monitored barriers knock out more often and are worth less.

    Attributes:
        strike: Strike price K.
        barrier: Knock-out level B, strictly above the strike.
    """

    strike: float
    barrier: float

    def __post_init__(self) -> None:
        if self.barrier <= self.strike:
            raise ValueError(
                f"barrier must exceed strike, got barrier={self.barrier} strike={self.strike}"
            )

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the knocked payoff from the full path.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        alive = paths.max(dim=0).values < self.barrier
        vanilla = torch.clamp(paths[-1] - self.strike, min=0.0)
        return vanilla * alive
