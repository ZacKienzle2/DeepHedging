"""Fixed-strike lookback option payoffs."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LookbackCall:
    """Fixed-strike lookback call paying ``max(M - K, 0)`` on the path maximum.

    Settling against the running maximum rather than the terminal price lets
    the holder capture the best level the underlying ever reached, so the
    payoff dominates the European call on every path and the contract is
    strictly more valuable. The maximum is monitored discretely on the
    rebalancing grid, inception included; a continuously monitored maximum is
    at least as high and the contract worth at least as much.

    Attributes:
        strike: Strike price K.
    """

    strike: float

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the lookback call payoff from the full path.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        return torch.clamp(paths.max(dim=0).values - self.strike, min=0.0)


@dataclass(frozen=True)
class LookbackPut:
    """Fixed-strike lookback put paying ``max(K - m, 0)`` on the path minimum.

    Settling against the running minimum lets the holder sell at the lowest
    level the underlying ever reached, so the payoff dominates the European
    put on every path. The minimum is monitored discretely on the
    rebalancing grid, inception included; a continuously monitored minimum is
    at least as low and the contract worth at least as much.

    Attributes:
        strike: Strike price K.
    """

    strike: float

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the lookback put payoff from the full path.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        return torch.clamp(self.strike - paths.min(dim=0).values, min=0.0)
