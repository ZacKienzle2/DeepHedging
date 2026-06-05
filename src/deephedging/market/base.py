"""Market simulator interface."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class PathSimulator(Protocol):
    """Generates asset price paths in time-major layout.

    Implementations return tensors of shape ``(n_steps + 1, n_paths)`` so that
    the per-step slice consumed by a sequential hedging policy is contiguous.
    The time-major layout coalesces the access pattern executed once per
    rebalancing date, which dominates the once-per-episode payoff reduction.
    """

    @property
    def n_steps(self) -> int:
        """Number of rebalancing intervals."""
        ...

    @property
    def maturity(self) -> float:
        """Horizon in years."""
        ...

    @property
    def device(self) -> str:
        """Device on which paths are generated."""
        ...

    def simulate(self, n_paths: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Simulates price paths.

        Args:
            n_paths: Number of independent paths.
            generator: Optional RNG for reproducible draws.

        Returns:
            Price paths of shape ``(n_steps + 1, n_paths)``.
        """
        ...
