"""Market simulator interface."""

from typing import Protocol, runtime_checkable

from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState


@runtime_checkable
class PathSimulator(Protocol):
    """Generates market paths in time-major layout.

    Implementations return a :class:`MarketState` whose spot grid has shape
    ``(n_steps + 1, n_paths)`` so that the per-step slice consumed by a
    sequential hedging policy is contiguous. The time-major layout
    coalesces the access pattern executed once per rebalancing date, which
    dominates the once-per-episode payoff reduction.
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

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates market paths.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state with spot grid ``(n_steps + 1, n_paths)``.
        """
        ...
