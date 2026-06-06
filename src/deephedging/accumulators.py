"""Online path-functional folds.

A streaming path generator produces one step at a time and never
materialises the full grid, so payoffs and features that read the whole
path must be expressible as per-step folds. This module defines that
contract in Python first; the fused CUDA generator implements the same
``init``, ``update``, ``finalize`` shape inside the kernel, and the
eager engine remains the parity oracle.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from deephedging.market.state import MarketState


@runtime_checkable
class PathAccumulator(Protocol):
    """Folds a per-path statistic over the spot grid one step at a time.

    Folds broadcast elementwise, so on a trailing-asset grid the carry
    keeps the asset axis and callers contract it themselves.
    """

    def init(self, spot0: torch.Tensor) -> torch.Tensor:
        """Initialises the fold state from the inception spot.

        Args:
            spot0: Spot at inception, shape ``(n_paths,)``.

        Returns:
            Initial fold state, shape ``(n_paths,)``.
        """
        ...

    def update(self, carry: torch.Tensor, spot: torch.Tensor) -> torch.Tensor:
        """Advances the fold by one monitoring date.

        Args:
            carry: Fold state entering the date, shape ``(n_paths,)``.
            spot: Spot at the date, shape ``(n_paths,)``.

        Returns:
            Fold state after the date, shape ``(n_paths,)``.
        """
        ...


@dataclass(frozen=True)
class RunningMaxAccumulator:
    """Running maximum of the spot, the up-and-out sufficient statistic."""

    def init(self, spot0: torch.Tensor) -> torch.Tensor:
        """Starts the maximum at the inception spot.

        Args:
            spot0: Spot at inception, shape ``(n_paths,)``.

        Returns:
            A clone of the inception spot.
        """
        return spot0.clone()

    def update(self, carry: torch.Tensor, spot: torch.Tensor) -> torch.Tensor:
        """Takes the elementwise maximum with the current spot.

        Args:
            carry: Running maximum so far.
            spot: Spot at the date.

        Returns:
            Updated running maximum.
        """
        return torch.maximum(carry, spot)


@dataclass(frozen=True)
class BarrierAliveAccumulator:
    """Survival indicator for an up-and-out barrier.

    Attributes:
        barrier: Knock-out level monitored discretely, inception included.
    """

    barrier: float

    def init(self, spot0: torch.Tensor) -> torch.Tensor:
        """Marks paths alive when inception is below the barrier.

        Args:
            spot0: Spot at inception, shape ``(n_paths,)``.

        Returns:
            One where alive, zero where knocked.
        """
        return (spot0 < self.barrier).to(spot0.dtype)

    def update(self, carry: torch.Tensor, spot: torch.Tensor) -> torch.Tensor:
        """Knocks paths whose spot reaches the barrier.

        Args:
            carry: Survival indicator so far.
            spot: Spot at the date.

        Returns:
            Updated survival indicator.
        """
        return carry * (spot < self.barrier).to(spot.dtype)


def fold_path(state: MarketState, accumulator: PathAccumulator) -> torch.Tensor:
    """Runs an accumulator over the full grid, the streaming reference.

    Iterates the monitoring dates exactly as a streaming generator would,
    so a fused kernel producing the same folds is testable against this
    function while the materialised grid remains available as the oracle.

    Args:
        state: Simulated market state.
        accumulator: The fold to run.

    Returns:
        Final fold state, shape ``(n_paths,)``.
    """
    carry = accumulator.init(state.spot[0])
    for t in range(1, state.n_steps + 1):
        carry = accumulator.update(carry, state.spot[t])
    return carry
