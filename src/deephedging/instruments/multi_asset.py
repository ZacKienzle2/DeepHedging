"""Adapters between single-asset payoffs and multi-asset grids.

A liability written on one asset of a multi-asset market needs the
payoff to read only its own column of the spot grid. The adapter keeps
the single-asset payoffs untouched and composable, so any of them can
serve as the liability in a multi-instrument hedging problem without
growing an asset-axis branch of their own.
"""

from dataclasses import dataclass

import torch

from deephedging.instruments.base import Payoff


@dataclass(frozen=True)
class SingleAssetPayoff:
    """Applies a single-asset payoff to one column of a multi-asset grid.

    Attributes:
        inner: The single-asset payoff defining the liability.
        asset: Index of the asset the liability is written on.
    """

    inner: Payoff
    asset: int = 0

    def __post_init__(self) -> None:
        if self.asset < 0:
            raise ValueError(f"asset must be non-negative, got {self.asset}")

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the inner payoff on the selected asset column.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths, n_assets)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.

        Raises:
            IndexError: If the grid has fewer assets than the index.
        """
        return self.inner(paths[..., self.asset])
