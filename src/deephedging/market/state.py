"""Market state container.

A bare price tensor cannot carry model state such as instantaneous
variance, so stochastic volatility would be invisible to feature maps.
MarketState pairs the spot grid with named auxiliary channels. The CUDA
generator later fills one output buffer per channel, so the container
maps one to one onto the kernel ABI.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class MarketState:
    """Simulated market paths with named auxiliary channels.

    Attributes:
        spot: Price grid of shape ``(n_steps + 1, n_paths)``. A trailing
            asset axis ``(n_steps + 1, n_paths, n_assets)`` is the reserved
            multi-asset convention; positions then contract the trailing
            axis in the gains reduction.
        aux: Named channels aligned with the spot grid, for example
            ``variance`` of shape ``(n_steps + 1, n_paths)`` under Heston.
    """

    spot: torch.Tensor
    aux: Mapping[str, torch.Tensor] = field(default_factory=dict)
    _cache: dict[str, torch.Tensor] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_spot(cls, spot: torch.Tensor) -> "MarketState":
        """Wraps a bare spot grid with no auxiliary channels.

        Args:
            spot: Price grid of shape ``(n_steps + 1, n_paths)``.

        Returns:
            A state carrying only the spot channel.
        """
        return cls(spot=spot)

    @property
    def n_steps(self) -> int:
        """Number of rebalancing intervals."""
        return self.spot.shape[0] - 1

    @property
    def n_paths(self) -> int:
        """Number of simulated paths."""
        return self.spot.shape[1]

    @property
    def terminal(self) -> torch.Tensor:
        """Terminal prices of shape ``(n_paths,)``."""
        return self.spot[-1]

    @property
    def device(self) -> torch.device:
        """Device holding the tensors."""
        return self.spot.device

    def running_max(self, t: int) -> torch.Tensor:
        """Running maximum of the spot over ``[0, t]`` per path.

        The full cumulative maximum is computed once and cached, so a
        per-step consumer pays ``O(T)`` total rather than ``O(T^2)``.

        Args:
            t: Index of the current rebalancing date.

        Returns:
            Running maxima of shape ``(n_paths,)``.
        """
        cached = self._cache.get("cummax")
        if cached is None:
            cached = torch.cummax(self.spot, dim=0).values
            self._cache["cummax"] = cached
        return cached[t]

    def to(self, device: torch.device | str) -> "MarketState":
        """Moves every channel to a device, returning self when already there.

        Args:
            device: Target device.

        Returns:
            A state on the target device.
        """
        if self.spot.device == torch.device(device):
            return self
        return MarketState(
            spot=self.spot.to(device),
            aux={name: channel.to(device) for name, channel in self.aux.items()},
        )
