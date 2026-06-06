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
class PathFolds:
    """Per-path statistics produced without materialising the grid.

    Attributes:
        terminal: Terminal spot per path, shape ``(n_paths,)``.
        running_max: Path maximum including inception, same shape.
        running_min: Path minimum including inception, same shape.
    """

    terminal: torch.Tensor
    running_max: torch.Tensor
    running_min: torch.Tensor


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

    def log_relative(self, t: int) -> torch.Tensor:
        """Log of the spot relative to inception, per path.

        The full log grid is computed once and cached, so a per-step
        consumer pays one batched kernel per episode rather than one
        elementwise log per rebalancing date.

        Args:
            t: Index of the current rebalancing date.

        Returns:
            ``log(spot[t] / spot[0])`` of shape ``(n_paths,)``, with a
            trailing asset axis when the spot grid carries one.
        """
        cached = self._cache.get("log_relative")
        if cached is None:
            logs = torch.log(self.spot)
            cached = logs - logs[0]
            self._cache["log_relative"] = cached
        return cached[t]

    def realised_variance(self, t: int, window: int) -> torch.Tensor:
        """Mean squared log return over the trailing window, per path.

        The squared-return cumulative sum is computed once and cached, so
        a per-step consumer pays one batched kernel per episode. The
        value is per step, not annualised; consumers divide by the step
        size. At inception no return exists and the estimate is zero,
        and shorter prefixes average over what is available, keeping the
        statistic adapted.

        Args:
            t: Index of the current rebalancing date.
            window: Number of trailing returns to average over.

        Returns:
            Per-step realised variance of shape ``(n_paths,)``.

        Raises:
            ValueError: If ``window`` is not positive.
        """
        if window < 1:
            raise ValueError(f"window must be positive, got {window}")
        cached = self._cache.get("squared_return_cumsum")
        if cached is None:
            log_grid = torch.log(self.spot)
            squared = (log_grid[1:] - log_grid[:-1]) ** 2
            cached = squared.new_zeros((self.n_steps + 1, *squared.shape[1:]))
            cached[1:] = torch.cumsum(squared, dim=0)
            self._cache["squared_return_cumsum"] = cached
        if t == 0:
            return cached[0]
        start = max(t - window, 0)
        return (cached[t] - cached[start]) / (t - start)

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
