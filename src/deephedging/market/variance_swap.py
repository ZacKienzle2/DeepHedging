"""Two-asset Heston market with a tradable variance swap.

The variance swap pays the average realised variance at its own
maturity, so its time-t value decomposes into the variance accrued so
far plus the conditional expectation of the remainder, which is affine
in the instantaneous variance because the CIR conditional mean solves a
linear ODE. The swap value therefore prices off the same simulated
variance path that drives the spot, and exposing it as a second asset
gives a hedging policy a direct instrument against volatility risk that
the spot alone cannot span.
"""

import math
from dataclasses import dataclass

import torch

from deephedging.market.cuda import CudaHestonSimulator
from deephedging.market.heston import HestonSimulator
from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState


@dataclass(frozen=True)
class HestonVarianceSwapSimulator:
    """Heston paths extended with a variance swap as a second asset.

    The spot grid carries a trailing asset axis with the spot first and
    the swap value second. The swap accrues realised variance by the
    trapezoid rule, which removes the quadrature bias the left rule
    carries whenever the variance starts away from its long-run level,
    and values the remaining leg from the CIR conditional expectation.
    The accrual and the valuation read the same clamped variance grid,
    so the swap price increments contain no convention mismatch a
    trained policy could exploit as spurious arbitrage. The swap
    maturity must strictly exceed the hedging horizon, which keeps the
    swap value strictly positive on the whole grid and the log-relative
    feature finite.

    Attributes:
        heston: The single-asset Heston market being extended; the
            eager and the fused-kernel samplers are interchangeable
            because both expose the clamped variance channel.
        vs_maturity: Variance swap maturity in years, strictly greater
            than the hedging horizon.
    """

    heston: HestonSimulator | CudaHestonSimulator
    vs_maturity: float

    def __post_init__(self) -> None:
        if self.vs_maturity <= self.heston.maturity:
            raise ValueError(
                f"vs_maturity must exceed the hedging horizon, got "
                f"vs_maturity={self.vs_maturity} maturity={self.heston.maturity}"
            )

    @property
    def n_steps(self) -> int:
        """Number of rebalancing intervals."""
        return self.heston.n_steps

    @property
    def maturity(self) -> float:
        """Horizon in years."""
        return self.heston.maturity

    @property
    def device(self) -> str:
        """Device on which paths are generated."""
        return self.heston.device

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates the two-asset market.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state whose spot grid has shape
            ``(n_steps + 1, n_paths, 2)`` holding spot then swap value,
            with the clamped ``variance`` channel attached.
        """
        return self._extend(self.heston.simulate(n_paths, noise=noise))

    def simulate_with_offset(self, n_paths: int, seed: int, offset: torch.Tensor) -> MarketState:
        """Simulates the two-asset market addressed by a device offset.

        Delegates generation to the wrapped sampler's offset entry
        point and builds the swap from the generated variance with
        tensor operations only, so the whole construction is capturable
        in a CUDA graph and replays draw fresh batches as the offset
        updates.

        Args:
            n_paths: Number of independent paths.
            seed: Philox seed shared with the offset's stream family.
            offset: One-element int64 CUDA tensor holding the shifted
                subsequence base.

        Returns:
            Market state whose spot grid has shape
            ``(n_steps + 1, n_paths, 2)`` holding spot then swap value,
            with the clamped ``variance`` channel attached.

        Raises:
            AttributeError: If the wrapped sampler offers no offset
                entry point, which the eager sampler does not.
        """
        if not isinstance(self.heston, CudaHestonSimulator):
            raise AttributeError("simulate_with_offset requires the fused Heston sampler")
        return self._extend(self.heston.simulate_with_offset(n_paths, seed, offset))

    def _extend(self, state: MarketState) -> MarketState:
        variance = state.aux["variance"]
        dt = self.heston.maturity / self.heston.n_steps
        accrued = variance.new_zeros(variance.shape)
        accrued[1:] = torch.cumsum(0.5 * dt * (variance[:-1] + variance[1:]), dim=0)
        kappa = self.heston.kappa
        theta = self.heston.theta
        remaining = self.vs_maturity - dt * torch.arange(
            self.heston.n_steps + 1, dtype=variance.dtype, device=variance.device
        ).unsqueeze(1)
        decay = (1.0 - torch.exp(-kappa * remaining)) / kappa
        swap = (accrued + (variance - theta) * decay + theta * remaining) / self.vs_maturity
        return MarketState(
            spot=torch.stack((state.spot, swap), dim=-1),
            aux={"variance": variance},
        )

    def initial_swap_value(self) -> float:
        """Computes the closed-form swap value at inception.

        Returns:
            The time-zero variance swap value.
        """
        kappa = self.heston.kappa
        theta = self.heston.theta
        decay = (1.0 - math.exp(-kappa * self.vs_maturity)) / kappa
        return ((self.heston.v0 - theta) * decay + theta * self.vs_maturity) / self.vs_maturity
