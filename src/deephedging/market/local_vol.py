"""Local volatility simulator."""

from dataclasses import dataclass

import torch

from deephedging.calibration.dupire import LocalVolSurface
from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState


@dataclass(frozen=True)
class LocalVolSimulator:
    """Log-Euler sampler under a Dupire local volatility surface.

    Each step looks the volatility up at the current calendar time and
    spot by bilinear interpolation on the surface grid, clamped to the
    grid edges so paths wandering past the wings see the boundary
    volatility rather than an extrapolated artefact. By Dupire's
    construction the one-dimensional marginals reproduce the vanilla
    prices of the generating surface, which the golden test verifies by
    repricing through the whole calibrate, build, simulate chain.

    Attributes:
        s0: Initial spot price.
        surface: Local volatility surface.
        maturity: Horizon in years.
        n_steps: Number of rebalancing intervals.
        mu: Drift under the simulation measure.
        dtype: Floating dtype of the generated paths.
        device: Device on which paths are generated.
    """

    s0: float
    surface: LocalVolSurface
    maturity: float
    n_steps: int
    mu: float = 0.0
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.s0 <= 0.0:
            raise ValueError(f"s0 must be positive, got {self.s0}")
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates local volatility market paths.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state whose spot grid satisfies ``spot[0] == s0``.
        """
        generator = noise.torch_generator(self.device) if noise is not None else None
        dt = self.maturity / self.n_steps
        sqrt_dt = dt**0.5
        strikes = self.surface.strikes.to(self.dtype).to(self.device)
        strike_low = strikes[0]
        strike_high = strikes[-1]
        z = torch.randn(
            (self.n_steps, n_paths),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        log_return = torch.zeros((n_paths,), dtype=self.dtype, device=self.device)
        out = torch.empty((self.n_steps + 1, n_paths), dtype=self.dtype, device=self.device)
        out[0] = 0.0
        for k in range(self.n_steps):
            row = self.surface.vol_row(k * dt).to(self.dtype).to(self.device)
            spot = self.s0 * torch.exp(log_return)
            clamped = torch.clamp(spot, min=strike_low, max=strike_high)
            index = torch.clamp(
                torch.searchsorted(strikes, clamped), min=1, max=strikes.shape[0] - 1
            )
            left = strikes[index - 1]
            weight = (clamped - left) / (strikes[index] - left)
            vol = (1.0 - weight) * row[index - 1] + weight * row[index]
            log_return = log_return + (self.mu - 0.5 * vol**2) * dt + vol * sqrt_dt * z[k]
            out[k + 1] = log_return
        return MarketState(spot=out.exp_().mul_(self.s0))
