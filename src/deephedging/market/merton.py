"""Merton jump-diffusion simulator."""

import math
from dataclasses import dataclass

import torch

from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState


@dataclass(frozen=True)
class MertonSimulator:
    """Jump-diffusion sampler, exact in distribution per step.

    Log returns compound a Gaussian diffusion with a compensated
    compound-Poisson jump component whose marks are normal, so each step
    samples the exact transition law rather than an Euler approximation.
    Jump counts come from ``torch.poisson`` driven by the seeded generator,
    which replays bitwise on both devices and has no truncation, so any
    jump intensity per step is admissible. The conditional
    jump sum given a count of ``n`` is normal with mean ``n * jump_mean``
    and variance ``n * jump_vol ** 2``, which collapses the per-jump loop
    into one Gaussian draw. The compensator keeps the price a martingale
    under zero drift. The cumulative jump count is exposed as the
    ``jumps`` channel for jump-aware feature maps.

    Attributes:
        s0: Initial spot price.
        sigma: Diffusive volatility.
        jump_intensity: Expected jumps per year.
        jump_mean: Mean of the lognormal jump mark exponent.
        jump_vol: Standard deviation of the jump mark exponent.
        maturity: Horizon in years.
        n_steps: Number of rebalancing intervals.
        mu: Drift under the simulation measure.
        dtype: Floating dtype of the generated paths.
        device: Device on which paths are generated.
    """

    s0: float
    sigma: float
    jump_intensity: float
    jump_mean: float
    jump_vol: float
    maturity: float
    n_steps: int
    mu: float = 0.0
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Rejects field values outside the documented domain."""
        if self.s0 <= 0.0:
            msg = f"s0 must be positive, got {self.s0}"
            raise ValueError(msg)
        if self.sigma < 0.0:
            msg = f"sigma must be non-negative, got {self.sigma}"
            raise ValueError(msg)
        if self.jump_intensity < 0.0:
            msg = f"jump_intensity must be non-negative, got {self.jump_intensity}"
            raise ValueError(msg)
        if self.jump_vol < 0.0:
            msg = f"jump_vol must be non-negative, got {self.jump_vol}"
            raise ValueError(msg)
        if self.maturity <= 0.0:
            msg = f"maturity must be positive, got {self.maturity}"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = f"n_steps must be at least 1, got {self.n_steps}"
            raise ValueError(msg)

    @property
    def mean_jump_size(self) -> float:
        """Expected proportional jump ``E[exp(J)] - 1``."""
        return math.exp(self.jump_mean + 0.5 * self.jump_vol**2) - 1.0

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates jump-diffusion market paths.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state whose spot grid satisfies ``spot[0] == s0`` and
            whose ``jumps`` channel counts jumps cumulatively per path.
        """
        generator = noise.torch_generator(self.device) if noise is not None else None
        dt = self.maturity / self.n_steps
        rate = self.jump_intensity * dt
        compensator = self.jump_intensity * self.mean_jump_size
        drift = (self.mu - compensator - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * dt**0.5

        shape = (self.n_steps, n_paths)
        z_diffusion = torch.randn(shape, dtype=self.dtype, device=self.device, generator=generator)
        rates = torch.full(shape, rate, dtype=self.dtype, device=self.device)
        counts = torch.poisson(rates, generator=generator)
        z_jump = torch.randn(shape, dtype=self.dtype, device=self.device, generator=generator)

        jump_sum = self.jump_mean * counts + self.jump_vol * torch.sqrt(counts) * z_jump
        increments = drift + diffusion * z_diffusion + jump_sum
        log_returns = torch.cumsum(increments, dim=0)
        out = log_returns.new_empty((self.n_steps + 1, n_paths))
        out[0] = 0.0
        out[1:] = log_returns
        cumulative_jumps = out.new_zeros((self.n_steps + 1, n_paths))
        cumulative_jumps[1:] = torch.cumsum(counts, dim=0)
        return MarketState(spot=out.exp_().mul_(self.s0), aux={"jumps": cumulative_jumps})
