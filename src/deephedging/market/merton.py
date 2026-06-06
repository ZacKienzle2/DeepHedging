"""Merton jump-diffusion simulator."""

import math
from dataclasses import dataclass

import torch

from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState

_MAX_JUMPS = 16


@dataclass(frozen=True)
class MertonSimulator:
    """Jump-diffusion sampler, exact in distribution per step.

    Log returns compound a Gaussian diffusion with a compensated
    compound-Poisson jump component whose marks are normal, so each step
    samples the exact transition law rather than an Euler approximation.
    Jump counts are drawn by inverting the truncated Poisson distribution
    function against the seeded uniform stream, because the library
    Poisson sampler accepts no generator and would silently break the
    exact-replay contract of the noise specification. The conditional
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
        if self.s0 <= 0.0:
            raise ValueError(f"s0 must be positive, got {self.s0}")
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        if self.jump_intensity < 0.0:
            raise ValueError(f"jump_intensity must be non-negative, got {self.jump_intensity}")
        if self.jump_vol < 0.0:
            raise ValueError(f"jump_vol must be non-negative, got {self.jump_vol}")
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")
        if self.jump_intensity * self.maturity / self.n_steps > 2.0:
            raise ValueError(
                "jump_intensity per step exceeds the truncated sampler's range; increase n_steps"
            )

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
        uniforms = torch.rand(shape, dtype=self.dtype, device=self.device, generator=generator)
        z_jump = torch.randn(shape, dtype=self.dtype, device=self.device, generator=generator)

        log_pmf = (
            torch.arange(_MAX_JUMPS + 1, dtype=torch.float64) * math.log(max(rate, 1e-300))
            - rate
            - torch.lgamma(torch.arange(_MAX_JUMPS + 1, dtype=torch.float64) + 1.0)
        )
        cdf = torch.exp(log_pmf).cumsum(dim=0).to(self.dtype).to(self.device)
        counts = torch.searchsorted(cdf, uniforms.reshape(-1).contiguous()).reshape(shape)
        counts = torch.clamp(counts, max=_MAX_JUMPS).to(self.dtype)

        jump_sum = self.jump_mean * counts + self.jump_vol * torch.sqrt(counts) * z_jump
        increments = drift + diffusion * z_diffusion + jump_sum
        log_returns = torch.cumsum(increments, dim=0)
        out = log_returns.new_empty((self.n_steps + 1, n_paths))
        out[0] = 0.0
        out[1:] = log_returns
        cumulative_jumps = out.new_zeros((self.n_steps + 1, n_paths))
        cumulative_jumps[1:] = torch.cumsum(counts, dim=0)
        return MarketState(spot=out.exp_().mul_(self.s0), aux={"jumps": cumulative_jumps})
