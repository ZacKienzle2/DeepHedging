"""Heston stochastic volatility simulator."""

import math
from dataclasses import dataclass

import torch

from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState


@dataclass(frozen=True)
class HestonSimulator:
    """Heston model sampled with the full-truncation Euler scheme.

    Dynamics: ``dS = mu S dt + sqrt(v) S dW_S`` and
    ``dv = kappa (theta - v) dt + xi sqrt(v) dW_v`` with
    ``d<W_S, W_v> = rho dt``. The full-truncation scheme (Lord et al.)
    clamps the variance at zero inside both drift and diffusion, which is
    the standard bias-minimising discretisation when the Feller condition
    ``2 kappa theta >= xi^2`` is violated. The price state is evolved in
    log space, matching the precision contract of the GBM simulator. The
    clamped variance path is exposed as the ``variance`` channel so
    volatility-aware feature maps can observe it.

    Attributes:
        s0: Initial spot price.
        v0: Initial variance.
        kappa: Mean-reversion speed of the variance.
        theta: Long-run variance level.
        xi: Volatility of variance.
        rho: Correlation between price and variance Brownian drivers.
        maturity: Horizon in years.
        n_steps: Number of rebalancing intervals.
        mu: Drift under the simulation measure.
        dtype: Floating dtype of the generated paths.
        device: Device on which paths are generated.
    """

    s0: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    maturity: float
    n_steps: int
    mu: float = 0.0
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.s0 <= 0.0:
            raise ValueError(f"s0 must be positive, got {self.s0}")
        if self.v0 < 0.0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta < 0.0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if self.xi < 0.0:
            raise ValueError(f"xi must be non-negative, got {self.xi}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates Heston market paths.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state whose spot grid satisfies ``spot[0] == s0`` and
            whose ``variance`` channel holds the clamped variance path.
        """
        generator = noise.torch_generator(self.device) if noise is not None else None
        dt = self.maturity / self.n_steps
        sqrt_dt = dt**0.5
        z = torch.randn(
            (2, self.n_steps, n_paths),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        z_spot = z[0]
        z_var = self.rho * z[0] + math.sqrt(1.0 - self.rho**2) * z[1]
        log_return = torch.zeros((n_paths,), dtype=self.dtype, device=self.device)
        variance = torch.full((n_paths,), self.v0, dtype=self.dtype, device=self.device)
        out = torch.empty((self.n_steps + 1, n_paths), dtype=self.dtype, device=self.device)
        var_out = torch.empty_like(out)
        out[0] = 0.0
        var_out[0] = variance
        for k in range(self.n_steps):
            variance_plus = torch.clamp(variance, min=0.0)
            vol = torch.sqrt(variance_plus)
            log_return = (
                log_return + (self.mu - 0.5 * variance_plus) * dt + vol * sqrt_dt * z_spot[k]
            )
            variance = (
                variance
                + self.kappa * (self.theta - variance_plus) * dt
                + self.xi * vol * sqrt_dt * z_var[k]
            )
            out[k + 1] = log_return
            var_out[k + 1] = torch.clamp(variance, min=0.0)
        return MarketState(spot=out.exp_().mul_(self.s0), aux={"variance": var_out})
