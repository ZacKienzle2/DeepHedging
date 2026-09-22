"""Rough Bergomi simulator by exact Cholesky sampling on the rebalancing grid."""

import math
from dataclasses import dataclass
from functools import cache

import mpmath
import torch

from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState


def _volterra_ratio(hurst: float, x: float) -> float:
    gamma = 0.5 - hurst
    return float(
        2.0 * hurst * x**-gamma / (1.0 - gamma) * mpmath.hyp2f1(1.0, gamma, 2.0 - gamma, 1.0 / x)
    )


@cache
def rough_bergomi_factor(hurst: float, rho: float, maturity: float, n_steps: int) -> torch.Tensor:
    """Cholesky factor of the joint law of the Volterra process and price driver.

    Builds the covariance of Bayer, Friz and Gatheral (2016, Section 4) on the
    dates ``t_i = i maturity / n_steps``: ``E[W_v W_u] = u^(2H) G(v / u)`` for
    ``v >= u`` with ``G(x) = 2H x^-g / (1 - g) 2F1(1, g; 2 - g; 1 / x)`` and
    ``g = 1/2 - H`` (their eq. 4.1), ``E[W_v Z_u] = rho D_H (v^(H + 1/2) -
    (v - min(u, v))^(H + 1/2))`` with ``D_H = sqrt(2H) / (H + 1/2)``, and
    ``E[Z_v Z_u] = min(u, v)``. The hypergeometric function comes from
    mpmath, and the factor is cached per configuration.

    Args:
        hurst: Hurst exponent ``H`` in ``(0, 1/2)``.
        rho: Correlation between the variance and price drivers.
        maturity: Horizon in years.
        n_steps: Number of grid dates.

    Returns:
        The float64 lower-triangular factor of shape ``(2 n_steps, 2 n_steps)``,
        whose first ``n_steps`` rows produce the Volterra process and last
        ``n_steps`` rows the price driver at the grid dates.
    """
    times = [maturity * (i + 1) / n_steps for i in range(n_steps)]
    volterra = torch.empty(n_steps, n_steps, dtype=torch.float64)
    for i, early in enumerate(times):
        volterra[i, i] = early ** (2.0 * hurst)
        for j in range(i + 1, n_steps):
            value = early ** (2.0 * hurst) * _volterra_ratio(hurst, times[j] / early)
            volterra[i, j] = volterra[j, i] = value
    grid = torch.tensor(times, dtype=torch.float64)
    scale = rho * math.sqrt(2.0 * hurst) / (hurst + 0.5)
    lag = (grid[:, None] - torch.minimum(grid[:, None], grid)).clamp(min=0.0)
    cross = scale * (grid[:, None] ** (hurst + 0.5) - lag ** (hurst + 0.5))
    driver = torch.minimum(grid[:, None], grid)
    covariance = torch.cat(
        (torch.cat((volterra, cross), dim=1), torch.cat((cross.T, driver), dim=1)), dim=0
    )
    return torch.linalg.cholesky(covariance)


@dataclass(frozen=True)
class RoughBergomiSimulator:
    """Rough Bergomi model of Bayer, Friz and Gatheral, sampled exactly.

    Dynamics: ``dS / S = sqrt(v) dZ`` and ``v_u = xi0 E(eta W_u)`` with the
    Volterra process ``W_u = sqrt(2H) int_0^u (u - s)^(H - 1/2) dW_s``, the
    Wick exponential ``E`` and ``d<W, Z> = rho dt`` (their Section 4). The
    joint Gaussian law of ``(W, Z)`` on the rebalancing dates is sampled
    exactly through its Cholesky factor, which the authors propose and call
    slow for fine grids. At a rebalancing grid of tens of dates the factor is
    small and the sampling is one matrix product per batch. The variance is
    then exact on the grid and only the price integral is discretised, by
    the log-Euler step with the variance at the left end of each interval.
    The variance path is exposed as the ``variance`` channel.

    Attributes:
        s0: Initial spot price.
        xi0: Flat initial forward variance.
        hurst: Hurst exponent ``H`` in ``(0, 1/2)``.
        eta: Volatility of volatility.
        rho: Correlation between the variance and price drivers.
        maturity: Horizon in years.
        n_steps: Number of rebalancing intervals.
        dtype: Floating dtype of the generated paths.
        device: Device on which paths are generated.
    """

    s0: float
    xi0: float
    hurst: float
    eta: float
    rho: float
    maturity: float
    n_steps: int
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Rejects field values outside the documented domain."""
        if self.s0 <= 0.0:
            msg = f"s0 must be positive, got {self.s0}"
            raise ValueError(msg)
        if self.xi0 <= 0.0:
            msg = f"xi0 must be positive, got {self.xi0}"
            raise ValueError(msg)
        if not 0.0 < self.hurst < 0.5:
            msg = f"hurst must be in (0, 1/2), got {self.hurst}"
            raise ValueError(msg)
        if self.eta < 0.0:
            msg = f"eta must be non-negative, got {self.eta}"
            raise ValueError(msg)
        if not -1.0 < self.rho < 1.0:
            msg = f"rho must be in (-1, 1), got {self.rho}"
            raise ValueError(msg)
        if self.maturity <= 0.0:
            msg = f"maturity must be positive, got {self.maturity}"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = f"n_steps must be at least 1, got {self.n_steps}"
            raise ValueError(msg)

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates rough Bergomi market paths.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state with the spot grid and the ``variance`` channel, both
            of shape ``(n_steps + 1, n_paths)``.
        """
        generator = noise.torch_generator(self.device) if noise is not None else None
        factor = rough_bergomi_factor(self.hurst, self.rho, self.maturity, self.n_steps)
        factor = factor.to(self.device)
        normals = torch.randn(
            (2 * self.n_steps, n_paths),
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )
        paths = factor @ normals
        volterra, driver = paths[: self.n_steps], paths[self.n_steps :]
        dt = self.maturity / self.n_steps
        times = torch.arange(1, self.n_steps + 1, dtype=torch.float64, device=self.device) * dt
        variance = volterra.new_empty((self.n_steps + 1, n_paths))
        variance[0] = self.xi0
        variance[1:] = self.xi0 * torch.exp(
            self.eta * volterra - 0.5 * self.eta**2 * times[:, None] ** (2.0 * self.hurst)
        )
        increments = torch.diff(driver, dim=0, prepend=torch.zeros_like(driver[:1]))
        held = variance[:-1]
        log_returns = torch.cumsum(held.sqrt() * increments - 0.5 * held * dt, dim=0)
        log_spot = torch.cat((torch.zeros_like(log_returns[:1]), log_returns))
        spot = self.s0 * torch.exp(log_spot)
        return MarketState(spot=spot.to(self.dtype), aux={"variance": variance.to(self.dtype)})
