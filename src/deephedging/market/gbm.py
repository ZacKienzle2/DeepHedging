"""Geometric Brownian motion simulator."""

import math
from collections import deque
from dataclasses import dataclass
from functools import cache
from typing import Literal

import torch

from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState


@cache
def brownian_bridge_matrix(n_steps: int) -> torch.Tensor:
    """Builds the Brownian bridge construction matrix on a unit-spaced grid.

    Row ``i`` expresses the level at time ``i + 1`` as a combination of the
    normals, column ``0`` fixing the terminal level and each later column one
    midpoint in breadth-first bisection order, which generalises Glasserman's
    Figure 3.2 from powers of two to any grid. The matrix ``A`` satisfies
    ``A A^T = min(i, j)``, and a grid of spacing ``dt`` scales it by
    ``sqrt(dt)``.

    Args:
        n_steps: Number of grid dates.

    Returns:
        The float64 matrix of shape ``(n_steps, n_steps)``.
    """
    basis = torch.eye(n_steps, dtype=torch.float64)
    levels = torch.zeros(n_steps + 1, n_steps, dtype=torch.float64)
    levels[n_steps] = math.sqrt(n_steps) * basis[0]
    intervals = deque([(0, n_steps)])
    column = 1
    while intervals:
        left, right = intervals.popleft()
        if right - left < 2:
            continue
        middle = (left + right) // 2
        mean = ((right - middle) * levels[left] + (middle - left) * levels[right]) / (right - left)
        spread = math.sqrt((middle - left) * (right - middle) / (right - left))
        levels[middle] = mean + spread * basis[column]
        column += 1
        intervals.extend(((left, middle), (middle, right)))
    return levels[1:]


@dataclass(frozen=True)
class GBMSimulator:
    """Exact-in-distribution GBM sampler evolved in log space.

    Log-space evolution keeps the accumulated state additive, bounding the
    floating-point error growth at ``sqrt(n_steps)`` and avoiding the biased
    increment-dropping failure mode of a multiplicative price-space recursion
    in reduced precision.

    The ``sobol`` sampler replaces the pseudo-random normals with randomised
    quasi-Monte Carlo. Each call draws a scrambled Sobol point set, whose
    linear digit scrambling (Glasserman, 2004, section 5.4) makes every call
    an independent replicate, and builds the path by the Brownian bridge
    construction of Glasserman's section 3.1.1, so the first and best
    distributed Sobol coordinate fixes the terminal level and later ones add
    successively finer detail. Glasserman's Table 5.6 finds the bridge
    ahead of the principal components construction on barrier options, and
    a clamped hedge is path-dependent in the same way. The net property
    holds for a power-of-two path count.

    Attributes:
        s0: Initial spot price.
        sigma: Volatility.
        maturity: Horizon in years.
        n_steps: Number of rebalancing intervals.
        mu: Drift under the simulation measure.
        dtype: Floating dtype of the generated paths.
        device: Device on which paths are generated.
        sampler: ``pseudo`` for independent normals, ``sobol`` for
            scrambled Sobol points through the Brownian bridge
            construction.
    """

    s0: float
    sigma: float
    maturity: float
    n_steps: int
    mu: float = 0.0
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    sampler: Literal["pseudo", "sobol"] = "pseudo"

    def __post_init__(self) -> None:
        """Rejects field values outside the documented domain."""
        if self.s0 <= 0.0:
            msg = f"s0 must be positive, got {self.s0}"
            raise ValueError(msg)
        if self.sigma < 0.0:
            msg = f"sigma must be non-negative, got {self.sigma}"
            raise ValueError(msg)
        if self.maturity <= 0.0:
            msg = f"maturity must be positive, got {self.maturity}"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = f"n_steps must be at least 1, got {self.n_steps}"
            raise ValueError(msg)
        if self.sampler not in ("pseudo", "sobol"):
            msg = f"sampler must be 'pseudo' or 'sobol', got {self.sampler!r}"
            raise ValueError(msg)

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates GBM market paths.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state whose spot grid satisfies ``spot[0] == s0``.
        """
        dt = self.maturity / self.n_steps
        if self.sampler == "sobol":
            log_returns = self._sobol_log_returns(n_paths, noise, dt)
        else:
            generator = noise.torch_generator(self.device) if noise is not None else None
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * dt**0.5
            z = torch.randn(
                (self.n_steps, n_paths),
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )
            log_returns = torch.cumsum(drift + diffusion * z, dim=0)
        out = log_returns.new_empty((self.n_steps + 1, n_paths))
        out[0] = 0.0
        out[1:] = log_returns
        return MarketState(spot=out.exp_().mul_(self.s0))

    def _sobol_log_returns(self, n_paths: int, noise: NoiseSpec | None, dt: float) -> torch.Tensor:
        seed = (
            noise.torch_generator().initial_seed()
            if noise is not None
            else int(torch.randint(2**62, ()))
        )
        engine = torch.quasirandom.SobolEngine(self.n_steps, scramble=True, seed=seed)
        uniforms = engine.draw(n_paths, dtype=torch.float64)
        normals = torch.special.ndtri(uniforms.add_(2.0 ** -(engine.MAXBIT + 1)))
        times = torch.arange(1, self.n_steps + 1, dtype=torch.float64) * dt
        levels = math.sqrt(dt) * (brownian_bridge_matrix(self.n_steps) @ normals.T)
        log_returns = (self.mu - 0.5 * self.sigma**2) * times[:, None] + self.sigma * levels
        return log_returns.to(dtype=self.dtype, device=self.device)
