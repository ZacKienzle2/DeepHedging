"""Geometric Brownian motion simulator."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GBMSimulator:
    """Exact-in-distribution GBM sampler evolved in log space.

    Log-space evolution keeps the accumulated state additive, bounding the
    floating-point error growth at ``sqrt(n_steps)`` and avoiding the biased
    increment-dropping failure mode of a multiplicative price-space recursion
    in reduced precision.

    Attributes:
        s0: Initial spot price.
        sigma: Volatility.
        maturity: Horizon in years.
        n_steps: Number of rebalancing intervals.
        mu: Drift under the simulation measure.
        dtype: Floating dtype of the generated paths.
        device: Device on which paths are generated.
    """

    s0: float
    sigma: float
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
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")

    def simulate(self, n_paths: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Simulates GBM price paths.

        Args:
            n_paths: Number of independent paths.
            generator: Optional RNG for reproducible draws.

        Returns:
            Price paths of shape ``(n_steps + 1, n_paths)`` with
            ``paths[0] == s0``.
        """
        dt = self.maturity / self.n_steps
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * dt**0.5
        z = torch.randn(
            (self.n_steps, n_paths),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        log_returns = torch.cumsum(drift + diffusion * z, dim=0)
        zero = log_returns.new_zeros((1, n_paths))
        return self.s0 * torch.exp(torch.cat((zero, log_returns), dim=0))
