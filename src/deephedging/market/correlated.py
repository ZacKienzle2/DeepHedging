"""Correlated multi-asset geometric Brownian motion."""

from dataclasses import dataclass

import torch

from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState


@dataclass(frozen=True)
class CorrelatedGBMSimulator:
    """Exact-in-distribution multi-asset GBM with a constant correlation.

    Each asset evolves in log space with its own volatility; Brownian
    drivers are correlated through the Cholesky factor of the supplied
    correlation matrix. The spot grid carries the trailing asset axis
    ``(n_steps + 1, n_paths, n_assets)`` per the MarketState convention,
    and positions contract that axis in the gains reduction.

    Attributes:
        s0: Initial spot price shared by every asset.
        sigmas: Per-asset volatilities.
        correlation: Correlation matrix as nested tuples; must be
            symmetric with unit diagonal and positive definite.
        maturity: Horizon in years.
        n_steps: Number of rebalancing intervals.
        mu: Drift shared by every asset under the simulation measure.
        dtype: Floating dtype of the generated paths.
        device: Device on which paths are generated.
    """

    s0: float
    sigmas: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    maturity: float
    n_steps: int
    mu: float = 0.0
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.s0 <= 0.0:
            raise ValueError(f"s0 must be positive, got {self.s0}")
        if not self.sigmas:
            raise ValueError("sigmas must contain at least one asset")
        if any(sigma < 0.0 for sigma in self.sigmas):
            raise ValueError("every volatility must be non-negative")
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")
        matrix = torch.tensor(self.correlation, dtype=torch.float64)
        n_assets = len(self.sigmas)
        if matrix.shape != (n_assets, n_assets):
            raise ValueError(
                f"correlation must be {n_assets}x{n_assets}, got {tuple(matrix.shape)}"
            )
        if not torch.allclose(matrix, matrix.T, atol=1e-12):
            raise ValueError("correlation must be symmetric")
        if not torch.allclose(matrix.diagonal(), torch.ones(n_assets, dtype=torch.float64)):
            raise ValueError("correlation must have a unit diagonal")
        try:
            torch.linalg.cholesky(matrix)
        except RuntimeError as error:
            raise ValueError("correlation must be positive definite") from error

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return len(self.sigmas)

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates correlated GBM market paths.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state whose spot grid has shape
            ``(n_steps + 1, n_paths, n_assets)`` with ``spot[0] == s0``.
        """
        generator = noise.torch_generator(self.device) if noise is not None else None
        n_assets = self.n_assets
        dt = self.maturity / self.n_steps
        sqrt_dt = dt**0.5
        cholesky = (
            torch.linalg.cholesky(torch.tensor(self.correlation, dtype=torch.float64))
            .to(self.dtype)
            .to(self.device)
        )
        sigmas = torch.tensor(self.sigmas, dtype=self.dtype, device=self.device)
        drift = (self.mu - 0.5 * sigmas**2) * dt
        z = torch.randn(
            (self.n_steps, n_paths, n_assets),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        increments = drift + sqrt_dt * sigmas * (z @ cholesky.T)
        log_returns = torch.cumsum(increments, dim=0)
        out = log_returns.new_empty((self.n_steps + 1, n_paths, n_assets))
        out[0] = 0.0
        out[1:] = log_returns
        return MarketState(spot=out.exp_().mul_(self.s0))
