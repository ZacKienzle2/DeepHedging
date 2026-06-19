"""Exponentially tilted simulation for tail importance sampling.

At high confidence levels only the worst ``(1 - alpha)`` fraction of
paths carries CVaR gradient, so batches must scale like the reciprocal
tail mass. Tilting the Gaussian driver shifts sampling toward the
loss-bearing region while per-path likelihood ratios keep every
expectation unbiased under the original measure. For increments drawn
with mean ``tilt`` the ratio is ``exp(-tilt * sum(z) + n * tilt^2 / 2)``
with ``z`` the tilted draws. The log ratio ships as the ``log_weight``
state channel and the risk measures consume it; nothing about the
episode engine changes. The tilt direction is book-dependent, positive
lifting the drift, which loads the tail of a short-call book.
"""

from dataclasses import dataclass

import torch

from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState

LOG_WEIGHT_CHANNEL = "log_weight"


@dataclass(frozen=True)
class TiltedGBMSimulator:
    """GBM sampled under an exponentially tilted driver.

    Attributes:
        s0: Initial spot price.
        sigma: Volatility.
        maturity: Horizon in years.
        n_steps: Number of rebalancing intervals.
        tilt: Per-step mean shift of the standard normal driver;
            positive shifts the drift upward.
        mu: Drift under the original measure.
        dtype: Floating dtype of the generated paths.
        device: Device on which paths are generated.
    """

    s0: float
    sigma: float
    maturity: float
    n_steps: int
    tilt: float
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
        if abs(self.tilt) > 1.0:
            raise ValueError(f"tilt beyond one standard deviation per step, got {self.tilt}")

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates tilted GBM paths with their likelihood ratios.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state with ``spot[0] == s0`` and the per-path log
            likelihood ratio in the ``log_weight`` channel.
        """
        generator = noise.torch_generator(self.device) if noise is not None else None
        dt = self.maturity / self.n_steps
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * dt**0.5
        z = torch.randn(
            (self.n_steps, n_paths),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        tilted = z + self.tilt
        log_returns = torch.cumsum(drift + diffusion * tilted, dim=0)
        out = log_returns.new_empty((self.n_steps + 1, n_paths))
        out[0] = 0.0
        out[1:] = log_returns
        log_weight = -self.tilt * tilted.sum(dim=0) + self.n_steps * self.tilt**2 / 2.0
        return MarketState(spot=out.exp_().mul_(self.s0), aux={LOG_WEIGHT_CHANNEL: log_weight})
