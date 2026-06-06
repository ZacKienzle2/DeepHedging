"""Dupire local volatility from a call price surface.

Dupire's formula reads the local variance off the surface as twice the
maturity derivative over the strike convexity, ``2 dC/dT / (K^2 d2C/dK2)``
under a zero rate. Both derivatives come from finite differences on a
dense grid priced by the COS engine, so the construction consumes the
calibrated model directly and no market interpolation sits between
calibration and simulation. The maturity derivative uses the exact
three-point formula for non-uniform spacing because the symmetric
central difference loses an order when the gaps differ; the strike
convexity is the standard second-order stencil on the uniform strike
grid. The variance is clamped to a wide admissible band because finite
differences near the wings divide one small number by another, and an
unguarded ratio there turns a tiny butterfly violation into an absurd
volatility.
"""

from dataclasses import dataclass

import torch

from deephedging.calibration.calibrate import price_surface
from deephedging.calibration.heston_cf import HestonParams

_MIN_VARIANCE = 1e-4
_MAX_VARIANCE = 4.0


@dataclass(frozen=True)
class LocalVolSurface:
    """Local volatility on a rectangular maturity-strike grid.

    Attributes:
        taus: Maturities of shape ``(n_maturities,)`` in float64.
        strikes: Strikes of shape ``(n_strikes,)`` in float64.
        vols: Local volatilities of shape ``(n_maturities, n_strikes)``.
    """

    taus: torch.Tensor
    strikes: torch.Tensor
    vols: torch.Tensor

    def vol_row(self, tau: float) -> torch.Tensor:
        """Interpolates the volatility row at a calendar time.

        Args:
            tau: Calendar time in years, clamped to the grid range.

        Returns:
            Volatilities over the strike grid, shape ``(n_strikes,)``.
        """
        position = float(
            torch.clamp(
                torch.searchsorted(self.taus, torch.tensor(tau, dtype=torch.float64)),
                min=1,
                max=self.taus.shape[0] - 1,
            )
        )
        upper = int(position)
        lower = upper - 1
        span = float(self.taus[upper] - self.taus[lower])
        weight = min(max((tau - float(self.taus[lower])) / span, 0.0), 1.0)
        return (1.0 - weight) * self.vols[lower] + weight * self.vols[upper]


def dupire_surface(
    params: HestonParams,
    s0: float,
    strikes: torch.Tensor,
    taus: torch.Tensor,
) -> LocalVolSurface:
    """Builds the local volatility surface from COS call prices.

    Args:
        params: Calibrated Heston parameters generating the surface.
        s0: Spot price.
        strikes: Uniform strike grid of shape ``(n_strikes,)`` in float64.
        taus: Increasing maturities of shape ``(n_maturities,)`` in
            float64; at least three.

    Returns:
        The local volatility surface on the interior strike grid.

    Raises:
        ValueError: If the grids are too small or the strikes non-uniform.
    """
    if taus.shape[0] < 3:
        raise ValueError(f"need at least three maturities, got {taus.shape[0]}")
    if strikes.shape[0] < 5:
        raise ValueError(f"need at least five strikes, got {strikes.shape[0]}")
    spacing = strikes.diff()
    if not torch.allclose(spacing, spacing[0].expand_as(spacing), atol=1e-9):
        raise ValueError("strikes must be uniformly spaced")
    strike_step = float(spacing[0])

    tensors = params.as_tensors()
    prices = torch.stack([price_surface(tensors, s0, strikes, float(tau)) for tau in taus])

    time_derivative = torch.empty_like(prices)
    time_derivative[0] = (prices[1] - prices[0]) / float(taus[1] - taus[0])
    time_derivative[-1] = (prices[-1] - prices[-2]) / float(taus[-1] - taus[-2])
    backward_span = (taus[1:-1] - taus[:-2]).unsqueeze(1)
    forward_span = (taus[2:] - taus[1:-1]).unsqueeze(1)
    total_span = backward_span + forward_span
    time_derivative[1:-1] = (
        -forward_span / (backward_span * total_span) * prices[:-2]
        + (forward_span - backward_span) / (backward_span * forward_span) * prices[1:-1]
        + backward_span / (forward_span * total_span) * prices[2:]
    )

    convexity = (prices[:, 2:] - 2.0 * prices[:, 1:-1] + prices[:, :-2]) / strike_step**2
    convexity = torch.clamp(convexity, min=1e-7)
    inner_strikes = strikes[1:-1]
    local_variance = 2.0 * time_derivative[:, 1:-1] / (inner_strikes**2 * convexity)
    local_variance = torch.clamp(local_variance, min=_MIN_VARIANCE, max=_MAX_VARIANCE)
    return LocalVolSurface(taus=taus, strikes=inner_strikes, vols=torch.sqrt(local_variance))
