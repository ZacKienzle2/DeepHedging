"""Dupire local volatility from a call price surface.

Dupire's formula reads the local variance off the surface as twice the
maturity derivative over the strike convexity, ``2 dC/dT / (K^2 d2C/dK2)``
under a zero rate. Both derivatives come from finite differences on a
dense grid priced by the COS engine, so the construction consumes the
calibrated model directly and no market interpolation sits between
calibration and simulation. The maturity derivative is ``torch.gradient``
with the maturities as coordinates, which uses the exact three-point formula
for non-uniform spacing, since the symmetric central difference loses an order
when the gaps differ; the strike convexity is the second difference
``torch.diff(n=2)`` over the uniform strike step squared. The variance is
clamped to a wide admissible band because finite differences near the wings
divide one small number by another, and an unguarded ratio there turns a tiny
butterfly violation into an absurd volatility.
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

    def vol_rows(self, times: torch.Tensor) -> torch.Tensor:
        """Interpolates the volatility rows at calendar times, linearly in time.

        Args:
            times: Calendar times in years of shape ``(n_times,)`` in float64,
                clamped to the grid range.

        Returns:
            Volatilities over the strike grid, shape ``(n_times, n_strikes)``.
        """
        upper = torch.searchsorted(self.taus, times).clamp(
            min=1, max=self.taus.shape[0] - 1
        )
        lower = upper - 1
        span = self.taus[upper] - self.taus[lower]
        weight = ((times - self.taus[lower]) / span).clamp(min=0.0, max=1.0)
        return torch.lerp(self.vols[lower], self.vols[upper], weight.unsqueeze(-1))


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
        msg = f"need at least three maturities, got {taus.shape[0]}"
        raise ValueError(msg)
    if bool((taus.diff() <= 0.0).any()):
        msg = "taus must be strictly increasing"
        raise ValueError(msg)
    if strikes.shape[0] < 5:
        msg = f"need at least five strikes, got {strikes.shape[0]}"
        raise ValueError(msg)
    spacing = strikes.diff()
    if not torch.allclose(spacing, spacing[0].expand_as(spacing), atol=1e-9):
        msg = "strikes must be uniformly spaced"
        raise ValueError(msg)
    strike_step = float(spacing[0])

    tensors = params.as_tensors()
    prices = torch.stack(
        [price_surface(tensors, s0, strikes, float(tau)) for tau in taus]
    )

    (time_derivative,) = torch.gradient(prices, spacing=(taus,), dim=0)
    convexity = torch.clamp(torch.diff(prices, n=2, dim=1) / strike_step**2, min=1e-7)
    inner_strikes = strikes[1:-1]
    local_variance = 2.0 * time_derivative[:, 1:-1] / (inner_strikes**2 * convexity)
    local_variance = torch.clamp(local_variance, min=_MIN_VARIANCE, max=_MAX_VARIANCE)
    return LocalVolSurface(
        taus=taus, strikes=inner_strikes, vols=torch.sqrt(local_variance)
    )
