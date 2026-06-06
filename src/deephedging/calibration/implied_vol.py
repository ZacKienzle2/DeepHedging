"""Vectorised Black-Scholes implied volatility.

Inversion runs a fixed number of Halley iterations from a
Brenner-Subrahmanyam seed, vectorised over the whole quote grid with no
per-element convergence branch, so the routine keeps a static shape.
The cubic convergence of Halley reaches round-off in a handful of
iterations across the moneyness range where vanilla quotes live, and a
test pins the residual rather than a runtime loop checking it. Prices
outside the no-arbitrage band invert to NaN instead of letting the
iteration diverge.
"""

import math

import torch

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def implied_vol(
    prices: torch.Tensor,
    spot: float,
    strikes: torch.Tensor,
    tau: float,
    n_iterations: int = 12,
) -> torch.Tensor:
    """Inverts call prices to Black-Scholes implied volatilities.

    Args:
        prices: Call prices of shape ``(n_strikes,)`` in float64.
        spot: Spot price.
        strikes: Strike grid of shape ``(n_strikes,)`` in float64.
        tau: Time to maturity in years.
        n_iterations: Fixed Halley iteration count.

    Returns:
        Implied volatilities of shape ``(n_strikes,)``; NaN where the
        price violates the no-arbitrage band.
    """
    intrinsic = torch.clamp(spot - strikes, min=0.0)
    valid = (prices > intrinsic) & (prices < spot)
    sqrt_tau = math.sqrt(tau)
    sigma = torch.clamp(prices * _SQRT_2PI / (spot * sqrt_tau), min=1e-3, max=4.0)
    for _ in range(n_iterations):
        model = _bs_price_grid(spot, strikes, sigma, tau)
        d1 = (torch.log(spot / strikes) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        density = torch.exp(-0.5 * d1**2) / _SQRT_2PI
        vega = spot * density * sqrt_tau
        residual = model - prices
        ratio = residual / torch.clamp(vega, min=1e-12)
        halley = ratio / (1.0 - 0.5 * ratio * (d1 * d2 / sigma))
        sigma = torch.clamp(sigma - halley, min=1e-4, max=5.0)
    return torch.where(valid, sigma, torch.full_like(sigma, float("nan")))


def _bs_price_grid(
    spot: float, strikes: torch.Tensor, sigma: torch.Tensor, tau: float
) -> torch.Tensor:
    sqrt_tau = math.sqrt(tau)
    d1 = (torch.log(spot / strikes) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return spot * torch.special.ndtr(d1) - strikes * torch.special.ndtr(d2)
