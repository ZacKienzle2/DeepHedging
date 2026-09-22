"""Vectorised Black-Scholes implied volatility.

Inversion brackets every quote with a fixed count of vectorised
bisection steps, which is globally convergent because the call price is
monotone in volatility, then polishes with Halley iterations whose
cubic convergence reaches round-off from the bracketed start. Both
stages run a fixed iteration count with no per-element branch, so the
routine keeps a static shape. Newton-style methods alone stall on deep
wings where vega underflows, which is why the bracket comes first.
Prices outside the no-arbitrage band, and any quote whose final
residual has not converged, invert to NaN rather than returning a
silently wrong value.
"""

import math

import torch

from deephedging.evaluation.black_scholes import bs_call_price, bs_call_vega


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
        price violates the no-arbitrage band or where the iteration did
        not converge, so a stalled deep-wing inversion can never return
        a silently wrong finite value.

    Raises:
        ValueError: If ``tau`` is not positive.
    """
    if tau <= 0.0:
        msg = f"tau must be positive, got {tau}"
        raise ValueError(msg)
    intrinsic = torch.clamp(spot - strikes, min=0.0)
    valid = (prices > intrinsic) & (prices < spot)
    sqrt_tau = math.sqrt(tau)
    low = torch.full_like(prices, 1e-4)
    high = torch.full_like(prices, 5.0)
    for _ in range(25):
        mid = 0.5 * (low + high)
        above = bs_call_price(spot, strikes, mid, tau) > prices
        high = torch.where(above, mid, high)
        low = torch.where(above, low, mid)
    sigma = 0.5 * (low + high)
    for _ in range(n_iterations):
        residual = bs_call_price(spot, strikes, sigma, tau) - prices
        ratio = residual / torch.clamp(bs_call_vega(spot, strikes, sigma, tau), min=1e-12)
        d1 = (torch.log(spot / strikes) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
        vomma_over_vega = d1 * (d1 - sigma * sqrt_tau) / sigma
        denominator = 1.0 - 0.5 * ratio * vomma_over_vega
        guarded = torch.where(
            denominator.abs() < 1e-8, torch.full_like(denominator, 1.0), denominator
        )
        sigma = torch.clamp(sigma - ratio / guarded, min=1e-4, max=5.0)
    final_residual = (bs_call_price(spot, strikes, sigma, tau) - prices).abs()
    converged = final_residual < 1e-6 * spot
    keep = valid & converged
    return torch.where(keep, sigma, torch.full_like(sigma, float("nan")))
