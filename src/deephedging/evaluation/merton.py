"""Merton jump-diffusion closed-form call price."""

import math

import torch
from torch.distributions import Poisson

from deephedging.evaluation.black_scholes import bs_call_price

_SERIES_TERMS = 40


def merton_call_price(
    spot: float,
    strike: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    tau: float,
) -> torch.Tensor:
    """Prices a European call under Merton jump-diffusion, zero rate.

    Conditioning on the jump count makes the terminal price lognormal,
    so the price is the plain-Poisson-weighted series of conditional
    Black-type prices with count-adjusted volatility and forward. Each
    term is undiscounted because this codebase prices in a zero-rate
    economy; the alternative textbook form reweights by the jump-size
    factor instead, and mixing the two double-counts it. The series is
    truncated where the Poisson weights are far below double round-off, and
    every term is priced by one broadcast Black-Scholes call weighted by
    ``torch.distributions.Poisson``.

    Args:
        spot: Spot price.
        strike: Strike price.
        sigma: Diffusive volatility.
        jump_intensity: Expected jumps per year.
        jump_mean: Mean of the jump mark exponent.
        jump_vol: Standard deviation of the jump mark exponent.
        tau: Time to maturity in years.

    Returns:
        Scalar call price in float64.

    Raises:
        ValueError: If ``sigma`` is not positive, leaving the zero-jump
            term without a defined Black-Scholes price.
    """
    if sigma <= 0.0:
        msg = f"sigma must be positive, got {sigma}"
        raise ValueError(msg)
    counts = torch.arange(_SERIES_TERMS, dtype=torch.float64)
    expected_jumps = torch.tensor(jump_intensity * tau, dtype=torch.float64)
    weights = Poisson(expected_jumps).log_prob(counts).exp()
    mean_jump_size = math.exp(jump_mean + 0.5 * jump_vol**2) - 1.0
    sigma_n = torch.sqrt(sigma**2 + counts * jump_vol**2 / tau)
    rate_n = -jump_intensity * mean_jump_size + counts * (jump_mean + 0.5 * jump_vol**2) / tau
    prices = bs_call_price(spot, strike, sigma_n, tau, rate=rate_n) * torch.exp(rate_n * tau)
    return (weights * prices).sum()
