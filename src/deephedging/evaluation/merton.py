"""Merton jump-diffusion closed-form call price."""

import math

import torch

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
    truncated where the Poisson weights are far below double round-off.

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
        ValueError: If both volatilities vanish, leaving the zero-jump
            term without a defined Black-Scholes price.
    """
    if sigma <= 0.0 and jump_vol <= 0.0:
        raise ValueError("sigma and jump_vol cannot both be zero")
    mean_jump_size = math.exp(jump_mean + 0.5 * jump_vol**2) - 1.0
    total = torch.zeros((), dtype=torch.float64)
    log_weight_base = -jump_intensity * tau
    for count in range(_SERIES_TERMS):
        log_weight = (
            log_weight_base
            + count * math.log(max(jump_intensity * tau, 1e-300))
            - math.lgamma(count + 1.0)
        )
        sigma_n = math.sqrt(sigma**2 + count * jump_vol**2 / tau)
        rate_n = -jump_intensity * mean_jump_size + count * (jump_mean + 0.5 * jump_vol**2) / tau
        discounted = bs_call_price(spot, strike, sigma_n, tau, rate=rate_n)
        total = total + math.exp(log_weight) * discounted * math.exp(rate_n * tau)
    return total
