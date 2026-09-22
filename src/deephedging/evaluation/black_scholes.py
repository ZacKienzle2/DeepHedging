"""Black-Scholes closed forms and the delta-hedge baseline.

Every function broadcasts over tensors in each market argument, so the
implied-volatility inversion and the calibration weights evaluate a whole
strike grid and a tensor of volatilities through the same formulas as the
scalar baselines. A float volatility is validated here; a tensor one is the
caller's to keep positive, since checking it would read it back to the host.
"""

import math

import torch

Market = torch.Tensor | float

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _d1(
    spot: torch.Tensor, strike: Market, sigma: Market, tau: torch.Tensor, rate: Market
) -> torch.Tensor:
    if isinstance(sigma, float) and sigma <= 0.0:
        msg = f"sigma must be positive, got {sigma}"
        raise ValueError(msg)
    if bool(torch.any(tau <= 0.0)):
        msg = "tau must be positive everywhere"
        raise ValueError(msg)
    return (torch.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))


def bs_call_price(
    spot: Market,
    strike: Market,
    sigma: Market,
    tau: Market,
    rate: Market = 0.0,
) -> torch.Tensor:
    """Black-Scholes price of a European call.

    Args:
        spot: Spot price; scalar or tensor.
        strike: Strike price; scalar or tensor.
        sigma: Volatility; scalar or tensor.
        tau: Time to maturity; scalar or tensor broadcastable with ``spot``.
        rate: Continuously compounded interest rate; scalar or tensor.

    Returns:
        Call price with the broadcast shape of the inputs.
    """
    spot_t = torch.as_tensor(spot, dtype=torch.float64)
    tau_t = torch.as_tensor(tau, dtype=torch.float64)
    d1 = _d1(spot_t, strike, sigma, tau_t, rate)
    d2 = d1 - sigma * torch.sqrt(tau_t)
    discount = torch.exp(-rate * tau_t)
    return spot_t * torch.special.ndtr(d1) - strike * discount * torch.special.ndtr(d2)


def bs_put_price(
    spot: Market,
    strike: float,
    sigma: float,
    tau: Market,
    rate: float = 0.0,
) -> torch.Tensor:
    """Black-Scholes price of a European put via put-call parity.

    Args:
        spot: Spot price; scalar or tensor.
        strike: Strike price.
        sigma: Volatility.
        tau: Time to maturity; scalar or tensor broadcastable with ``spot``.
        rate: Continuously compounded interest rate.

    Returns:
        Put price with the broadcast shape of ``spot`` and ``tau``.
    """
    spot_t = torch.as_tensor(spot, dtype=torch.float64)
    tau_t = torch.as_tensor(tau, dtype=torch.float64)
    call = bs_call_price(spot_t, strike, sigma, tau_t, rate)
    return call - spot_t + strike * torch.exp(-rate * tau_t)


def bs_call_delta(
    spot: Market,
    strike: float,
    sigma: float,
    tau: Market,
    rate: float = 0.0,
) -> torch.Tensor:
    """Black-Scholes delta of a European call.

    Args:
        spot: Spot price; scalar or tensor.
        strike: Strike price.
        sigma: Volatility.
        tau: Time to maturity; scalar or tensor broadcastable with ``spot``.
        rate: Continuously compounded interest rate.

    Returns:
        Call delta ``N(d1)`` with the broadcast shape of the inputs.
    """
    spot_t = torch.as_tensor(spot, dtype=torch.float64)
    tau_t = torch.as_tensor(tau, dtype=torch.float64)
    return torch.special.ndtr(_d1(spot_t, strike, sigma, tau_t, rate))


def bs_call_vega(
    spot: Market,
    strike: Market,
    sigma: Market,
    tau: Market,
    rate: float = 0.0,
) -> torch.Tensor:
    """Black-Scholes vega of a European call.

    Args:
        spot: Spot price; scalar or tensor.
        strike: Strike price; scalar or tensor.
        sigma: Volatility; scalar or tensor.
        tau: Time to maturity; scalar or tensor broadcastable with ``spot``.
        rate: Continuously compounded interest rate.

    Returns:
        Vega with the broadcast shape of the inputs.
    """
    spot_t = torch.as_tensor(spot, dtype=torch.float64)
    tau_t = torch.as_tensor(tau, dtype=torch.float64)
    d1 = _d1(spot_t, strike, sigma, tau_t, rate)
    return spot_t * torch.exp(-0.5 * d1**2) / _SQRT_2PI * torch.sqrt(tau_t)


def delta_hedge_positions(
    paths: torch.Tensor,
    strike: float,
    sigma: float,
    maturity: float,
    rate: float = 0.0,
) -> torch.Tensor:
    """Black-Scholes delta-hedge positions along simulated paths.

    The continuously-rebalanced delta hedge replicates exactly under GBM
    regardless of drift; under discrete rebalancing its hedging error shrinks
    like ``1 / sqrt(n_steps)``, making it the canonical baseline for any
    learned policy in the frictionless limit.

    Args:
        paths: Price paths of shape ``(n_steps + 1, n_paths)``.
        strike: Strike price of the hedged call.
        sigma: Volatility used for the deltas.
        maturity: Horizon in years matching the path grid.
        rate: Continuously compounded interest rate.

    Returns:
        Positions of shape ``(n_steps, n_paths)``; row ``t`` is the delta
        held over ``[t, t + 1)``.
    """
    n_steps = paths.shape[0] - 1
    dt = maturity / n_steps
    times = torch.arange(n_steps, dtype=torch.float64, device=paths.device) * dt
    tau = (maturity - times).unsqueeze(1)
    deltas = bs_call_delta(paths[:-1].to(torch.float64), strike, sigma, tau, rate)
    return deltas.to(paths.dtype)
