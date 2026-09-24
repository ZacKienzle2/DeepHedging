"""Black-Scholes closed forms and the delta-hedge baseline.

Every function broadcasts over tensors in each market argument, so the
implied-volatility inversion and the calibration weights evaluate a whole
strike grid and a tensor of volatilities through the same formulas as the
scalar baselines. A float volatility is validated here; a tensor one is the
caller's to keep positive, since checking it would read it back to the host.

Prices go through the normalised Black function of Jaeckel (2015, Section 6)
rather than ``S N(d1) - K N(d2)``, whose two terms cancel catastrophically
for out-of-the-money options and whose put by parity cancels against the
forward. The out-of-the-money value is evaluated in four regimes, two by
series derived from ``Y' = 1 + z Y`` for ``Y = Phi / phi``: the Taylor
expansion of ``Y(h + t) - Y(h - t)`` in ``t`` from the recursion
``Y^(n+1) = z Y^(n) + n Y^(n-1)``, and the asymptotic series of Jaeckel's
eq. 6.13 differenced term by term through ``a^-m - b^-m = a^-1 (a^-(m-1) -
b^-(m-1)) + b^-(m-1) (a^-1 - b^-1)``, whose summands share one sign.
"""

import math

import torch

Market = torch.Tensor | float

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_TAU_SMALL = 2.0 * torch.finfo(torch.float64).eps ** (1.0 / 16.0)
_H_LARGE = -10.0
_ASYMPTOTIC_TERMS = 17
_TAYLOR_ORDER = 17


def normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """Standard normal distribution function, accurate in the lower tail.

    Evaluated as ``erfc(-z / sqrt(2)) / 2``, Jaeckel's eq. 6.15. On this
    build ``torch.special.ndtr`` keeps absolute but not relative accuracy
    for negative arguments: against 40-digit references it is off by a
    relative 1e-10 at ``z = -5`` and 2e-2 at ``z = -8``, where the ``erfc``
    form stays near 6e-15.

    Args:
        z: Arguments of any shape.

    Returns:
        ``Phi(z)`` with the shape of ``z``.
    """
    cdf: torch.Tensor = 0.5 * torch.special.erfc(-z / _SQRT_2)
    return cdf


def _mills(z: torch.Tensor) -> torch.Tensor:
    ratio: torch.Tensor = 0.5 * _SQRT_2PI * torch.special.erfcx(-z / _SQRT_2)
    return ratio


def _asymptotic_difference(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    inverse_a, inverse_b = 1.0 / (h + t), 1.0 / (h - t)
    step = -2.0 * t * inverse_a * inverse_b
    difference, power_b = step, inverse_b
    total = -difference
    coefficient = 1.0
    for n in range(1, _ASYMPTOTIC_TERMS + 1):
        for _ in range(2):
            difference = difference * inverse_a + power_b * step
            power_b = power_b * inverse_b
        coefficient *= 2 * n - 1
        total = total - (-1) ** n * coefficient * difference
    return total


def _taylor_difference(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    previous = _mills(h)
    current = 1.0 + h * previous
    total = current * t
    power, factorial = t, 1.0
    for n in range(1, _TAYLOR_ORDER):
        previous, current = current, h * current + n * previous
        power = power * t
        factorial *= n + 1
        if n % 2 == 0:
            total = total + current * power / factorial
    return 2.0 * total


def normalised_black(
    x: torch.Tensor, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalised out-of-the-money Black call, its logarithm and vega ratio.

    Evaluates ``b(x, s) = Phi(h + t) e^(x/2) - Phi(h - t) e^(-x/2)`` with
    ``h = x / s`` and ``t = s / 2`` (Jaeckel's eq. 2.4) in the regimes of his
    Figure 6: the asymptotic series for large negative ``h``, the Taylor
    series for small ``t``, the difference of cumulative normals for large
    ``t``, and eq. 6.10 through ``erfcx`` elsewhere. The ``erfcx`` form runs
    on every element, with the large-``t`` elements moved to a harmless point
    where it would overflow, so autograd meets no infinity in an unselected
    branch; the other regimes overwrite only their own elements, and only
    when they have any.

    Args:
        x: Log-moneyness ``log(F / K)``, non-positive.
        s: Total volatility ``sigma * sqrt(tau)``, positive, of the shape of
            ``x``.

    Returns:
        The price ``b``, its logarithm, and ``b'(s) / b(s)``, the last two
        from eq. 6.10 without forming ``b`` or ``b'``, both of which
        underflow in the deep wing.
    """
    h, t = x / s, 0.5 * s
    large = (h < _H_LARGE) & (t < -h + _H_LARGE + _TAU_SMALL)
    small = ~large & (t < _TAU_SMALL)
    direct = ~large & ~small & (t > 0.85 - h)
    finite_h = torch.where(direct, -1.0, h)
    finite_t = torch.where(direct, 0.5, t)
    difference = _mills(finite_h + finite_t) - _mills(finite_h - finite_t)
    if bool(large.any()):
        difference[large] = _asymptotic_difference(h[large], t[large])
    if bool(small.any()):
        difference[small] = _taylor_difference(h[small], t[small])
    log_vega = -0.5 * (h * h + t * t) - _LOG_SQRT_2PI
    log_price = log_vega + torch.log(difference)
    ratio = 1.0 / difference
    if bool(direct.any()):
        hd, td = h[direct], t[direct]
        exact = torch.ones_like(h)
        exact[direct] = normal_cdf(hd + td) * torch.exp(hd * td) - normal_cdf(
            hd - td
        ) * torch.exp(-hd * td)
        log_price = torch.where(direct, torch.log(exact), log_price)
        ratio = torch.where(direct, torch.exp(log_vega - log_price), ratio)
    return torch.exp(log_price), log_price, ratio


def _black_price(
    spot: Market, strike: Market, sigma: Market, tau: Market, rate: Market, sign: float
) -> torch.Tensor:
    spot_t = torch.as_tensor(spot, dtype=torch.float64)
    tau_t = torch.as_tensor(tau, dtype=torch.float64)
    if isinstance(sigma, float) and sigma <= 0.0:
        msg = f"sigma must be positive, got {sigma}"
        raise ValueError(msg)
    if bool(torch.any(tau_t <= 0.0)):
        msg = "tau must be positive everywhere"
        raise ValueError(msg)
    x = torch.log(spot_t / strike) + rate * tau_t
    moneyness, s = torch.broadcast_tensors(sign * x, sigma * torch.sqrt(tau_t))
    in_the_money = moneyness > 0.0
    time_value, _, _ = normalised_black(
        torch.where(in_the_money, -moneyness, moneyness), s
    )
    intrinsic = torch.where(in_the_money, 2.0 * torch.sinh(0.5 * moneyness), 0.0)
    return (
        torch.sqrt(spot_t * strike)
        * torch.exp(-0.5 * rate * tau_t)
        * (time_value + intrinsic)
    )


def _d1(
    spot: torch.Tensor, strike: Market, sigma: Market, tau: torch.Tensor, rate: Market
) -> torch.Tensor:
    if isinstance(sigma, float) and sigma <= 0.0:
        msg = f"sigma must be positive, got {sigma}"
        raise ValueError(msg)
    if bool(torch.any(tau <= 0.0)):
        msg = "tau must be positive everywhere"
        raise ValueError(msg)
    return (torch.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (
        sigma * torch.sqrt(tau)
    )


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
    return _black_price(spot, strike, sigma, tau, rate, sign=1.0)


def bs_put_price(
    spot: Market,
    strike: float,
    sigma: float,
    tau: Market,
    rate: float = 0.0,
) -> torch.Tensor:
    """Black-Scholes price of a European put.

    The put shares the call's out-of-the-money time value (Jaeckel's
    put-call invariance, eq. 2.6), so it is priced directly rather than by
    parity, which subtracts the forward from the call and cancels for an
    out-of-the-money put.

    Args:
        spot: Spot price; scalar or tensor.
        strike: Strike price.
        sigma: Volatility.
        tau: Time to maturity; scalar or tensor broadcastable with ``spot``.
        rate: Continuously compounded interest rate.

    Returns:
        Put price with the broadcast shape of ``spot`` and ``tau``.
    """
    return _black_price(spot, strike, sigma, tau, rate, sign=-1.0)


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
    delta: torch.Tensor = torch.special.ndtr(_d1(spot_t, strike, sigma, tau_t, rate))
    return delta


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
