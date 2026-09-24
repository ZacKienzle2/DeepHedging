"""Black-Scholes implied volatility by Jaeckel's rational method.

Implements "Let's Be Rational" (Jaeckel, 2015). A call quote is reduced to
the normalised price ``beta`` of an out-of-the-money call at log-moneyness
``x <= 0`` through the reciprocal-strike and put-call invariances (eqs.
2.5-2.7). A four-branch initial guess (Section 4), built from rational cubic
interpolation (Delbourgo and Gregory, eq. 4.10) with nonlinear transforms
that are asymptotically exact at both ends of the price range, is refined by
two third-order Householder steps (eqs. 5.5-5.7) on a three-branch objective
(eq. 5.1). Jaeckel shows that two steps reach the attainable float64
precision for every input.

The attainable precision is set by how accurately the normalised Black
function is evaluated (Jaeckel's Section 6), so the objective reuses
:func:`~deephedging.evaluation.black_scholes.normalised_black`, which
evaluates it in his four regimes and returns its logarithm and vega ratio
without underflow for the lower objective branch.

Prices outside the no-arbitrage band invert to NaN rather than a silently
wrong value.
"""

import math

import torch

from deephedging.evaluation.black_scholes import normal_cdf, normalised_black

_SQRT_THREE = math.sqrt(3.0)
_SQRT_TWO = math.sqrt(2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
_EPS = torch.finfo(torch.float64).eps


def _black_complement(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    h, t = x / s, 0.5 * s
    return torch.exp(0.5 * x) * normal_cdf(-h - t) + torch.exp(-0.5 * x) * normal_cdf(
        h - t
    )


def _vega(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * ((x / s) ** 2 + (0.5 * s) ** 2)) / _SQRT_TWO_PI


def _rational_cubic(
    point: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    value_left: torch.Tensor,
    value_right: torch.Tensor,
    slope_left: torch.Tensor,
    slope_right: torch.Tensor,
    control: torch.Tensor,
) -> torch.Tensor:
    width = right - left
    s = (point - left) / width
    o = 1.0 - s
    numerator = (
        value_right * s**3
        + (control * value_right - width * slope_right) * s**2 * o
        + (control * value_left + width * slope_left) * s * o**2
        + value_left * o**3
    )
    return numerator / (1.0 + (control - 3.0) * s * o)


def _control(
    left: torch.Tensor,
    right: torch.Tensor,
    value_left: torch.Tensor,
    value_right: torch.Tensor,
    slope_left: torch.Tensor,
    slope_right: torch.Tensor,
    curvature: torch.Tensor,
    at_right: bool,
) -> torch.Tensor:
    width = right - left
    secant = (value_right - value_left) / width
    numerator = 0.5 * width * curvature + (slope_right - slope_left)
    denominator = (slope_right - secant) if at_right else (secant - slope_left)
    control = numerator / denominator
    monotone = (slope_left + slope_right) / secant
    return torch.clamp(
        torch.maximum(control, monotone), min=-1.0 + _EPS, max=1.0 / _EPS
    )


def _initial_guess(
    x: torch.Tensor, beta: torch.Tensor, b_max: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_c = torch.sqrt(2.0 * x.abs())
    b_c, _, _ = normalised_black(x, s_c)
    vega_c = _vega(x, s_c)
    s_l = s_c - b_c / vega_c
    s_u = s_c + (b_max - b_c) / vega_c
    b_l, _, _ = normalised_black(x, s_l)
    b_u, _, _ = normalised_black(x, s_u)
    vega_l, vega_u = _vega(x, s_l), _vega(x, s_u)
    zero = torch.zeros_like(x)

    centre_left = _rational_cubic(
        beta,
        b_l,
        b_c,
        s_l,
        s_c,
        1.0 / vega_l,
        1.0 / vega_c,
        _control(b_l, b_c, s_l, s_c, 1.0 / vega_l, 1.0 / vega_c, zero, at_right=True),
    )
    centre_right = _rational_cubic(
        beta,
        b_c,
        b_u,
        s_c,
        s_u,
        1.0 / vega_c,
        1.0 / vega_u,
        _control(b_c, b_u, s_c, s_u, 1.0 / vega_c, 1.0 / vega_u, zero, at_right=False),
    )

    f_u = normal_cdf(-0.5 * s_u)
    slope_u = -0.5 * torch.exp(0.5 * (x / s_u) ** 2)
    curvature_u = (
        math.sqrt(0.5 * math.pi)
        * (x * x / s_u**3)
        * torch.exp((x / s_u) ** 2 + s_u**2 / 8.0)
    )
    half = torch.full_like(x, -0.5)
    upper_value = _rational_cubic(
        beta,
        b_u,
        b_max,
        f_u,
        zero,
        slope_u,
        half,
        _control(b_u, b_max, f_u, zero, slope_u, half, curvature_u, at_right=False),
    )
    upper = -2.0 * torch.special.ndtri(upper_value.clamp(min=0.0, max=0.5))

    z = -x.abs() / (_SQRT_THREE * s_l)
    phi_z = normal_cdf(z)
    f_l = 2.0 * math.pi * x.abs() / (3.0 * _SQRT_THREE) * phi_z**3
    slope_l = 2.0 * math.pi * z**2 * phi_z**2 * torch.exp(z**2 + s_l**2 / 8.0)
    curvature_l = (
        (math.pi / 6.0)
        * (z**2 / s_l**3)
        * phi_z
        * torch.exp(2.0 * z**2 + s_l**2 / 4.0)
        * (
            8.0 * _SQRT_THREE * s_l * x.abs()
            + (3.0 * s_l**2 * (s_l**2 - 8.0) - 8.0 * x**2)
            * (0.5 * _SQRT_TWO_PI * torch.special.erfcx(-z / _SQRT_TWO))
        )
    )
    one = torch.ones_like(x)
    lower_value = _rational_cubic(
        beta,
        zero,
        b_l,
        zero,
        f_l,
        one,
        slope_l,
        _control(zero, b_l, zero, f_l, one, slope_l, curvature_l, at_right=True),
    )
    cube = _SQRT_THREE * torch.pow(
        lower_value.clamp(min=0.0) / (2.0 * math.pi * x.abs()), 1.0 / 3.0
    )
    lower = (x / (_SQRT_THREE * torch.special.ndtri(cube.clamp(max=0.5)))).abs()

    guess = torch.where(
        beta < b_l,
        lower,
        torch.where(
            beta <= b_c, centre_left, torch.where(beta <= b_u, centre_right, upper)
        ),
    )
    guess = torch.where(guess.isnan(), s_c, guess)
    return torch.maximum(guess, _EPS * s_c).clamp(max=1.0 / _EPS), b_l, b_u


def _householder_step(
    x: torch.Tensor,
    s: torch.Tensor,
    beta: torch.Tensor,
    b_max: torch.Tensor,
    lower_branch: torch.Tensor,
    upper_branch: torch.Tensor,
) -> torch.Tensor:
    price, log_price, q = normalised_black(x, s)
    vega = _vega(x, s)
    second = x * x / s**3 - 0.25 * s
    third = second**2 - 3.0 * x * x / s**4 - 0.25

    newton = -(price - beta) / vega
    gamma, delta = second, third

    first_log = q
    second_log = q * second - q**2
    third_log = q * third - 3.0 * q**2 * second + 2.0 * q**3
    g = 1.0 / log_price - 1.0 / torch.log(beta)
    g1 = -first_log / log_price**2
    g2 = -second_log / log_price**2 + 2.0 * first_log**2 / log_price**3
    g3 = (
        -third_log / log_price**2
        + 6.0 * first_log * second_log / log_price**3
        - 6.0 * first_log**3 / log_price**4
    )
    newton = torch.where(lower_branch, -g / g1, newton)
    gamma = torch.where(lower_branch, g2 / g1, gamma)
    delta = torch.where(lower_branch, g3 / g1, delta)

    complement = _black_complement(x, s)
    p = vega / complement
    g = torch.log(b_max - beta) - torch.log(complement)
    g2 = p * second + p**2
    g3 = p * third + 3.0 * p**2 * second + 2.0 * p**3
    newton = torch.where(upper_branch, -g / p, newton)
    gamma = torch.where(upper_branch, g2 / p, gamma)
    delta = torch.where(upper_branch, g3 / p, delta)

    step = (
        newton
        * (1.0 + 0.5 * gamma * newton)
        / (1.0 + newton * (gamma + delta * newton / 6.0))
    )
    updated = s + step
    return torch.where(updated > 0.0, updated, 0.5 * s)


def implied_vol(
    prices: torch.Tensor,
    spot: float,
    strikes: torch.Tensor,
    tau: float,
    n_iterations: int = 2,
) -> torch.Tensor:
    """Inverts call prices to Black-Scholes implied volatilities.

    Args:
        prices: Call prices of shape ``(n_strikes,)`` in float64.
        spot: Spot price, which is the forward under the zero rate.
        strikes: Strike grid of shape ``(n_strikes,)`` in float64.
        tau: Time to maturity in years.
        n_iterations: Third-order Householder steps after the initial
            guess; Jaeckel's two reach float64 precision.

    Returns:
        Implied volatilities of shape ``(n_strikes,)``; NaN where the price
        is at or outside the no-arbitrage band ``(intrinsic, spot)``, or
        where an in-the-money quote's time value is within two units in the
        last place of the quote, so that rounding alone decides it.

    Raises:
        ValueError: If ``tau`` is not positive.
    """
    if tau <= 0.0:
        msg = f"tau must be positive, got {tau}"
        raise ValueError(msg)
    log_moneyness = torch.log(spot / strikes)
    intrinsic = 2.0 * torch.sinh(0.5 * log_moneyness).clamp(min=0.0)
    normalised = prices / torch.sqrt(spot * strikes)
    beta = normalised - intrinsic
    x = -log_moneyness.abs()
    b_max = torch.exp(0.5 * x)
    valid = (beta > 2.0 * _EPS * normalised) & (beta < b_max)
    beta = torch.where(valid, beta, 0.5 * b_max)
    at_the_money = x == 0.0

    s, b_l, b_u = _initial_guess(x, beta, b_max)
    lower_branch = beta < b_l
    upper_branch = beta > torch.maximum(b_u, 0.5 * b_max)
    for _ in range(n_iterations):
        s = _householder_step(x, s, beta, b_max, lower_branch, upper_branch)
    s = torch.where(at_the_money, 2.0 * _SQRT_TWO * torch.special.erfinv(beta), s)
    sigma = s / math.sqrt(tau)
    return torch.where(valid, sigma, torch.full_like(sigma, float("nan")))
