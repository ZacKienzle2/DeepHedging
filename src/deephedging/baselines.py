"""Implementations that faster ones replaced, kept as equivalence oracles.

WORKFLOW.md keeps the obvious version of a function when it is rewritten, and
``nox -s generate -- deephedging.baselines.<old> deephedging.<module>.<new>``
writes the Hypothesis test that compares the two over generated inputs. Each
function here is the version its replacement's commit measured against, with
the loop that the rewrite removed. Where the replaced version was itself wrong,
its oracle is the defining formula in 400-digit mpmath arithmetic instead,
which no cancellation reaches. None of them is exported or used at run time.
"""

import functools
import math
from typing import Any, cast

import mpmath
import torch
from torch.utils.checkpoint import checkpoint

from deephedging.evaluation.black_scholes import bs_call_price
from deephedging.evaluation.inference import BootstrapInterval, MetricFunction
from deephedging.features import DefaultFeatures, FeatureMap
from deephedging.frictions.base import CostModel
from deephedging.instruments.base import Payoff
from deephedging.market.state import MarketState
from deephedging.policies.base import HedgePolicy


def hedge_pnl_settled_per_date(
    state: MarketState,
    policy: HedgePolicy,
    payoff: Payoff,
    cost_model: CostModel,
    premium: float = 0.0,
    liquidate_terminal: bool = False,
    checkpoint_steps: bool = False,
    feature_map: FeatureMap | None = None,
) -> torch.Tensor:
    """Settles gains and costs inside the loop over dates.

    Args:
        state: Simulated market state.
        policy: Hedging policy network.
        payoff: Liability payoff, charged at maturity.
        cost_model: Transaction cost model.
        premium: Premium received for the liability at inception.
        liquidate_terminal: Whether to charge the cost of closing the final
            position at the terminal price.
        checkpoint_steps: Whether to gradient-checkpoint each policy call.
        feature_map: Observation builder; defaults to the baseline features.

    Returns:
        PnL per path of shape ``(n_paths,)``.
    """
    paths = state.spot
    n_steps = state.n_steps

    def per_path(values: torch.Tensor) -> torch.Tensor:
        return values.sum(dim=-1) if values.dim() == 2 else values

    features_of = feature_map if feature_map is not None else DefaultFeatures()
    taus = torch.arange(n_steps, 0, -1, dtype=paths.dtype, device=paths.device) / n_steps
    position = paths.new_zeros(paths.shape[1:])
    pnl = paths.new_zeros(state.n_paths) + premium
    hidden: torch.Tensor | None = None
    use_checkpoint = checkpoint_steps and torch.is_grad_enabled()
    for t in range(n_steps):
        spot = paths[t]
        features = features_of(state, t, taus[t], position)
        if use_checkpoint:
            output = checkpoint(policy, features, hidden, use_reentrant=False)
            new_position, hidden = cast("tuple[torch.Tensor, torch.Tensor | None]", output)
        else:
            new_position, hidden = policy(features, hidden)
        new_position = new_position.to(paths.dtype)
        pnl = (
            pnl
            + per_path(new_position * (paths[t + 1] - spot))
            - per_path(cost_model(new_position - position, spot))
        )
        position = new_position
    if liquidate_terminal:
        pnl = pnl - per_path(cost_model(position, paths[-1]))
    return pnl - payoff(paths)


def merton_call_price_per_term(
    spot: float,
    strike: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    tau: float,
) -> torch.Tensor:
    """Sums the Merton series one Black-Scholes call per jump count.

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
    """
    mean_jump_size = math.exp(jump_mean + 0.5 * jump_vol**2) - 1.0
    total = torch.zeros((), dtype=torch.float64)
    log_weight_base = -jump_intensity * tau
    for count in range(40):
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


def bootstrap_metric_per_resample(
    pnl: torch.Tensor,
    metric: MetricFunction,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Draws and scores one bootstrap resample per loop iteration.

    Args:
        pnl: PnL per path of shape ``(n_paths,)``.
        metric: Map from a PnL sample to a scalar tensor.
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided coverage in ``(0, 1)``.
        seed: Seed for the resampling generator.

    Returns:
        The point estimate and its confidence bounds.
    """
    n_paths = pnl.shape[0]
    generator = torch.Generator(device=pnl.device).manual_seed(seed)
    estimates = pnl.new_empty(n_resamples)
    for index in range(n_resamples):
        draw = torch.randint(n_paths, (n_paths,), generator=generator, device=pnl.device)
        estimates[index] = metric(pnl[draw])
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(metric(pnl)),
        low=float(torch.quantile(estimates, tail)),
        high=float(torch.quantile(estimates, 1.0 - tail)),
        confidence=confidence,
    )


def _call_by_mpmath(spot: Any, strike: Any, sigma: Any, tau: Any, rate: Any, digits: int) -> Any:
    with mpmath.workdps(digits):
        s, k, v, t, r = (mpmath.mpf(value) for value in (spot, strike, sigma, tau, rate))
        d1 = (mpmath.log(s / k) + (r + v**2 / 2) * t) / (v * mpmath.sqrt(t))
        d2 = d1 - v * mpmath.sqrt(t)
        return s * mpmath.ncdf(d1) - k * mpmath.exp(-r * t) * mpmath.ncdf(d2)


def bs_call_price_by_mpmath(
    spot: float, strike: float, sigma: float, tau: float, rate: float = 0.0
) -> torch.Tensor:
    """Prices a call by ``S N(d1) - K exp(-r tau) N(d2)`` in 400 digits.

    Four hundred digits leave the subtraction exact for every price down to
    the float64 underflow threshold.

    Args:
        spot: Spot price.
        strike: Strike price.
        sigma: Volatility.
        tau: Time to maturity.
        rate: Continuously compounded interest rate.

    Returns:
        Scalar float64 call price, correctly rounded from the exact value.
    """
    value = _call_by_mpmath(spot, strike, sigma, tau, rate, digits=400)
    return torch.tensor(float(value), dtype=torch.float64)


def implied_vol_by_mpmath(
    prices: torch.Tensor, spot: float, strikes: torch.Tensor, tau: float
) -> torch.Tensor:
    """Inverts each call price by bisection on the price in 60-digit arithmetic.

    Sixty digits hold every price above ``1e-40`` of spot exactly through the
    subtraction, and bisection needs no derivative and cannot leave its
    bracket.

    Args:
        prices: Call prices of shape ``(n_strikes,)``.
        spot: Spot price.
        strikes: Strike grid of shape ``(n_strikes,)``.
        tau: Time to maturity in years.

    Returns:
        Implied volatilities of shape ``(n_strikes,)`` in float64.
    """
    roots: list[float] = []
    with mpmath.workdps(60):
        for price, strike in zip(prices.tolist(), strikes.tolist(), strict=True):
            gap = functools.partial(_price_gap, spot, strike, tau, price)
            root = mpmath.findroot(
                gap,
                (mpmath.mpf("1e-8"), mpmath.mpf(20)),
                solver="bisect",
                tol=mpmath.mpf("1e-40"),
                maxsteps=400,
                verify=False,
            )
            roots.append(float(root))
    return torch.tensor(roots, dtype=torch.float64)


def _price_gap(spot: float, strike: float, tau: float, price: float, sigma: Any) -> Any:
    return _call_by_mpmath(spot, strike, sigma, tau, 0.0, digits=60) - mpmath.mpf(price)


def rough_bergomi_factor_per_maturity(
    hurst: float, rho: float, maturity: float, n_steps: int
) -> torch.Tensor:
    """Builds the rough Bergomi factor from the covariance at the given maturity.

    Evaluates the hypergeometric function for every pair of dates of this
    maturity's grid and factors the covariance, the build that rescaling a
    cached unit-maturity factor replaced.

    Args:
        hurst: Hurst exponent ``H`` in ``(0, 1/2)``.
        rho: Correlation between the variance and price drivers.
        maturity: Horizon in years.
        n_steps: Number of grid dates.

    Returns:
        The float64 lower-triangular factor of shape ``(2 n_steps, 2 n_steps)``.
    """
    gamma = 0.5 - hurst
    times = [maturity * (i + 1) / n_steps for i in range(n_steps)]
    volterra = torch.empty(n_steps, n_steps, dtype=torch.float64)
    for i, early in enumerate(times):
        volterra[i, i] = early ** (2.0 * hurst)
        for j in range(i + 1, n_steps):
            ratio = times[j] / early
            hypergeometric = mpmath.hyp2f1(1.0, gamma, 2.0 - gamma, 1.0 / ratio)
            value = early ** (2.0 * hurst) * float(
                2.0 * hurst * ratio**-gamma / (1.0 - gamma) * hypergeometric
            )
            volterra[i, j] = volterra[j, i] = value
    grid = torch.tensor(times, dtype=torch.float64)
    scale = rho * math.sqrt(2.0 * hurst) / (hurst + 0.5)
    lag = (grid[:, None] - torch.minimum(grid[:, None], grid)).clamp(min=0.0)
    cross = scale * (grid[:, None] ** (hurst + 0.5) - lag ** (hurst + 0.5))
    driver = torch.minimum(grid[:, None], grid)
    covariance = torch.cat(
        (torch.cat((volterra, cross), dim=1), torch.cat((cross.T, driver), dim=1)), dim=0
    )
    return torch.linalg.cholesky(covariance)
