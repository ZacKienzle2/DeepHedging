"""Implementations that faster ones replaced, kept as equivalence oracles.

WORKFLOW.md keeps the obvious version of a function when it is rewritten, and
``nox -s generate -- deephedging.baselines.<old> deephedging.<module>.<new>``
writes the Hypothesis test that compares the two over generated inputs. Each
function here is the version its replacement's commit measured against, with
the loop that the rewrite removed. None of them is exported or used at run time.
"""

import math
from typing import cast

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
