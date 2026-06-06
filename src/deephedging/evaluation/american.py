"""American put valuation by least-squares Monte Carlo.

Longstaff and Schwartz estimate the continuation value at each exercise
date by regressing realised discounted cashflows on basis functions of
the spot over in-the-money paths, exercising where intrinsic value
beats the fitted continuation. The regression uses a quadratic
polynomial basis, sufficient for the one-dimensional put. A binomial
tree on the same dynamics serves as the converged reference the Monte
Carlo estimate is pinned against, and the zero-rate degeneracy gives a
second free golden: without interest on the strike an American put is
worth exactly its European twin.
"""

import math

import torch

from deephedging.market.gbm import GBMSimulator
from deephedging.market.noise import NoiseSpec
from deephedging.pricing import PriceEstimate


def binomial_american_put(
    s0: float,
    strike: float,
    sigma: float,
    maturity: float,
    rate: float,
    n_steps: int = 2000,
) -> float:
    """Prices an American put on a Cox-Ross-Rubinstein tree.

    Args:
        s0: Spot price.
        strike: Strike price.
        sigma: Volatility.
        maturity: Horizon in years.
        rate: Continuously compounded interest rate.
        n_steps: Tree depth; two thousand converges past the Monte
            Carlo tolerances used in the goldens.

    Returns:
        The tree price.
    """
    dt = maturity / n_steps
    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp(rate * dt)
    probability = (growth - down) / (up - down)
    discount = math.exp(-rate * dt)
    exponents = torch.arange(n_steps + 1, dtype=torch.float64)
    spots = s0 * up ** (2.0 * exponents - n_steps)
    values = torch.clamp(strike - spots, min=0.0)
    for step in range(n_steps - 1, -1, -1):
        values = discount * (probability * values[1:] + (1.0 - probability) * values[:-1])
        spots = s0 * up ** (2.0 * torch.arange(step + 1, dtype=torch.float64) - step)
        values = torch.maximum(values, torch.clamp(strike - spots, min=0.0))
    return float(values[0])


def lsm_american_put(
    simulator: GBMSimulator,
    strike: float,
    rate: float,
    n_paths: int = 200_000,
    seed: int = 0,
) -> PriceEstimate:
    """Prices an American put by least-squares Monte Carlo.

    Args:
        simulator: Risk-neutral dynamics; its drift must equal ``rate``.
        strike: Strike price.
        rate: Continuously compounded interest rate.
        n_paths: Number of simulated paths.
        seed: Noise seed for a reproducible estimate.

    Returns:
        The discounted estimate with its standard error.

    Raises:
        ValueError: If the simulator drift disagrees with the rate.
    """
    if simulator.mu != rate:
        raise ValueError(f"risk-neutral pricing needs simulator drift {rate}, got {simulator.mu}")
    state = simulator.simulate(n_paths, noise=NoiseSpec(seed=seed))
    paths = state.spot.to(torch.float64)
    n_steps = state.n_steps
    dt = simulator.maturity / n_steps
    step_discount = math.exp(-rate * dt)

    cashflow = torch.clamp(strike - paths[-1], min=0.0)
    for t in range(n_steps - 1, 0, -1):
        cashflow = cashflow * step_discount
        spot = paths[t]
        intrinsic = torch.clamp(strike - spot, min=0.0)
        in_the_money = intrinsic > 0.0
        if int(in_the_money.sum()) < 3:
            continue
        regressors = torch.stack(
            (
                torch.ones_like(spot[in_the_money]),
                spot[in_the_money],
                spot[in_the_money] ** 2,
            ),
            dim=1,
        )
        coefficients = torch.linalg.lstsq(
            regressors, cashflow[in_the_money].unsqueeze(1)
        ).solution.squeeze(1)
        continuation = regressors @ coefficients
        exercise = intrinsic[in_the_money] > continuation
        updated = torch.where(exercise, intrinsic[in_the_money], cashflow[in_the_money])
        cashflow = cashflow.clone()
        cashflow[in_the_money] = updated
    cashflow = cashflow * step_discount
    value = float(cashflow.mean())
    spread = float(cashflow.std()) / math.sqrt(n_paths)
    return PriceEstimate(value=value, standard_error=spread, provenance="lsm")
