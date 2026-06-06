"""End-to-end training smoke tests.

Assertions are statistical relationships, not bitwise goldens. A trained
policy must beat not hedging, and the recorded objective must improve.
"""

import pytest
import torch

from deephedging.evaluation import bs_call_price, expected_shortfall
from deephedging.frictions import NoCost, ProportionalCost
from deephedging.instruments import EuropeanCall
from deephedging.market import GBMSimulator, NoiseSpec
from deephedging.policies import FeedForwardPolicy
from deephedging.risk import CVaR
from deephedging.training import TrainConfig, hedge_pnl, train


@pytest.mark.slow
def test_trained_policy_beats_no_hedge() -> None:
    torch.manual_seed(21)
    sigma, maturity, strike = 0.2, 0.25, 100.0
    sim = GBMSimulator(s0=100.0, sigma=sigma, maturity=maturity, n_steps=10)
    payoff = EuropeanCall(strike=strike)
    premium = float(bs_call_price(100.0, strike, sigma, maturity))
    policy = FeedForwardPolicy(hidden_sizes=(16, 16))
    config = TrainConfig(n_iterations=300, batch_paths=1024, lr=2e-3, seed=4)
    result = train(sim, policy, payoff, NoCost(), CVaR(alpha=0.9), config, premium=premium)

    eval_state = sim.simulate(50_000, noise=NoiseSpec(seed=99))
    with torch.no_grad():
        hedged = hedge_pnl(eval_state, policy, payoff, NoCost(), premium=premium)
    unhedged = premium - payoff(eval_state.spot)

    es_hedged = float(expected_shortfall(hedged, alpha=0.9))
    es_unhedged = float(expected_shortfall(unhedged, alpha=0.9))
    assert es_hedged < 0.5 * es_unhedged

    early = sum(result.losses[:20]) / 20
    late = sum(result.losses[-20:]) / 20
    assert late < early


@pytest.mark.slow
def test_training_is_reproducible_with_seed() -> None:
    sigma, maturity, strike = 0.2, 0.25, 100.0
    sim = GBMSimulator(s0=100.0, sigma=sigma, maturity=maturity, n_steps=5)
    payoff = EuropeanCall(strike=strike)
    config = TrainConfig(n_iterations=20, batch_paths=256, seed=8, liquidate_terminal=True)

    def run() -> list[float]:
        torch.manual_seed(42)
        policy = FeedForwardPolicy(hidden_sizes=(8,))
        result = train(sim, policy, payoff, ProportionalCost(rate=1e-3), CVaR(alpha=0.9), config)
        return result.losses

    first, second = run(), run()
    assert len(first) == len(second) == config.n_iterations
    assert all(abs(a - b) < 1e-6 for a, b in zip(first, second, strict=True))
