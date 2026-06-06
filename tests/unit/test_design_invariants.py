"""Tests pinning the variance-reduction design decisions.

Antithetic pairing is off by default because a converged hedge drives
the residual loss toward an even function of the Brownian driver, where
negated noise reproduces the loss and pairing inflates the estimator
variance it is meant to cut. An additive policy-independent control
variate is confined to evaluation because it provably leaves training
gradients untouched. Both claims are measured here rather than assumed,
so a regression in either invalidates the documented design.
"""

import pytest
import torch

from deephedging import (
    CVaR,
    EuropeanCall,
    FeedForwardPolicy,
    GBMSimulator,
    MarketState,
    NoCost,
    NoiseSpec,
    TrainConfig,
    bs_call_price,
    delta_hedge_positions,
    hedge_pnl,
    pnl_from_positions,
    train,
)

_SIGMA = 0.2
_MATURITY = 30 / 365
_N_STEPS = 30
_STRIKE = 100.0


def _antithetic_states(n_paths: int, seed: int) -> tuple[MarketState, MarketState]:
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn((_N_STEPS, n_paths), generator=generator)
    dt = _MATURITY / _N_STEPS

    def build(driver: torch.Tensor) -> MarketState:
        increments = -0.5 * _SIGMA**2 * dt + _SIGMA * dt**0.5 * driver
        log_spot = torch.zeros((_N_STEPS + 1, n_paths))
        log_spot[1:] = torch.cumsum(increments, dim=0)
        return MarketState.from_spot(100.0 * torch.exp(log_spot))

    return build(z), build(-z)


def _residual_correlation(policy: FeedForwardPolicy, premium: float) -> tuple[float, float]:
    plus, minus = _antithetic_states(50_000, seed=11)
    payoff = EuropeanCall(strike=_STRIKE)
    with torch.no_grad():
        loss_plus = -hedge_pnl(plus, policy, payoff, NoCost(), premium=premium)
        loss_minus = -hedge_pnl(minus, policy, payoff, NoCost(), premium=premium)
    correlation = float(torch.corrcoef(torch.stack((loss_plus, loss_minus)))[0, 1])
    paired = 0.5 * (loss_plus + loss_minus)
    variance_ratio = float(paired.var() / (0.5 * loss_plus.var()))
    return correlation, variance_ratio


@pytest.mark.slow
def test_antithetic_pairing_flips_from_helpful_to_harmful() -> None:
    torch.manual_seed(3)
    premium = float(bs_call_price(100.0, _STRIKE, _SIGMA, _MATURITY))
    untrained = FeedForwardPolicy(hidden_sizes=(64, 64))
    untrained_corr, untrained_ratio = _residual_correlation(untrained, premium)
    assert untrained_corr < 0.0
    assert untrained_ratio < 1.0

    sim = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=_N_STEPS)
    trained = FeedForwardPolicy(hidden_sizes=(64, 64))
    config = TrainConfig(n_iterations=800, batch_paths=8192, lr=1e-3, seed=4)
    train(
        sim,
        trained,
        EuropeanCall(strike=_STRIKE),
        NoCost(),
        CVaR(alpha=0.95),
        config,
        premium=premium,
    )
    trained_corr, trained_ratio = _residual_correlation(trained, premium)
    assert trained_corr > 0.5
    assert trained_ratio > 1.5


def test_additive_control_variate_leaves_gradients_untouched() -> None:
    torch.manual_seed(5)
    sim = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=_N_STEPS)
    payoff = EuropeanCall(strike=_STRIKE)
    premium = float(bs_call_price(100.0, _STRIKE, _SIGMA, _MATURITY))
    state = sim.simulate(4096, noise=NoiseSpec(seed=21))
    deltas = delta_hedge_positions(state.spot, _STRIKE, _SIGMA, _MATURITY)
    control = pnl_from_positions(state, deltas, payoff, NoCost(), premium=premium)
    control = control - control.mean()

    policy = FeedForwardPolicy(hidden_sizes=(32, 32))
    plain_loss = (-hedge_pnl(state, policy, payoff, NoCost(), premium=premium)).mean()
    plain_grads = torch.autograd.grad(plain_loss, list(policy.parameters()))
    cv_loss = (-(hedge_pnl(state, policy, payoff, NoCost(), premium=premium) - control)).mean()
    cv_grads = torch.autograd.grad(cv_loss, list(policy.parameters()))

    for plain, controlled in zip(plain_grads, cv_grads, strict=True):
        assert torch.equal(plain, controlled)
