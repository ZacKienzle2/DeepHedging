"""Tests for the no-transaction-band policy."""

import pytest
import torch

from deephedging import (
    EuropeanCall,
    GBMSimulator,
    NoiseSpec,
    NoTransactionBandPolicy,
    ProportionalCost,
    bs_call_delta,
    hedge_pnl,
)

_SIGMA = 0.2
_MATURITY = 30 / 365


def _features(log_moneyness: float, tau: float, held: float, n: int = 8) -> torch.Tensor:
    column = torch.tensor([log_moneyness, tau, held])
    return column.expand(n, 3).clone()


def test_position_stays_inside_band_around_delta() -> None:
    torch.manual_seed(31)
    policy = NoTransactionBandPolicy(sigma=_SIGMA, maturity=_MATURITY)
    features = _features(0.05, 0.5, 0.0)
    position, state = policy(features)
    delta = float(bs_call_delta(100.0 * torch.tensor(0.05).exp(), 100.0, _SIGMA, 0.5 * _MATURITY))
    assert state is None
    assert position.shape == (8,)
    assert torch.all(position <= delta + 2.0)
    assert torch.all(position >= delta - 2.0)


def test_held_position_inside_band_is_kept() -> None:
    torch.manual_seed(32)
    policy = NoTransactionBandPolicy(sigma=_SIGMA, maturity=_MATURITY)
    spot_ratio = torch.tensor(0.0)
    delta = float(bs_call_delta(100.0, 100.0, _SIGMA, 0.5 * _MATURITY))
    features = _features(float(spot_ratio), 0.5, delta)
    position, _ = policy(features)
    assert torch.allclose(position, torch.full((8,), delta), atol=1e-6)


def test_far_inventory_is_clamped_to_the_nearer_edge() -> None:
    torch.manual_seed(33)
    policy = NoTransactionBandPolicy(sigma=_SIGMA, maturity=_MATURITY)
    high, _ = policy(_features(0.0, 0.5, 5.0))
    low, _ = policy(_features(0.0, 0.5, -5.0))
    assert torch.all(high > low)
    assert torch.all(high <= 5.0)
    assert torch.all(low >= -5.0)


def test_gradients_reach_the_binding_edge() -> None:
    torch.manual_seed(34)
    policy = NoTransactionBandPolicy(sigma=_SIGMA, maturity=_MATURITY)
    position, _ = policy(_features(0.0, 0.5, 5.0))
    position.sum().backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads
    assert any(float(g.abs().sum()) > 0.0 for g in grads)


def test_band_anchor_matches_closed_form_delta() -> None:
    torch.manual_seed(36)
    policy = NoTransactionBandPolicy(sigma=_SIGMA, maturity=_MATURITY)
    for log_moneyness in (-0.1, 0.0, 0.08):
        for tau in (0.2, 0.5, 1.0):
            high_features = _features(log_moneyness, tau, 5.0, n=1)
            low_features = _features(log_moneyness, tau, -5.0, n=1)
            with torch.no_grad():
                upper, _ = policy(high_features)
                lower, _ = policy(low_features)
                high_widths = torch.nn.functional.softplus(policy.net(high_features))
                low_widths = torch.nn.functional.softplus(policy.net(low_features))
            spot = 100.0 * torch.tensor(log_moneyness).exp()
            delta = float(bs_call_delta(spot, 100.0, _SIGMA, tau * _MATURITY))
            assert float(upper) == pytest.approx(delta + float(high_widths[0, 1]), abs=1e-5)
            assert float(lower) == pytest.approx(delta - float(low_widths[0, 0]), abs=1e-5)


def test_no_gradient_when_held_inside_band() -> None:
    torch.manual_seed(37)
    policy = NoTransactionBandPolicy(sigma=_SIGMA, maturity=_MATURITY)
    delta = float(bs_call_delta(100.0, 100.0, _SIGMA, 0.5 * _MATURITY))
    position, _ = policy(_features(0.0, 0.5, delta))
    grads = torch.autograd.grad(position.sum(), list(policy.parameters()), allow_unused=True)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in grads)


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        NoTransactionBandPolicy(sigma=0.0, maturity=_MATURITY)
    with pytest.raises(ValueError):
        NoTransactionBandPolicy(sigma=_SIGMA, maturity=0.0)
    with pytest.raises(ValueError):
        NoTransactionBandPolicy(sigma=_SIGMA, maturity=_MATURITY, strike_ratio=0.0)


def test_runs_through_the_episode_engine() -> None:
    torch.manual_seed(35)
    sim = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=10)
    state = sim.simulate(64, noise=NoiseSpec(seed=41))
    policy = NoTransactionBandPolicy(sigma=_SIGMA, maturity=_MATURITY)
    pnl = hedge_pnl(state, policy, EuropeanCall(strike=100.0), ProportionalCost(rate=1e-3))
    assert pnl.shape == (64,)
    pnl.mean().backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads
