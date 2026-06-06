"""Tests for the hedging episode engine."""

import math

import torch

from deephedging.evaluation import bs_call_price, delta_hedge_positions
from deephedging.frictions import NoCost, ProportionalCost
from deephedging.instruments import EuropeanCall
from deephedging.market import GBMSimulator, MarketState, NoiseSpec
from deephedging.policies.base import HedgePolicy
from deephedging.training import hedge_pnl, pnl_from_positions


class ConstantPolicy(HedgePolicy):
    """Holds a fixed position at every rebalancing date."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return features.new_full((features.shape[0],), self.value), None


class ZeroPolicy(ConstantPolicy):
    """Never hedges."""

    def __init__(self) -> None:
        super().__init__(0.0)


def _state(n_paths: int = 512, n_steps: int = 20, seed: int = 5) -> MarketState:
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=n_steps)
    return sim.simulate(n_paths, noise=NoiseSpec(seed=seed))


def test_zero_policy_pnl_is_premium_minus_payoff() -> None:
    state = _state()
    payoff = EuropeanCall(strike=100.0)
    premium = 7.97
    pnl = hedge_pnl(state, ZeroPolicy(), payoff, NoCost(), premium=premium)
    expected = premium - payoff(state.spot)
    assert torch.allclose(pnl, expected, atol=1e-6)


def test_loop_engine_matches_vectorised_oracle() -> None:
    state = _state()
    payoff = EuropeanCall(strike=100.0)
    cost = ProportionalCost(rate=1e-3)
    value = 0.7
    pnl_loop = hedge_pnl(state, ConstantPolicy(value), payoff, cost, premium=1.0)
    positions = state.spot.new_full((state.n_steps, state.n_paths), value)
    pnl_vec = pnl_from_positions(state, positions, payoff, cost, premium=1.0)
    assert torch.allclose(pnl_loop, pnl_vec, atol=1e-5)


def test_costs_strictly_reduce_pnl() -> None:
    state = _state()
    payoff = EuropeanCall(strike=100.0)
    free = hedge_pnl(state, ConstantPolicy(0.5), payoff, NoCost())
    costly = hedge_pnl(state, ConstantPolicy(0.5), payoff, ProportionalCost(1e-3))
    assert torch.all(costly <= free)
    assert float((free - costly).mean()) > 0.0


def test_terminal_liquidation_charges_final_close() -> None:
    state = _state()
    payoff = EuropeanCall(strike=100.0)
    cost = ProportionalCost(rate=1e-3)
    held = hedge_pnl(state, ConstantPolicy(0.5), payoff, cost)
    closed = hedge_pnl(state, ConstantPolicy(0.5), payoff, cost, liquidate_terminal=True)
    expected_charge = cost(state.spot.new_full((state.n_paths,), 0.5), state.spot[-1])
    assert torch.allclose(held - closed, expected_charge, atol=1e-6)


def test_discrete_delta_hedge_shrinks_pnl_dispersion() -> None:
    n_steps = 100
    sigma, maturity, strike = 0.2, 1.0, 100.0
    sim = GBMSimulator(s0=100.0, sigma=sigma, maturity=maturity, n_steps=n_steps)
    state = sim.simulate(20_000, noise=NoiseSpec(seed=9))
    payoff = EuropeanCall(strike=strike)
    premium = float(bs_call_price(100.0, strike, sigma, maturity))
    positions = delta_hedge_positions(state.spot, strike, sigma, maturity)
    hedged = pnl_from_positions(state, positions, payoff, NoCost(), premium=premium)
    unhedged = premium - payoff(state.spot)
    assert float(hedged.std()) < 0.2 * float(unhedged.std())
    standard_error = float(hedged.std()) / math.sqrt(hedged.shape[0])
    assert abs(float(hedged.mean())) < 5.0 * standard_error


def test_recurrent_policy_threads_state_and_matches_checkpointed_gradients() -> None:
    from deephedging.policies import RecurrentPolicy

    torch.manual_seed(17)
    policy = RecurrentPolicy(hidden_size=8)
    state = _state(n_paths=32, n_steps=6)
    payoff = EuropeanCall(strike=100.0)

    def run(checkpoint_steps: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        policy.zero_grad()
        pnl = hedge_pnl(state, policy, payoff, NoCost(), checkpoint_steps=checkpoint_steps)
        pnl.mean().backward()
        grads = [p.grad.detach().clone() for p in policy.parameters() if p.grad is not None]
        return pnl.detach(), grads

    pnl_plain, grads_plain = run(checkpoint_steps=False)
    pnl_ckpt, grads_ckpt = run(checkpoint_steps=True)
    n_params = sum(1 for _ in policy.parameters())
    assert len(grads_plain) == n_params
    assert any(bool(torch.any(g != 0)) for g in grads_plain)
    assert torch.allclose(pnl_plain, pnl_ckpt, atol=1e-6)
    for a, b in zip(grads_plain, grads_ckpt, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_amp_episode_keeps_fp32_state_and_stays_close() -> None:
    from deephedging.policies import FeedForwardPolicy

    torch.manual_seed(19)
    policy = FeedForwardPolicy(hidden_sizes=(16,))
    state = _state(n_paths=256, n_steps=10)
    payoff = EuropeanCall(strike=100.0)
    plain = hedge_pnl(state, policy, payoff, NoCost(), premium=4.0)
    mixed = hedge_pnl(state, policy, payoff, NoCost(), premium=4.0, amp=True)
    assert mixed.dtype == state.spot.dtype
    assert torch.all(torch.isfinite(mixed))
    assert float((mixed - plain).abs().mean()) < 0.05 * float(plain.abs().mean() + 1.0)


def test_checkpointed_episode_matches_plain_gradients() -> None:
    from deephedging.policies import FeedForwardPolicy

    torch.manual_seed(13)
    policy = FeedForwardPolicy(hidden_sizes=(8,))
    state = _state(n_paths=64, n_steps=8)
    payoff = EuropeanCall(strike=100.0)

    def run(checkpoint_steps: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        policy.zero_grad()
        pnl = hedge_pnl(state, policy, payoff, NoCost(), checkpoint_steps=checkpoint_steps)
        loss = pnl.mean()
        loss.backward()
        grads = [p.grad.detach().clone() for p in policy.parameters() if p.grad is not None]
        return loss.detach(), grads

    loss_plain, grads_plain = run(checkpoint_steps=False)
    loss_ckpt, grads_ckpt = run(checkpoint_steps=True)
    assert torch.allclose(loss_plain, loss_ckpt, atol=1e-7)
    assert len(grads_plain) == len(grads_ckpt)
    for a, b in zip(grads_plain, grads_ckpt, strict=True):
        assert torch.allclose(a, b, atol=1e-6)
