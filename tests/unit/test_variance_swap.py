"""Tests for the variance swap market and the single-asset payoff adapter."""

import math

import pytest
import torch

from deephedging.frictions import ProportionalCost
from deephedging.instruments import EuropeanCall, SingleAssetPayoff
from deephedging.market import HestonSimulator, HestonVarianceSwapSimulator, NoiseSpec
from deephedging.training import pnl_from_positions

_V0 = 0.04
_KAPPA = 1.5
_THETA = 0.04


def _heston(maturity: float = 0.25, n_steps: int = 30) -> HestonSimulator:
    return HestonSimulator(
        s0=100.0,
        v0=_V0,
        kappa=_KAPPA,
        theta=_THETA,
        xi=0.5,
        rho=-0.7,
        maturity=maturity,
        n_steps=n_steps,
    )


def _simulator(
    maturity: float = 0.25, n_steps: int = 30, vs_maturity: float = 1.0
) -> HestonVarianceSwapSimulator:
    return HestonVarianceSwapSimulator(heston=_heston(maturity, n_steps), vs_maturity=vs_maturity)


def test_shape_initial_values_and_replay() -> None:
    sim = _simulator()
    state = sim.simulate(512, noise=NoiseSpec(seed=41))
    assert state.spot.shape == (31, 512, 2)
    assert torch.all(state.spot[0, :, 0] == 100.0)
    expected = sim.initial_swap_value()
    assert torch.allclose(state.spot[0, :, 1], torch.full((512,), expected), atol=1e-6)
    replay = sim.simulate(512, noise=NoiseSpec(seed=41))
    assert torch.equal(state.spot, replay.spot)


def test_spot_column_matches_plain_heston_bitwise() -> None:
    sim = _simulator()
    noise = NoiseSpec(seed=43)
    plain = sim.heston.simulate(256, noise=noise)
    wrapped = sim.simulate(256, noise=noise)
    assert torch.equal(wrapped.spot[..., 0], plain.spot)
    assert torch.equal(wrapped.aux["variance"], plain.aux["variance"])


def test_swap_value_strictly_positive() -> None:
    state = _simulator().simulate(50_000, noise=NoiseSpec(seed=47))
    assert torch.all(state.spot[..., 1] > 0.0)


def test_initial_swap_value_closed_form() -> None:
    sim = _simulator(vs_maturity=2.0)
    decay = (1.0 - math.exp(-_KAPPA * 2.0)) / _KAPPA
    expected = ((_V0 - _THETA) * decay + _THETA * 2.0) / 2.0
    assert sim.initial_swap_value() == pytest.approx(expected)


def test_swap_value_recomputes_from_grid() -> None:
    sim = _simulator(n_steps=10)
    state = sim.simulate(64, noise=NoiseSpec(seed=53))
    variance = state.aux["variance"]
    dt = sim.maturity / sim.n_steps
    t = 7
    accrued = (0.5 * dt * (variance[:t] + variance[1 : t + 1])).sum(dim=0)
    remaining = sim.vs_maturity - t * dt
    decay = (1.0 - math.exp(-_KAPPA * remaining)) / _KAPPA
    expected = (accrued + (variance[t] - _THETA) * decay + _THETA * remaining) / sim.vs_maturity
    assert torch.allclose(state.spot[t, :, 1], expected, atol=1e-6)


def test_swap_mean_is_stationary_at_long_run_variance() -> None:
    state = _simulator().simulate(200_000, noise=NoiseSpec(seed=59))
    swap = state.spot[..., 1]
    assert swap[-1].mean().item() == pytest.approx(_THETA, rel=0.01)


def test_rejects_swap_maturity_inside_horizon() -> None:
    with pytest.raises(ValueError):
        HestonVarianceSwapSimulator(heston=_heston(maturity=1.0), vs_maturity=1.0)


def test_single_asset_payoff_reads_one_column() -> None:
    state = _simulator().simulate(128, noise=NoiseSpec(seed=61))
    payoff = SingleAssetPayoff(inner=EuropeanCall(strike=100.0))
    direct = EuropeanCall(strike=100.0)(state.spot[..., 0])
    assert torch.equal(payoff(state.spot), direct)
    assert payoff(state.spot).shape == (128,)


def test_single_asset_payoff_rejects_negative_index() -> None:
    with pytest.raises(ValueError):
        SingleAssetPayoff(inner=EuropeanCall(strike=100.0), asset=-1)


def test_two_asset_pnl_contracts_asset_axis() -> None:
    sim = _simulator(n_steps=2)
    state = sim.simulate(16, noise=NoiseSpec(seed=67))
    positions = torch.zeros((2, 16, 2))
    positions[:, :, 0] = 0.5
    positions[:, :, 1] = 2.0
    payoff = SingleAssetPayoff(inner=EuropeanCall(strike=100.0))
    pnl = pnl_from_positions(state, positions, payoff, ProportionalCost(rate=0.01))
    gains = 0.5 * (state.spot[-1, :, 0] - state.spot[0, :, 0]) + 2.0 * (
        state.spot[-1, :, 1] - state.spot[0, :, 1]
    )
    costs = 0.01 * (0.5 * state.spot[0, :, 0] + 2.0 * state.spot[0, :, 1])
    expected = gains - costs - payoff(state.spot)
    assert pnl.shape == (16,)
    assert torch.allclose(pnl, expected, atol=1e-4)
