"""Tests for the Merton jump-diffusion simulator and closed form."""

import math

import pytest
import torch

from deephedging.evaluation import merton_call_price
from deephedging.instruments import EuropeanCall
from deephedging.market import MertonSimulator, NoiseSpec
from deephedging.pricing import MonteCarloPricer


def _simulator(**overrides: float | int) -> MertonSimulator:
    params: dict[str, float | int] = {
        "s0": 100.0,
        "sigma": 0.2,
        "jump_intensity": 1.0,
        "jump_mean": -0.1,
        "jump_vol": 0.15,
        "maturity": 1.0,
        "n_steps": 50,
    }
    params.update(overrides)
    return MertonSimulator(**params)  # type: ignore[arg-type]


def test_shape_initial_value_and_replay() -> None:
    sim = _simulator()
    state = sim.simulate(512, noise=NoiseSpec(seed=101))
    assert state.spot.shape == (51, 512)
    assert torch.all(state.spot[0] == 100.0)
    assert torch.all(state.spot > 0.0)
    replay = sim.simulate(512, noise=NoiseSpec(seed=101))
    assert torch.equal(state.spot, replay.spot)
    assert torch.equal(state.aux["jumps"], replay.aux["jumps"])


def test_jump_channel_counts_cumulatively() -> None:
    state = _simulator().simulate(4096, noise=NoiseSpec(seed=103))
    jumps = state.aux["jumps"]
    assert torch.all(jumps[0] == 0.0)
    assert torch.all(jumps[1:] >= jumps[:-1])
    expected_total = 1.0 * 1.0
    assert abs(float(jumps[-1].mean()) - expected_total) < 0.1


def test_zero_drift_terminal_mean_is_martingale() -> None:
    state = _simulator().simulate(400_000, noise=NoiseSpec(seed=107))
    terminal = state.terminal
    standard_error = float(terminal.std()) / math.sqrt(terminal.shape[0])
    assert abs(float(terminal.mean()) - 100.0) < 4.0 * standard_error


def test_zero_intensity_recovers_gbm_moments() -> None:
    sim = _simulator(jump_intensity=0.0)
    state = sim.simulate(200_000, noise=NoiseSpec(seed=109))
    log_returns = torch.log(state.terminal / 100.0)
    assert torch.all(state.aux["jumps"] == 0.0)
    assert abs(float(log_returns.mean()) - (-0.5 * 0.04)) < 5e-3
    assert abs(float(log_returns.std()) - 0.2) < 5e-3


def test_jumps_fatten_tails() -> None:
    plain = _simulator(jump_intensity=0.0).simulate(200_000, noise=NoiseSpec(seed=113))
    jumpy = _simulator().simulate(200_000, noise=NoiseSpec(seed=113))

    def excess_kurtosis(state_terminal: torch.Tensor) -> float:
        log_returns = torch.log(state_terminal / 100.0)
        centred = log_returns - log_returns.mean()
        return float((centred**4).mean() / (centred**2).mean() ** 2) - 3.0

    assert excess_kurtosis(jumpy.terminal) > excess_kurtosis(plain.terminal) + 0.2


def test_monte_carlo_matches_closed_form() -> None:
    sim = _simulator(n_steps=100)
    for strike in (90.0, 100.0, 110.0):
        estimate = MonteCarloPricer(n_paths=400_000, seed=127).price(
            EuropeanCall(strike=strike), sim
        )
        reference = float(
            merton_call_price(
                spot=100.0,
                strike=strike,
                sigma=0.2,
                jump_intensity=1.0,
                jump_mean=-0.1,
                jump_vol=0.15,
                tau=1.0,
            )
        )
        assert abs(estimate.value - reference) < 3.0 * estimate.standard_error


def test_closed_form_zero_intensity_is_black_scholes() -> None:
    from deephedging.evaluation import bs_call_price

    merton = float(
        merton_call_price(
            spot=100.0,
            strike=100.0,
            sigma=0.2,
            jump_intensity=0.0,
            jump_mean=-0.1,
            jump_vol=0.15,
            tau=1.0,
        )
    )
    reference = float(bs_call_price(100.0, 100.0, 0.2, 1.0))
    assert abs(merton - reference) < 1e-10


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        _simulator(jump_intensity=-1.0)
    with pytest.raises(ValueError):
        _simulator(jump_vol=-0.1)
    with pytest.raises(ValueError):
        _simulator(jump_intensity=200.0, n_steps=10)
