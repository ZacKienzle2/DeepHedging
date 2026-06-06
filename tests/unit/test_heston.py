"""Tests for the Heston simulator."""

import math

import pytest
import torch

from deephedging.market import HestonSimulator, NoiseSpec


def _simulator(**overrides: float | int) -> HestonSimulator:
    params: dict[str, float | int] = {
        "s0": 100.0,
        "v0": 0.04,
        "kappa": 1.5,
        "theta": 0.04,
        "xi": 0.5,
        "rho": -0.7,
        "maturity": 1.0,
        "n_steps": 100,
    }
    params.update(overrides)
    return HestonSimulator(**params)  # type: ignore[arg-type]


def test_shape_initial_value_and_variance_channel() -> None:
    state = _simulator().simulate(64, noise=NoiseSpec(seed=19))
    assert state.spot.shape == (101, 64)
    assert torch.all(state.spot[0] == 100.0)
    assert torch.all(state.spot > 0.0)
    variance = state.aux["variance"]
    assert variance.shape == (101, 64)
    assert torch.all(variance[0] == 0.04)
    assert torch.all(variance >= 0.0)


def test_reproducible_with_noise_spec() -> None:
    sim = _simulator()
    first = sim.simulate(16, noise=NoiseSpec(seed=3))
    second = sim.simulate(16, noise=NoiseSpec(seed=3))
    assert torch.equal(first.spot, second.spot)
    assert torch.equal(first.aux["variance"], second.aux["variance"])


def test_zero_vol_of_vol_degenerates_to_gbm_marginals() -> None:
    v0 = 0.04
    state = _simulator(v0=v0, theta=v0, xi=0.0, rho=0.0).simulate(200_000, noise=NoiseSpec(seed=19))
    log_returns = torch.log(state.terminal / 100.0)
    expected_std = math.sqrt(v0 * 1.0)
    expected_mean = -0.5 * v0 * 1.0
    assert abs(float(log_returns.std()) - expected_std) < 5e-3
    assert abs(float(log_returns.mean()) - expected_mean) < 5e-3


def test_zero_drift_terminal_mean_near_s0() -> None:
    terminal = _simulator().simulate(200_000, noise=NoiseSpec(seed=19)).terminal
    standard_error = float(terminal.std()) / math.sqrt(terminal.shape[0])
    assert abs(float(terminal.mean()) - 100.0) < max(5.0 * standard_error, 0.3)


def test_negative_rho_skews_left_tail() -> None:
    down = _simulator(rho=-0.9).simulate(100_000, noise=NoiseSpec(seed=19))
    up = _simulator(rho=0.9).simulate(100_000, noise=NoiseSpec(seed=19))

    def central_third_moment(terminal: torch.Tensor) -> float:
        log_returns = torch.log(terminal / 100.0)
        centred = log_returns - log_returns.mean()
        return float((centred**3).mean())

    assert central_third_moment(down.terminal) < central_third_moment(up.terminal)


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        _simulator(rho=1.5)
    with pytest.raises(ValueError):
        _simulator(kappa=0.0)
    with pytest.raises(ValueError):
        _simulator(v0=-0.01)
    with pytest.raises(ValueError):
        _simulator(xi=-0.1)
