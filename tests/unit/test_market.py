"""Tests for market simulators."""

import math

import pytest
import torch

from deephedging.market import GBMSimulator, MarketState, NoiseSpec


def test_shape_and_initial_value() -> None:
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=30)
    state = sim.simulate(64, noise=NoiseSpec(seed=7))
    assert isinstance(state, MarketState)
    assert state.spot.shape == (31, 64)
    assert state.n_steps == 30
    assert state.n_paths == 64
    assert torch.all(state.spot[0] == 100.0)
    assert torch.all(state.spot > 0.0)


def test_reproducible_with_noise_spec() -> None:
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=10)
    first = sim.simulate(16, noise=NoiseSpec(seed=3))
    second = sim.simulate(16, noise=NoiseSpec(seed=3))
    assert torch.equal(first.spot, second.spot)


def test_noise_streams_are_independent() -> None:
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=10)
    base = NoiseSpec(seed=3)
    first = sim.simulate(16, noise=base.child(1))
    second = sim.simulate(16, noise=base.child(2))
    assert not torch.equal(first.spot, second.spot)


def test_zero_drift_terminal_mean() -> None:
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=50)
    terminal = sim.simulate(200_000, noise=NoiseSpec(seed=7)).terminal
    standard_error = float(terminal.std()) / math.sqrt(terminal.shape[0])
    assert abs(float(terminal.mean()) - 100.0) < 4.0 * standard_error


def test_terminal_log_moments() -> None:
    sigma, maturity = 0.25, 2.0
    sim = GBMSimulator(s0=100.0, sigma=sigma, maturity=maturity, n_steps=20)
    state = sim.simulate(200_000, noise=NoiseSpec(seed=7))
    log_returns = torch.log(state.terminal / 100.0)
    expected_mean = -0.5 * sigma**2 * maturity
    expected_std = sigma * math.sqrt(maturity)
    assert abs(float(log_returns.mean()) - expected_mean) < 5e-3
    assert abs(float(log_returns.std()) - expected_std) < 5e-3


def test_drift_shifts_terminal_mean() -> None:
    mu = 0.1
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=20, mu=mu)
    terminal = sim.simulate(200_000, noise=NoiseSpec(seed=7)).terminal
    expected = 100.0 * math.exp(mu)
    assert abs(float(terminal.mean()) - expected) / expected < 5e-3


def test_running_max_cached_and_correct() -> None:
    sim = GBMSimulator(s0=100.0, sigma=0.3, maturity=1.0, n_steps=20)
    state = sim.simulate(128, noise=NoiseSpec(seed=11))
    for t in (0, 7, 20):
        expected = state.spot[: t + 1].max(dim=0).values
        assert torch.equal(state.running_max(t), expected)


def test_state_to_same_device_is_identity() -> None:
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=5)
    state = sim.simulate(8, noise=NoiseSpec(seed=5))
    assert state.to("cpu") is state


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        GBMSimulator(s0=-1.0, sigma=0.2, maturity=1.0, n_steps=10)
    with pytest.raises(ValueError):
        GBMSimulator(s0=100.0, sigma=-0.1, maturity=1.0, n_steps=10)
    with pytest.raises(ValueError):
        GBMSimulator(s0=100.0, sigma=0.2, maturity=0.0, n_steps=10)
    with pytest.raises(ValueError):
        GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=0)
