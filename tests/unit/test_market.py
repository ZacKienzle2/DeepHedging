"""Tests for market simulators."""

import math

import torch

from deephedging.market import GBMSimulator


def _make_generator(seed: int = 7) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def test_shape_and_initial_value() -> None:
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=30)
    paths = sim.simulate(64, generator=_make_generator())
    assert paths.shape == (31, 64)
    assert torch.all(paths[0] == 100.0)
    assert torch.all(paths > 0.0)


def test_reproducible_with_seed() -> None:
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=10)
    first = sim.simulate(16, generator=_make_generator(3))
    second = sim.simulate(16, generator=_make_generator(3))
    assert torch.equal(first, second)


def test_zero_drift_terminal_mean() -> None:
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=50)
    paths = sim.simulate(200_000, generator=_make_generator())
    terminal = paths[-1]
    standard_error = float(terminal.std()) / math.sqrt(terminal.shape[0])
    assert abs(float(terminal.mean()) - 100.0) < 4.0 * standard_error


def test_terminal_log_moments() -> None:
    sigma, maturity = 0.25, 2.0
    sim = GBMSimulator(s0=100.0, sigma=sigma, maturity=maturity, n_steps=20)
    paths = sim.simulate(200_000, generator=_make_generator())
    log_returns = torch.log(paths[-1] / 100.0)
    expected_mean = -0.5 * sigma**2 * maturity
    expected_std = sigma * math.sqrt(maturity)
    assert abs(float(log_returns.mean()) - expected_mean) < 5e-3
    assert abs(float(log_returns.std()) - expected_std) < 5e-3


def test_drift_shifts_terminal_mean() -> None:
    mu = 0.1
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=20, mu=mu)
    paths = sim.simulate(200_000, generator=_make_generator())
    expected = 100.0 * math.exp(mu)
    assert abs(float(paths[-1].mean()) - expected) / expected < 5e-3


def test_invalid_parameters_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        GBMSimulator(s0=-1.0, sigma=0.2, maturity=1.0, n_steps=10)
    with pytest.raises(ValueError):
        GBMSimulator(s0=100.0, sigma=-0.1, maturity=1.0, n_steps=10)
    with pytest.raises(ValueError):
        GBMSimulator(s0=100.0, sigma=0.2, maturity=0.0, n_steps=10)
    with pytest.raises(ValueError):
        GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=0)
