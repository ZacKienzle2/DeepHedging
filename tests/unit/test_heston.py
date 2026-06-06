"""Tests for the Heston simulator."""

import math

import pytest
import torch

from deephedging.market import HestonSimulator


def _make_generator(seed: int = 19) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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


def test_shape_and_initial_value() -> None:
    sim = _simulator()
    paths = sim.simulate(64, generator=_make_generator())
    assert paths.shape == (101, 64)
    assert torch.all(paths[0] == 100.0)
    assert torch.all(paths > 0.0)


def test_reproducible_with_seed() -> None:
    sim = _simulator()
    first = sim.simulate(16, generator=_make_generator(3))
    second = sim.simulate(16, generator=_make_generator(3))
    assert torch.equal(first, second)


def test_zero_vol_of_vol_degenerates_to_gbm_marginals() -> None:
    v0 = 0.04
    sim = _simulator(v0=v0, theta=v0, xi=0.0, rho=0.0)
    paths = sim.simulate(200_000, generator=_make_generator())
    log_returns = torch.log(paths[-1] / 100.0)
    expected_std = math.sqrt(v0 * 1.0)
    expected_mean = -0.5 * v0 * 1.0
    assert abs(float(log_returns.std()) - expected_std) < 5e-3
    assert abs(float(log_returns.mean()) - expected_mean) < 5e-3


def test_zero_drift_terminal_mean_near_s0() -> None:
    sim = _simulator()
    paths = sim.simulate(200_000, generator=_make_generator())
    terminal = paths[-1]
    standard_error = float(terminal.std()) / math.sqrt(terminal.shape[0])
    assert abs(float(terminal.mean()) - 100.0) < max(5.0 * standard_error, 0.3)


def test_negative_rho_skews_left_tail() -> None:
    down = _simulator(rho=-0.9).simulate(100_000, generator=_make_generator())
    up = _simulator(rho=0.9).simulate(100_000, generator=_make_generator())

    def central_third_moment(paths: torch.Tensor) -> float:
        log_returns = torch.log(paths[-1] / 100.0)
        centred = log_returns - log_returns.mean()
        return float((centred**3).mean())

    assert central_third_moment(down) < central_third_moment(up)


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        _simulator(rho=1.5)
    with pytest.raises(ValueError):
        _simulator(kappa=0.0)
    with pytest.raises(ValueError):
        _simulator(v0=-0.01)
    with pytest.raises(ValueError):
        _simulator(xi=-0.1)
