"""Tests for barrier payoffs."""

import pytest
import torch

from deephedging.instruments import EuropeanCall, UpAndOutCall
from deephedging.market import GBMSimulator, NoiseSpec


def _paths(n_paths: int = 50_000, seed: int = 23) -> torch.Tensor:
    sim = GBMSimulator(s0=100.0, sigma=0.3, maturity=1.0, n_steps=50)
    return sim.simulate(n_paths, noise=NoiseSpec(seed=seed)).spot


def test_unreachable_barrier_equals_vanilla() -> None:
    paths = _paths()
    vanilla = EuropeanCall(strike=100.0)
    knocked = UpAndOutCall(strike=100.0, barrier=1e9)
    assert torch.equal(knocked(paths), vanilla(paths))


def test_knocked_paths_pay_zero() -> None:
    paths = torch.tensor(
        [
            [100.0, 100.0],
            [125.0, 110.0],
            [115.0, 112.0],
        ]
    )
    payoff = UpAndOutCall(strike=100.0, barrier=120.0)
    result = payoff(paths)
    assert float(result[0]) == 0.0
    assert float(result[1]) == 12.0


def test_barrier_payoff_never_exceeds_vanilla() -> None:
    paths = _paths()
    vanilla = EuropeanCall(strike=100.0)
    knocked = UpAndOutCall(strike=100.0, barrier=130.0)
    assert torch.all(knocked(paths) <= vanilla(paths))
    assert float(knocked(paths).mean()) < float(vanilla(paths).mean())


def test_barrier_below_strike_rejected() -> None:
    with pytest.raises(ValueError):
        UpAndOutCall(strike=100.0, barrier=90.0)
