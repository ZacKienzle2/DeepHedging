"""Tests for barrier payoffs."""

import pytest
import torch

from deephedging.instruments import (
    DoubleKnockOutCall,
    DownAndInCall,
    DownAndOutCall,
    EuropeanCall,
    UpAndInCall,
    UpAndOutCall,
)
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


def test_up_in_out_parity_reconstructs_vanilla() -> None:
    paths = _paths()
    vanilla = EuropeanCall(strike=100.0)(paths)
    knocked_out = UpAndOutCall(strike=100.0, barrier=130.0)(paths)
    knocked_in = UpAndInCall(strike=100.0, barrier=130.0)(paths)
    assert torch.allclose(knocked_out + knocked_in, vanilla)


def test_down_in_out_parity_reconstructs_vanilla() -> None:
    paths = _paths()
    vanilla = EuropeanCall(strike=100.0)(paths)
    knocked_out = DownAndOutCall(strike=100.0, barrier=80.0)(paths)
    knocked_in = DownAndInCall(strike=100.0, barrier=80.0)(paths)
    assert torch.allclose(knocked_out + knocked_in, vanilla)


def test_down_and_out_unreachable_barrier_equals_vanilla() -> None:
    paths = _paths()
    vanilla = EuropeanCall(strike=100.0)(paths)
    knocked = DownAndOutCall(strike=100.0, barrier=1e-6)(paths)
    assert torch.equal(knocked, vanilla)


def test_double_knock_out_stays_within_single_barriers() -> None:
    paths = _paths()
    double = DoubleKnockOutCall(strike=100.0, lower_barrier=80.0, upper_barrier=130.0)(paths)
    up = UpAndOutCall(strike=100.0, barrier=130.0)(paths)
    down = DownAndOutCall(strike=100.0, barrier=80.0)(paths)
    assert torch.all(double <= up + 1e-9)
    assert torch.all(double <= down + 1e-9)


def test_double_knock_out_wide_corridor_equals_vanilla() -> None:
    paths = _paths()
    vanilla = EuropeanCall(strike=100.0)(paths)
    double = DoubleKnockOutCall(strike=100.0, lower_barrier=1e-6, upper_barrier=1e9)(paths)
    assert torch.equal(double, vanilla)


def test_corridor_pays_zero_on_either_breach() -> None:
    paths = torch.tensor(
        [
            [100.0, 100.0, 100.0],
            [105.0, 70.0, 140.0],
            [112.0, 95.0, 110.0],
        ]
    )
    payoff = DoubleKnockOutCall(strike=100.0, lower_barrier=80.0, upper_barrier=130.0)
    result = payoff(paths)
    assert float(result[0]) == 12.0
    assert float(result[1]) == 0.0
    assert float(result[2]) == 0.0


def test_barrier_family_validation() -> None:
    with pytest.raises(ValueError):
        UpAndInCall(strike=100.0, barrier=90.0)
    with pytest.raises(ValueError):
        DownAndOutCall(strike=100.0, barrier=0.0)
    with pytest.raises(ValueError):
        DownAndInCall(strike=100.0, barrier=-1.0)
    with pytest.raises(ValueError):
        DoubleKnockOutCall(strike=100.0, lower_barrier=120.0, upper_barrier=110.0)
