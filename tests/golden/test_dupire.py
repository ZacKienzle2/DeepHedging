"""Golden tests for the Dupire local volatility chain."""

import pytest
import torch

from deephedging import LocalVolSimulator, NoiseSpec
from deephedging.calibration import HestonParams, dupire_surface, price_surface
from deephedging.instruments import EuropeanCall

_PARAMS = HestonParams(v0=0.045, kappa=2.0, theta=0.05, xi=0.4, rho=-0.6)
_S0 = 100.0
_GRID_STRIKES = torch.arange(40.0, 220.0, 2.5, dtype=torch.float64)
_GRID_TAUS = torch.tensor(
    [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5], dtype=torch.float64
)


def test_flat_surface_recovers_constant_volatility() -> None:
    flat = HestonParams(v0=0.04, kappa=5.0, theta=0.04, xi=1e-4, rho=0.0)
    surface = dupire_surface(flat, _S0, _GRID_STRIKES, _GRID_TAUS)
    log_moneyness = torch.log(surface.strikes / _S0)
    worst = 0.0
    interior = zip(surface.vols[1:-1], surface.taus[1:-1].tolist(), strict=True)
    for row, tau in interior:
        supported = log_moneyness.abs() < 1.5 * 0.2 * tau**0.5
        worst = max(worst, float((row[supported] - 0.2).abs().max()))
    assert worst < 1e-2


@pytest.mark.slow
def test_dupire_chain_reprices_heston_vanillas() -> None:
    surface = dupire_surface(_PARAMS, _S0, _GRID_STRIKES, _GRID_TAUS)
    maturity = 1.0
    sim = LocalVolSimulator(s0=_S0, surface=surface, maturity=maturity, n_steps=100)
    state = sim.simulate(400_000, noise=NoiseSpec(seed=179))
    strikes = torch.tensor([85.0, 100.0, 115.0], dtype=torch.float64)
    references = price_surface(_PARAMS.as_tensors(), _S0, strikes, maturity)
    for strike, reference in zip(strikes.tolist(), references.tolist(), strict=True):
        estimate = EuropeanCall(strike=strike)(state.spot)
        mean = float(estimate.mean())
        standard_error = float(estimate.std()) / (400_000**0.5)
        assert abs(mean - reference) < max(4.0 * standard_error, 0.05)


def test_simulator_shape_replay_and_validation() -> None:
    surface = dupire_surface(_PARAMS, _S0, _GRID_STRIKES, _GRID_TAUS)
    sim = LocalVolSimulator(s0=_S0, surface=surface, maturity=0.5, n_steps=20)
    state = sim.simulate(256, noise=NoiseSpec(seed=181))
    assert state.spot.shape == (21, 256)
    assert torch.all(state.spot[0] == _S0)
    replay = sim.simulate(256, noise=NoiseSpec(seed=181))
    assert torch.equal(state.spot, replay.spot)
    with pytest.raises(ValueError):
        LocalVolSimulator(s0=-1.0, surface=surface, maturity=0.5, n_steps=20)


def test_surface_grid_validation() -> None:
    with pytest.raises(ValueError):
        dupire_surface(_PARAMS, _S0, _GRID_STRIKES, _GRID_TAUS[:2])
    uneven = torch.tensor([80.0, 90.0, 95.0, 100.0, 120.0], dtype=torch.float64)
    with pytest.raises(ValueError):
        dupire_surface(_PARAMS, _S0, uneven, _GRID_TAUS)
