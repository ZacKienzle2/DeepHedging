"""Golden tests for Heston surface calibration."""

import pytest
import torch

from deephedging.calibration import (
    CalibrationConfig,
    HestonAnalyticPricer,
    HestonParams,
    calibrate_heston,
    price_surface,
)
from deephedging.instruments import EuropeanCall, EuropeanPut
from deephedging.market import HestonSimulator
from deephedging.pricing import MonteCarloPricer

_TRUE = HestonParams(v0=0.045, kappa=2.0, theta=0.05, xi=0.4, rho=-0.6)
_S0 = 100.0
_TAU = 1.0
_STRIKES = torch.tensor([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0], dtype=torch.float64)


def test_synthetic_parameter_recovery() -> None:
    taus = (0.25, 1.0, 3.0)
    market = torch.stack([price_surface(_TRUE.as_tensors(), _S0, _STRIKES, tau) for tau in taus])
    initial = HestonParams(v0=0.09, kappa=1.0, theta=0.02, xi=0.7, rho=-0.2)
    result = calibrate_heston(
        market, _S0, _STRIKES, taus, initial, CalibrationConfig(n_iterations=800, lr=5e-2)
    )
    assert result.final_loss < result.losses[0] * 1e-4
    for tau, row in zip(taus, market, strict=True):
        refit = price_surface(result.params.as_tensors(), _S0, _STRIKES, tau)
        assert torch.allclose(refit, row, atol=2e-3)
    assert abs(result.params.v0 - _TRUE.v0) < 0.005
    assert abs(result.params.theta - _TRUE.theta) < 0.01
    assert abs(result.params.rho - _TRUE.rho) < 0.05
    assert abs(result.params.xi - _TRUE.xi) < 0.08
    assert abs(result.params.kappa - _TRUE.kappa) < 0.5


def test_analytic_pricer_agrees_with_monte_carlo() -> None:
    sim = HestonSimulator(
        s0=_S0, v0=0.045, kappa=2.0, theta=0.05, xi=0.4, rho=-0.6, maturity=_TAU, n_steps=200
    )
    payoff = EuropeanCall(strike=100.0)
    analytic = HestonAnalyticPricer().price(payoff, sim)
    sampled = MonteCarloPricer(n_paths=400_000, seed=89).price(payoff, sim)
    assert analytic.standard_error == 0.0
    assert analytic.provenance == "heston-cos"
    assert abs(analytic.value - sampled.value) < max(3.0 * sampled.standard_error, 0.03)


def test_analytic_pricer_rejects_out_of_scope() -> None:
    from deephedging.market import GBMSimulator

    heston = HestonSimulator(
        s0=_S0, v0=0.045, kappa=2.0, theta=0.05, xi=0.4, rho=-0.6, maturity=_TAU, n_steps=10
    )
    with pytest.raises(TypeError):
        HestonAnalyticPricer().price(EuropeanPut(strike=100.0), heston)
    with pytest.raises(TypeError):
        HestonAnalyticPricer().price(
            EuropeanCall(strike=100.0),
            GBMSimulator(s0=_S0, sigma=0.2, maturity=_TAU, n_steps=10),
        )
