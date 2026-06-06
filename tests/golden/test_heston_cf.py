"""Golden tests for the COS Heston pricer and implied volatility."""

import math

import pytest
import torch

from deephedging.calibration import (
    HestonParams,
    cos_call_price,
    heston_cf,
    heston_cumulants,
    implied_vol,
)
from deephedging.evaluation import bs_call_price
from deephedging.instruments import EuropeanCall
from deephedging.market import HestonSimulator
from deephedging.pricing import MonteCarloPricer

_PARAMS = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
_TAU = 1.0
_S0 = 100.0


def _price(
    strikes: torch.Tensor, params: HestonParams = _PARAMS, tau: float = _TAU
) -> torch.Tensor:
    tensors = params.as_tensors()
    c1, c2 = heston_cumulants(tau, *tensors)
    width = 12.0 * math.sqrt(max(c2, 1e-12))
    a, b = c1 - width, c1 + width

    def cf(u: torch.Tensor) -> torch.Tensor:
        return heston_cf(u, tau, *tensors)

    return cos_call_price(cf, _S0, strikes, a, b)


def test_martingale_property() -> None:
    tensors = _PARAMS.as_tensors()
    at_zero = heston_cf(torch.tensor([0.0], dtype=torch.float64), _TAU, *tensors)
    assert torch.allclose(at_zero, torch.ones_like(at_zero))
    at_minus_i = heston_cf(torch.tensor([-1j], dtype=torch.complex128), _TAU, *tensors)
    assert torch.allclose(at_minus_i.real, torch.ones(1, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(at_minus_i.imag, torch.zeros(1, dtype=torch.float64), atol=1e-12)


def test_cos_price_matches_monte_carlo() -> None:
    strikes = torch.tensor([80.0, 90.0, 100.0, 110.0, 120.0], dtype=torch.float64)
    cos_prices = _price(strikes)
    sim = HestonSimulator(
        s0=_S0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, maturity=_TAU, n_steps=200
    )
    for strike, cos_value in zip(strikes.tolist(), cos_prices.tolist(), strict=True):
        estimate = MonteCarloPricer(n_paths=400_000, seed=83).price(
            EuropeanCall(strike=strike), sim
        )
        assert abs(cos_value - estimate.value) < max(3.0 * estimate.standard_error, 0.03)


def test_degenerate_limit_recovers_black_scholes() -> None:
    flat = HestonParams(v0=0.04, kappa=5.0, theta=0.04, xi=1e-4, rho=0.0)
    strikes = torch.tensor([85.0, 100.0, 115.0], dtype=torch.float64)
    cos_prices = _price(strikes, params=flat)
    for strike, value in zip(strikes.tolist(), cos_prices.tolist(), strict=True):
        reference = float(bs_call_price(_S0, strike, 0.2, _TAU))
        assert abs(value - reference) < 1e-3


def test_price_monotone_and_convergent() -> None:
    strikes = torch.tensor([80.0, 90.0, 100.0, 110.0, 120.0], dtype=torch.float64)
    prices = _price(strikes)
    assert torch.all(prices[1:] < prices[:-1])
    tensors = _PARAMS.as_tensors()
    c1, c2 = heston_cumulants(_TAU, *tensors)
    width = 12.0 * math.sqrt(c2)

    def cf(u: torch.Tensor) -> torch.Tensor:
        return heston_cf(u, _TAU, *tensors)

    finer = cos_call_price(cf, _S0, strikes, c1 - width, c1 + width, n_terms=320)
    assert torch.allclose(prices, finer, atol=1e-7)


def test_cf_continuity_no_branch_jump() -> None:
    tensors = _PARAMS.as_tensors()
    u = torch.linspace(0.0, 200.0, 4001, dtype=torch.float64)
    values = heston_cf(u, 5.0, *tensors)
    angles = torch.angle(values)
    steps = angles.diff()
    wrapped_steps = torch.remainder(steps + math.pi, 2.0 * math.pi) - math.pi
    assert float(wrapped_steps.abs().max()) < 0.5


def test_cf_autograd_gradients_finite() -> None:
    values = [0.04, 1.5, 0.04, 0.5, -0.7]
    tensors = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in values]
    c1, c2 = heston_cumulants(_TAU, *[t.detach() for t in tensors])
    width = 12.0 * math.sqrt(c2)

    def cf(u: torch.Tensor) -> torch.Tensor:
        return heston_cf(u, _TAU, *tensors)

    price = cos_call_price(
        cf, _S0, torch.tensor([100.0], dtype=torch.float64), c1 - width, c1 + width
    )
    price.sum().backward()
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad)


def test_implied_vol_round_trip() -> None:
    strikes = torch.tensor([70.0, 85.0, 100.0, 115.0, 130.0], dtype=torch.float64)
    sigmas = torch.tensor([0.15, 0.2, 0.25, 0.3, 0.35], dtype=torch.float64)
    sqrt_tau = math.sqrt(_TAU)
    d1 = (torch.log(_S0 / strikes) + 0.5 * sigmas**2 * _TAU) / (sigmas * sqrt_tau)
    d2 = d1 - sigmas * sqrt_tau
    prices = _S0 * torch.special.ndtr(d1) - strikes * torch.special.ndtr(d2)
    recovered = implied_vol(prices, _S0, strikes, _TAU)
    assert torch.allclose(recovered, sigmas, atol=1e-8)


def test_implied_vol_rejects_arbitrage_violations() -> None:
    strikes = torch.tensor([100.0, 100.0], dtype=torch.float64)
    prices = torch.tensor([-1.0, 150.0], dtype=torch.float64)
    result = implied_vol(prices, _S0, strikes, _TAU)
    assert torch.isnan(result).all()


def test_invalid_params_rejected() -> None:
    with pytest.raises(ValueError):
        HestonParams(v0=0.04, kappa=0.0, theta=0.04, xi=0.5, rho=-0.7)
    with pytest.raises(ValueError):
        HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-1.0)
