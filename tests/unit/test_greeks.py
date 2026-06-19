"""Tests for autodiff Greeks against Black-Scholes closed forms."""

from functools import partial

import torch

from deephedging.evaluation import bs_call_delta, bs_call_vega, european_greeks


def _call(
    strike: float,
    spot: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
    rate: torch.Tensor,
) -> torch.Tensor:
    d1 = (torch.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    return spot * torch.special.ndtr(d1) - strike * torch.exp(-rate * tau) * torch.special.ndtr(d2)


def _put(
    strike: float,
    spot: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
    rate: torch.Tensor,
) -> torch.Tensor:
    return _call(strike, spot, sigma, tau, rate) - spot + strike * torch.exp(-rate * tau)


def test_autodiff_delta_and_vega_match_closed_form() -> None:
    strike = 100.0
    spot = torch.linspace(80.0, 120.0, 9, dtype=torch.float64)
    sigma, tau, rate = 0.2, 0.5, 0.01
    greeks = european_greeks(partial(_call, strike), spot, sigma, tau, rate)
    assert torch.allclose(greeks.delta, bs_call_delta(spot, strike, sigma, tau, rate), atol=1e-8)
    assert torch.allclose(greeks.vega, bs_call_vega(spot, strike, sigma, tau, rate), atol=1e-6)


def test_gamma_is_positive_and_matches_finite_difference() -> None:
    strike = 100.0
    spot = torch.tensor([90.0, 100.0, 110.0], dtype=torch.float64)
    sigma, tau, rate = 0.25, 0.5, 0.0
    greeks = european_greeks(partial(_call, strike), spot, sigma, tau, rate)
    assert bool(torch.all(greeks.gamma > 0.0))
    bump = 1e-3
    finite_difference = (
        bs_call_delta(spot + bump, strike, sigma, tau, rate)
        - bs_call_delta(spot - bump, strike, sigma, tau, rate)
    ) / (2.0 * bump)
    assert torch.allclose(greeks.gamma, finite_difference, atol=1e-4)


def test_theta_is_negative_and_rho_is_positive_for_a_call() -> None:
    greeks = european_greeks(partial(_call, 100.0), 100.0, 0.2, 0.5, 0.02)
    assert float(greeks.theta) < 0.0
    assert float(greeks.rho) > 0.0


def test_put_delta_follows_put_call_parity() -> None:
    strike = 100.0
    spot = torch.linspace(80.0, 120.0, 5, dtype=torch.float64)
    greeks = european_greeks(partial(_put, strike), spot, 0.2, 0.5, 0.01)
    expected = bs_call_delta(spot, strike, 0.2, 0.5, 0.01) - 1.0
    assert torch.allclose(greeks.delta, expected, atol=1e-8)


def test_scalar_inputs_return_scalar_greeks() -> None:
    strike = 100.0
    greeks = european_greeks(partial(_call, strike), 100.0, 0.2, 0.5, 0.0)
    assert greeks.delta.shape == torch.Size([])
    assert abs(float(greeks.delta) - float(bs_call_delta(100.0, strike, 0.2, 0.5, 0.0))) < 1e-8
