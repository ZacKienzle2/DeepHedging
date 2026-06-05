"""Golden tests for the Black-Scholes closed forms.

Reference values are analytic, not simulated, so tolerances are tight.
"""

import torch

from deephedging.evaluation import bs_call_delta, bs_call_price, bs_put_price


def test_atm_call_price_zero_rate() -> None:
    price = bs_call_price(spot=100.0, strike=100.0, sigma=0.2, tau=1.0)
    assert abs(float(price) - 7.965567) < 1e-4


def test_atm_call_price_with_rate() -> None:
    price = bs_call_price(spot=100.0, strike=100.0, sigma=0.2, tau=1.0, rate=0.05)
    assert abs(float(price) - 10.450584) < 1e-4


def test_atm_call_delta_zero_rate() -> None:
    delta = bs_call_delta(spot=100.0, strike=100.0, sigma=0.2, tau=1.0)
    assert abs(float(delta) - 0.539828) < 1e-4


def test_put_call_parity() -> None:
    spots = torch.tensor([80.0, 95.0, 100.0, 110.0, 130.0], dtype=torch.float64)
    strike, sigma, tau, rate = 100.0, 0.25, 0.7, 0.03
    call = bs_call_price(spots, strike, sigma, tau, rate)
    put = bs_put_price(spots, strike, sigma, tau, rate)
    import math

    forward = spots - strike * math.exp(-rate * tau)
    assert torch.allclose(call - put, forward, atol=1e-10)


def test_deep_itm_delta_approaches_one() -> None:
    delta = bs_call_delta(spot=300.0, strike=100.0, sigma=0.2, tau=0.5)
    assert float(delta) > 0.999


def test_deep_otm_delta_approaches_zero() -> None:
    delta = bs_call_delta(spot=30.0, strike=100.0, sigma=0.2, tau=0.5)
    assert float(delta) < 1e-3
