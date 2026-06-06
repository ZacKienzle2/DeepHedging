"""Tests for the pricing seam."""

import pytest

from deephedging.instruments import EuropeanCall, EuropeanPut, UpAndOutCall
from deephedging.market import GBMSimulator, HestonSimulator
from deephedging.pricing import BlackScholesPricer, MonteCarloPricer, PriceEstimate


def _gbm() -> GBMSimulator:
    return GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=50)


def test_closed_form_and_monte_carlo_agree() -> None:
    payoff = EuropeanCall(strike=100.0)
    closed = BlackScholesPricer().price(payoff, _gbm())
    sampled = MonteCarloPricer(n_paths=400_000, seed=3).price(payoff, _gbm())
    assert closed.standard_error == 0.0
    assert sampled.standard_error > 0.0
    assert abs(closed.value - sampled.value) < 3.0 * sampled.standard_error


def test_put_closed_form_matches_monte_carlo() -> None:
    payoff = EuropeanPut(strike=105.0)
    closed = BlackScholesPricer().price(payoff, _gbm())
    sampled = MonteCarloPricer(n_paths=400_000, seed=5).price(payoff, _gbm())
    assert abs(closed.value - sampled.value) < 3.0 * sampled.standard_error


def test_estimate_carries_provenance() -> None:
    estimate = BlackScholesPricer().price(EuropeanCall(strike=100.0), _gbm())
    assert isinstance(estimate, PriceEstimate)
    assert estimate.provenance == "black-scholes"
    assert (
        MonteCarloPricer(n_paths=1000).price(EuropeanCall(strike=100.0), _gbm()).provenance
        == "monte-carlo"
    )


def test_closed_form_rejects_out_of_scope() -> None:
    heston = HestonSimulator(
        s0=100.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, maturity=1.0, n_steps=10
    )
    with pytest.raises(TypeError):
        BlackScholesPricer().price(EuropeanCall(strike=100.0), heston)
    with pytest.raises(TypeError):
        BlackScholesPricer().price(UpAndOutCall(strike=100.0, barrier=120.0), _gbm())


def test_monte_carlo_prices_any_payoff_under_any_model() -> None:
    heston = HestonSimulator(
        s0=100.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, maturity=1.0, n_steps=50
    )
    barrier = UpAndOutCall(strike=100.0, barrier=120.0)
    vanilla = EuropeanCall(strike=100.0)
    pricer = MonteCarloPricer(n_paths=100_000, seed=7)
    knocked = pricer.price(barrier, heston)
    plain = pricer.price(vanilla, heston)
    assert 0.0 < knocked.value < plain.value


def test_monte_carlo_reproducible_by_seed() -> None:
    pricer = MonteCarloPricer(n_paths=10_000, seed=11)
    first = pricer.price(EuropeanCall(strike=100.0), _gbm())
    second = pricer.price(EuropeanCall(strike=100.0), _gbm())
    assert first == second
