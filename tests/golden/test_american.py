"""Golden tests for American put valuation."""

import pytest

from deephedging.evaluation import binomial_american_put, bs_put_price, lsm_american_put
from deephedging.market import GBMSimulator

_S0, _STRIKE, _SIGMA, _MATURITY, _RATE = 100.0, 100.0, 0.2, 1.0, 0.05


def test_binomial_converges_with_depth() -> None:
    coarse = binomial_american_put(_S0, _STRIKE, _SIGMA, _MATURITY, _RATE, n_steps=500)
    fine = binomial_american_put(_S0, _STRIKE, _SIGMA, _MATURITY, _RATE, n_steps=2000)
    assert abs(coarse - fine) < 5e-3


def test_american_dominates_european_with_positive_premium() -> None:
    american = binomial_american_put(_S0, _STRIKE, _SIGMA, _MATURITY, _RATE)
    european = float(bs_put_price(_S0, _STRIKE, _SIGMA, _MATURITY, rate=_RATE))
    assert american > european + 0.05


def test_zero_rate_collapses_to_european() -> None:
    american = binomial_american_put(_S0, _STRIKE, _SIGMA, _MATURITY, rate=0.0)
    european = float(bs_put_price(_S0, _STRIKE, _SIGMA, _MATURITY, rate=0.0))
    assert abs(american - european) < 5e-3


@pytest.mark.slow
def test_lsm_matches_binomial_tree() -> None:
    simulator = GBMSimulator(s0=_S0, sigma=_SIGMA, maturity=_MATURITY, n_steps=50, mu=_RATE)
    estimate = lsm_american_put(simulator, _STRIKE, _RATE, n_paths=400_000, seed=229)
    reference = binomial_american_put(_S0, _STRIKE, _SIGMA, _MATURITY, _RATE)
    assert estimate.provenance == "lsm"
    assert abs(estimate.value - reference) < max(4.0 * estimate.standard_error, 0.05)
    european = float(bs_put_price(_S0, _STRIKE, _SIGMA, _MATURITY, rate=_RATE))
    assert estimate.value > european


def test_lsm_rejects_drift_mismatch() -> None:
    simulator = GBMSimulator(s0=_S0, sigma=_SIGMA, maturity=_MATURITY, n_steps=10, mu=0.0)
    with pytest.raises(ValueError):
        lsm_american_put(simulator, _STRIKE, rate=_RATE, n_paths=64)
