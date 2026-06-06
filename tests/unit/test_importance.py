"""Tests for tail importance sampling."""

import math

import pytest
import torch

from deephedging.instruments import EuropeanCall
from deephedging.market import GBMSimulator, NoiseSpec, TiltedGBMSimulator
from deephedging.market.tilted import LOG_WEIGHT_CHANNEL
from deephedging.risk import CVaR

_SIGMA, _MATURITY, _STEPS = 0.2, 0.25, 20


def _tilted(tilt: float) -> TiltedGBMSimulator:
    return TiltedGBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=_STEPS, tilt=tilt)


def test_reweighted_martingale_mean() -> None:
    state = _tilted(0.4).simulate(400_000, noise=NoiseSpec(seed=191))
    weights = torch.exp(state.aux[LOG_WEIGHT_CHANNEL].to(torch.float64))
    terminal = state.terminal.to(torch.float64)
    weighted = weights * terminal
    standard_error = float(weighted.std()) / math.sqrt(weighted.shape[0])
    assert abs(float(weighted.mean()) - 100.0) < 4.0 * standard_error
    assert abs(float(weights.mean()) - 1.0) < 4.0 * float(weights.std()) / math.sqrt(
        weights.shape[0]
    )


def test_replay_and_zero_tilt_matches_plain_gbm() -> None:
    tilted = _tilted(0.0)
    plain = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=_STEPS)
    noise = NoiseSpec(seed=193)
    first = tilted.simulate(256, noise=noise)
    second = tilted.simulate(256, noise=noise)
    assert torch.equal(first.spot, second.spot)
    assert torch.equal(first.spot, plain.simulate(256, noise=noise).spot)
    assert torch.all(first.aux[LOG_WEIGHT_CHANNEL] == 0.0)


def test_weighted_cvar_objective_is_unbiased() -> None:
    payoff = EuropeanCall(strike=100.0)
    plain_state = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=_STEPS).simulate(
        400_000, noise=NoiseSpec(seed=197)
    )
    plain_loss = payoff(plain_state.spot).to(torch.float64)

    tilted_state = _tilted(0.4).simulate(400_000, noise=NoiseSpec(seed=199))
    tilted_loss = payoff(tilted_state.spot).to(torch.float64)
    weights = torch.exp(tilted_state.aux[LOG_WEIGHT_CHANNEL].to(torch.float64))

    measure = CVaR(alpha=0.95).double()
    measure.warm_start(plain_loss)
    with torch.no_grad():
        plain_value = float(measure(plain_loss))
        weighted_value = float(measure(tilted_loss, weights=weights))
    assert abs(plain_value - weighted_value) < 0.05 * plain_value


def test_tilt_enriches_the_tail() -> None:
    alpha = 0.99
    payoff = EuropeanCall(strike=100.0)
    plain_state = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=_STEPS).simulate(
        200_000, noise=NoiseSpec(seed=211)
    )
    plain_loss = payoff(plain_state.spot)
    threshold = torch.quantile(plain_loss.to(torch.float64), alpha)

    tilted_state = _tilted(0.4).simulate(200_000, noise=NoiseSpec(seed=211))
    tilted_loss = payoff(tilted_state.spot).to(torch.float64)
    plain_fraction = float((plain_loss.to(torch.float64) > threshold).double().mean())
    tilted_fraction = float((tilted_loss > threshold).double().mean())
    assert tilted_fraction > 5.0 * plain_fraction


def test_tilt_reduces_tail_estimator_variance() -> None:
    alpha = 0.99
    payoff = EuropeanCall(strike=100.0)
    plain_state = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=_STEPS).simulate(
        400_000, noise=NoiseSpec(seed=223)
    )
    plain_loss = payoff(plain_state.spot).to(torch.float64)
    threshold = torch.quantile(plain_loss, alpha)
    plain_excess = torch.relu(plain_loss - threshold)

    tilted_state = _tilted(0.4).simulate(400_000, noise=NoiseSpec(seed=227))
    tilted_loss = payoff(tilted_state.spot).to(torch.float64)
    weights = torch.exp(tilted_state.aux[LOG_WEIGHT_CHANNEL].to(torch.float64))
    weighted_excess = weights * torch.relu(tilted_loss - threshold)

    assert float(weighted_excess.var()) < 0.5 * float(plain_excess.var())
    assert abs(float(weighted_excess.mean()) - float(plain_excess.mean())) < 4.0 * (
        float(plain_excess.std()) / math.sqrt(400_000)
        + float(weighted_excess.std()) / math.sqrt(400_000)
    )


def test_invalid_tilt_rejected() -> None:
    with pytest.raises(ValueError):
        _tilted(1.5)
