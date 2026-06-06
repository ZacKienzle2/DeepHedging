"""Tests for the correlated multi-asset simulator and basket payoffs."""

import math

import pytest
import torch

from deephedging.frictions import ProportionalCost
from deephedging.instruments import BasketCall, GeometricBasketCall
from deephedging.market import CorrelatedGBMSimulator, GBMSimulator, MarketState, NoiseSpec
from deephedging.pricing import MonteCarloPricer
from deephedging.training import pnl_from_positions

_CORRELATION = (
    (1.0, 0.6, 0.3),
    (0.6, 1.0, 0.5),
    (0.3, 0.5, 1.0),
)
_SIGMAS = (0.2, 0.25, 0.3)


def _simulator(n_steps: int = 25) -> CorrelatedGBMSimulator:
    return CorrelatedGBMSimulator(
        s0=100.0, sigmas=_SIGMAS, correlation=_CORRELATION, maturity=1.0, n_steps=n_steps
    )


def test_shape_initial_value_and_replay() -> None:
    sim = _simulator()
    state = sim.simulate(512, noise=NoiseSpec(seed=41))
    assert state.spot.shape == (26, 512, 3)
    assert state.n_paths == 512
    assert torch.all(state.spot[0] == 100.0)
    replay = sim.simulate(512, noise=NoiseSpec(seed=41))
    assert torch.equal(state.spot, replay.spot)


def test_single_asset_matches_gbm_bitwise() -> None:
    multi = CorrelatedGBMSimulator(
        s0=100.0, sigmas=(0.2,), correlation=((1.0,),), maturity=1.0, n_steps=20
    )
    single = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=20)
    noise = NoiseSpec(seed=43)
    assert torch.equal(
        multi.simulate(256, noise=noise).spot.squeeze(-1), single.simulate(256, noise=noise).spot
    )


def test_empirical_correlation_recovers_target() -> None:
    state = _simulator(n_steps=1).simulate(400_000, noise=NoiseSpec(seed=47))
    log_returns = torch.log(state.spot[-1] / 100.0)
    centred = log_returns - log_returns.mean(dim=0, keepdim=True)
    covariance = centred.T @ centred / centred.shape[0]
    scale = covariance.diagonal().sqrt()
    correlation = covariance / (scale.unsqueeze(0) * scale.unsqueeze(1))
    target = torch.tensor(_CORRELATION)
    assert torch.allclose(correlation, target, atol=1e-2)


def test_geometric_basket_matches_closed_form() -> None:
    sigmas = torch.tensor(_SIGMAS, dtype=torch.float64)
    correlation = torch.tensor(_CORRELATION, dtype=torch.float64)
    covariance = correlation * sigmas.unsqueeze(0) * sigmas.unsqueeze(1)
    n_assets = sigmas.shape[0]
    maturity, strike, s0 = 1.0, 100.0, 100.0

    variance_g = float(covariance.sum()) / n_assets**2 * maturity
    drift_g = float((-0.5 * sigmas**2).mean()) * maturity
    forward = s0 * math.exp(drift_g + 0.5 * variance_g)
    sigma_g = math.sqrt(variance_g / maturity)
    d1 = (math.log(forward / strike) + 0.5 * sigma_g**2 * maturity) / (
        sigma_g * math.sqrt(maturity)
    )
    d2 = d1 - sigma_g * math.sqrt(maturity)
    normal = torch.distributions.Normal(0.0, 1.0)
    reference = forward * float(normal.cdf(torch.tensor(d1))) - strike * float(
        normal.cdf(torch.tensor(d2))
    )

    estimate = MonteCarloPricer(n_paths=400_000, seed=53).price(
        GeometricBasketCall(strike=strike), _simulator()
    )
    assert abs(estimate.value - reference) < 3.0 * estimate.standard_error


def test_arithmetic_dominates_geometric_pathwise() -> None:
    state = _simulator().simulate(50_000, noise=NoiseSpec(seed=59))
    arithmetic = BasketCall(strike=100.0, weights=(1 / 3, 1 / 3, 1 / 3))
    geometric = GeometricBasketCall(strike=100.0)
    assert torch.all(arithmetic(state.spot) >= geometric(state.spot) - 1e-5)


def test_pnl_from_positions_contracts_asset_axis() -> None:
    spot = torch.tensor(
        [
            [[100.0, 50.0], [100.0, 50.0]],
            [[110.0, 45.0], [90.0, 55.0]],
        ]
    )
    state = MarketState.from_spot(spot)
    positions = torch.tensor([[[1.0, 2.0], [0.5, -1.0]]])
    cost = ProportionalCost(rate=0.01)

    def zero_payoff(paths: torch.Tensor) -> torch.Tensor:
        return paths.new_zeros(paths.shape[1])

    pnl = pnl_from_positions(state, positions, zero_payoff, cost, premium=0.0)
    expected_gains = torch.tensor([1.0 * 10.0 + 2.0 * -5.0, 0.5 * -10.0 + -1.0 * 5.0])
    expected_costs = torch.tensor(
        [0.01 * (1.0 * 100.0 + 2.0 * 50.0), 0.01 * (0.5 * 100.0 + 1.0 * 50.0)]
    )
    assert torch.allclose(pnl, expected_gains - expected_costs, atol=1e-6)


def test_invalid_correlation_rejected() -> None:
    with pytest.raises(ValueError):
        CorrelatedGBMSimulator(
            s0=100.0, sigmas=(0.2, 0.3), correlation=((1.0, 0.5),), maturity=1.0, n_steps=5
        )
    with pytest.raises(ValueError):
        CorrelatedGBMSimulator(
            s0=100.0,
            sigmas=(0.2, 0.3),
            correlation=((1.0, 0.9), (0.5, 1.0)),
            maturity=1.0,
            n_steps=5,
        )
    with pytest.raises(ValueError):
        CorrelatedGBMSimulator(
            s0=100.0,
            sigmas=(0.2, 0.3),
            correlation=((1.0, 1.2), (1.2, 1.0)),
            maturity=1.0,
            n_steps=5,
        )
