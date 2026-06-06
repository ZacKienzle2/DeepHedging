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


def test_multi_asset_loop_matches_vectorised_oracle() -> None:
    from deephedging import MultiAssetFeatures
    from deephedging.policies.base import HedgePolicy
    from deephedging.training import hedge_pnl

    class ConstantVectorPolicy(HedgePolicy):
        def __init__(self, values: tuple[float, ...]) -> None:
            super().__init__()
            self.values = values

        def forward(
            self, features: torch.Tensor, state: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            row = features.new_tensor(self.values)
            return row.expand(features.shape[0], len(self.values)), None

    sim = _simulator(n_steps=12)
    state = sim.simulate(256, noise=NoiseSpec(seed=61))
    payoff = GeometricBasketCall(strike=100.0)
    cost = ProportionalCost(rate=1e-3)
    values = (0.4, -0.2, 0.7)
    pnl_loop = hedge_pnl(
        state,
        ConstantVectorPolicy(values),
        payoff,
        cost,
        premium=2.0,
        liquidate_terminal=True,
        feature_map=MultiAssetFeatures(n_assets=3),
    )
    positions = state.spot.new_tensor(values).expand(state.n_steps, state.n_paths, 3)
    pnl_vec = pnl_from_positions(
        state, positions, payoff, cost, premium=2.0, liquidate_terminal=True
    )
    assert torch.allclose(pnl_loop, pnl_vec, atol=1e-5)


@pytest.mark.slow
def test_multi_asset_training_beats_no_hedge() -> None:
    from deephedging import CVaR, MultiAssetFeatures, TrainConfig, train
    from deephedging.evaluation import expected_shortfall
    from deephedging.frictions import NoCost
    from deephedging.policies import FeedForwardPolicy
    from deephedging.training import hedge_pnl

    torch.manual_seed(67)
    sim = CorrelatedGBMSimulator(
        s0=100.0,
        sigmas=(0.2, 0.3),
        correlation=((1.0, 0.5), (0.5, 1.0)),
        maturity=0.25,
        n_steps=8,
    )
    payoff = GeometricBasketCall(strike=100.0)
    premium = MonteCarloPricer(n_paths=200_000, seed=71).price(payoff, sim).value
    feature_map = MultiAssetFeatures(n_assets=2)
    policy = FeedForwardPolicy(
        n_features=feature_map.n_features, hidden_sizes=(16, 16), n_outputs=2
    )
    config = TrainConfig(n_iterations=250, batch_paths=1024, lr=2e-3, seed=9)
    train(
        sim,
        policy,
        payoff,
        NoCost(),
        CVaR(alpha=0.9),
        config,
        premium=premium,
        feature_map=feature_map,
    )

    eval_state = sim.simulate(50_000, noise=NoiseSpec(seed=73))
    with torch.no_grad():
        hedged = hedge_pnl(
            eval_state, policy, payoff, NoCost(), premium=premium, feature_map=feature_map
        )
    unhedged = premium - payoff(eval_state.spot)
    assert float(expected_shortfall(hedged, alpha=0.9)) < 0.6 * float(
        expected_shortfall(unhedged, alpha=0.9)
    )


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
