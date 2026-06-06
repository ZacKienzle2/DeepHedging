"""Tests for feature maps."""

import pytest
import torch

from deephedging.features import DefaultFeatures, RunningMaxFeatures, VarianceFeatures
from deephedging.frictions import NoCost
from deephedging.instruments import EuropeanCall
from deephedging.market import GBMSimulator, HestonSimulator, MarketState, NoiseSpec
from deephedging.policies import FeedForwardPolicy
from deephedging.training import hedge_pnl


def _state(n_paths: int = 256, n_steps: int = 12, seed: int = 37) -> MarketState:
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=n_steps)
    return sim.simulate(n_paths, noise=NoiseSpec(seed=seed))


def test_default_map_matches_implicit_default() -> None:
    torch.manual_seed(41)
    policy = FeedForwardPolicy(hidden_sizes=(8,))
    state = _state()
    payoff = EuropeanCall(strike=100.0)
    implicit = hedge_pnl(state, policy, payoff, NoCost())
    explicit = hedge_pnl(state, policy, payoff, NoCost(), feature_map=DefaultFeatures())
    assert torch.equal(implicit, explicit)


def test_default_features_shape_and_values() -> None:
    state = _state()
    position = state.spot.new_full((state.n_paths,), 0.3)
    tau = state.spot.new_tensor(0.5)
    features = DefaultFeatures()(state, 4, tau, position)
    assert features.shape == (state.n_paths, 3)
    assert torch.allclose(features[:, 0], torch.log(state.spot[4] / state.spot[0]), atol=1e-6)
    assert torch.all(features[:, 1] == 0.5)
    assert torch.all(features[:, 2] == 0.3)


def test_running_max_features_monotone_and_adapted() -> None:
    state = _state()
    position = state.spot.new_zeros(state.n_paths)
    tau = state.spot.new_tensor(0.5)
    feature_map = RunningMaxFeatures()
    assert feature_map.n_features == 4
    previous = None
    for t in range(state.n_steps + 1):
        features = feature_map(state, t, tau, position)
        running_max = features[:, 3]
        assert torch.all(running_max >= torch.log(state.spot[t] / state.spot[0]) - 1e-6)
        if previous is not None:
            assert torch.all(running_max >= previous - 1e-6)
        previous = running_max


def test_variance_features_read_heston_channel() -> None:
    sim = HestonSimulator(
        s0=100.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, maturity=1.0, n_steps=12
    )
    state = sim.simulate(64, noise=NoiseSpec(seed=23))
    position = state.spot.new_zeros(state.n_paths)
    tau = state.spot.new_tensor(0.5)
    feature_map = VarianceFeatures()
    assert feature_map.n_features == 4
    features = feature_map(state, 5, tau, position)
    assert torch.equal(features[:, 3], state.aux["variance"][5])


def test_variance_features_require_variance_channel() -> None:
    state = _state()
    position = state.spot.new_zeros(state.n_paths)
    tau = state.spot.new_tensor(0.5)
    with pytest.raises(KeyError):
        VarianceFeatures()(state, 0, tau, position)


def test_custom_map_drives_wider_policy() -> None:
    torch.manual_seed(43)
    policy = FeedForwardPolicy(n_features=4, hidden_sizes=(8,))
    state = _state()
    payoff = EuropeanCall(strike=100.0)
    pnl = hedge_pnl(state, policy, payoff, NoCost(), feature_map=RunningMaxFeatures())
    assert pnl.shape == (state.n_paths,)
    pnl.mean().backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads
