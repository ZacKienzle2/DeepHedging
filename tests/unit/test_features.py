"""Tests for feature maps."""

import torch

from deephedging.features import DefaultFeatures, RunningMaxFeatures
from deephedging.frictions import NoCost
from deephedging.instruments import EuropeanCall
from deephedging.market import GBMSimulator
from deephedging.policies import FeedForwardPolicy
from deephedging.training import hedge_pnl


def _paths(n_paths: int = 256, n_steps: int = 12, seed: int = 37) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sim = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=n_steps)
    return sim.simulate(n_paths, generator=generator)


def test_default_map_matches_implicit_default() -> None:
    torch.manual_seed(41)
    policy = FeedForwardPolicy(hidden_sizes=(8,))
    paths = _paths()
    payoff = EuropeanCall(strike=100.0)
    implicit = hedge_pnl(paths, policy, payoff, NoCost())
    explicit = hedge_pnl(paths, policy, payoff, NoCost(), feature_map=DefaultFeatures())
    assert torch.equal(implicit, explicit)


def test_default_features_shape_and_values() -> None:
    paths = _paths()
    position = paths.new_full((paths.shape[1],), 0.3)
    tau = paths.new_tensor(0.5)
    features = DefaultFeatures()(paths, 4, tau, position)
    assert features.shape == (paths.shape[1], 3)
    assert torch.allclose(features[:, 0], torch.log(paths[4] / paths[0]))
    assert torch.all(features[:, 1] == 0.5)
    assert torch.all(features[:, 2] == 0.3)


def test_running_max_features_monotone_and_adapted() -> None:
    paths = _paths()
    position = paths.new_zeros(paths.shape[1])
    tau = paths.new_tensor(0.5)
    feature_map = RunningMaxFeatures()
    assert feature_map.n_features == 4
    previous = None
    for t in range(paths.shape[0]):
        features = feature_map(paths, t, tau, position)
        running_max = features[:, 3]
        assert torch.all(running_max >= torch.log(paths[t] / paths[0]) - 1e-6)
        if previous is not None:
            assert torch.all(running_max >= previous - 1e-6)
        previous = running_max


def test_custom_map_drives_wider_policy() -> None:
    torch.manual_seed(43)
    policy = FeedForwardPolicy(n_features=4, hidden_sizes=(8,))
    paths = _paths()
    payoff = EuropeanCall(strike=100.0)
    pnl = hedge_pnl(paths, policy, payoff, NoCost(), feature_map=RunningMaxFeatures())
    assert pnl.shape == (paths.shape[1],)
    pnl.mean().backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads
