"""Tests for the permutation-equivariant deep-set policy."""

import pytest
import torch

from deephedging.policies import DeepSetPolicy


def _features(n_paths: int, n_assets: int, seed: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    log_moneyness = 0.1 * torch.randn(n_paths, n_assets, generator=generator)
    tau = torch.rand(n_paths, 1, generator=generator)
    position = torch.randn(n_paths, n_assets, generator=generator)
    return torch.cat((log_moneyness, tau, position), dim=-1)


def test_output_shape_matches_asset_count() -> None:
    policy = DeepSetPolicy(n_assets=4)
    position, state = policy(_features(16, 4))
    assert position.shape == (16, 4)
    assert state is None


def test_policy_is_permutation_equivariant() -> None:
    torch.manual_seed(0)
    n_assets = 5
    policy = DeepSetPolicy(n_assets=n_assets)
    features = _features(32, n_assets)
    perm = torch.tensor([2, 0, 4, 1, 3])
    log_moneyness = features[:, :n_assets]
    tau = features[:, n_assets : n_assets + 1]
    position = features[:, n_assets + 1 :]
    permuted = torch.cat((log_moneyness[:, perm], tau, position[:, perm]), dim=-1)
    with torch.no_grad():
        base = policy(features)[0]
        permuted_out = policy(permuted)[0]
    assert torch.allclose(permuted_out, base[:, perm], atol=1e-5)


def test_parameter_count_is_independent_of_asset_count() -> None:
    small = sum(p.numel() for p in DeepSetPolicy(n_assets=2).parameters())
    large = sum(p.numel() for p in DeepSetPolicy(n_assets=50).parameters())
    assert small == large


def test_rejects_bad_asset_count_and_feature_width() -> None:
    with pytest.raises(ValueError):
        DeepSetPolicy(n_assets=0)
    policy = DeepSetPolicy(n_assets=3)
    with pytest.raises(ValueError):
        policy(torch.zeros(8, 6))


@pytest.mark.slow
def test_deep_set_policy_trains_a_multi_asset_hedge() -> None:
    from deephedging import CVaR, MultiAssetFeatures, TrainConfig, train
    from deephedging.evaluation import expected_shortfall
    from deephedging.frictions import NoCost
    from deephedging.instruments import GeometricBasketCall
    from deephedging.market import CorrelatedGBMSimulator, NoiseSpec
    from deephedging.pricing import MonteCarloPricer
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
    policy = DeepSetPolicy(n_assets=2, hidden_sizes=(16, 16), latent_size=16)
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
