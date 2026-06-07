"""Tests for the path-regenerating backward.

Every simulator replays bitwise from a noise spec, so a backward that
regenerates the path grid instead of storing it satisfies the recompute
contract of gradient checkpointing exactly. These tests pin the loss
and gradient parity against the storing path across plain, nested,
weighted, and auxiliary-channel training, plus the guards that keep the
flag away from configurations where regeneration would silently train
on wrong paths.
"""

import pytest
import torch

from deephedging.features import VarianceFeatures
from deephedging.frictions import ProportionalCost
from deephedging.instruments import EuropeanCall
from deephedging.market import GBMSimulator, HestonSimulator, TiltedGBMSimulator
from deephedging.policies import FeedForwardPolicy
from deephedging.risk import CVaR
from deephedging.training import TrainConfig, train

_SIGMA = 0.2
_MATURITY = 30 / 365
_STRIKE = 100.0


def _losses_and_grads(
    regenerate: bool, **config_overrides: object
) -> tuple[list[float], list[torch.Tensor]]:
    torch.manual_seed(42)
    sim = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=8)
    policy = FeedForwardPolicy(hidden_sizes=(8, 8))
    config = TrainConfig(
        n_iterations=4,
        batch_paths=512,
        seed=8,
        regenerate_paths=regenerate,
        **config_overrides,  # type: ignore[arg-type]
    )
    result = train(
        sim,
        policy,
        EuropeanCall(strike=_STRIKE),
        ProportionalCost(rate=1e-3),
        CVaR(alpha=0.9),
        config,
    )
    grads = [p.grad.clone() for p in policy.parameters() if p.grad is not None]
    return result.losses, grads


def test_regenerated_losses_match_stored_bitwise() -> None:
    stored_losses, stored_grads = _losses_and_grads(regenerate=False)
    regen_losses, regen_grads = _losses_and_grads(regenerate=True)
    assert stored_losses == regen_losses
    for stored, regenerated in zip(stored_grads, regen_grads, strict=True):
        assert torch.allclose(stored, regenerated, atol=1e-7)


def test_regeneration_nests_with_step_checkpointing() -> None:
    stored_losses, _ = _losses_and_grads(regenerate=False)
    nested_losses, _ = _losses_and_grads(regenerate=True, checkpoint_steps=True)
    assert all(abs(a - b) < 1e-6 for a, b in zip(stored_losses, nested_losses, strict=True))


def test_regeneration_reproduces_importance_weights() -> None:
    def run(regenerate: bool) -> list[float]:
        torch.manual_seed(43)
        sim = TiltedGBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=6, tilt=0.4)
        policy = FeedForwardPolicy(hidden_sizes=(8,))
        config = TrainConfig(n_iterations=3, batch_paths=512, seed=9, regenerate_paths=regenerate)
        return train(
            sim,
            policy,
            EuropeanCall(strike=_STRIKE),
            ProportionalCost(rate=1e-3),
            CVaR(alpha=0.95),
            config,
        ).losses

    assert run(False) == run(True)


def test_regeneration_reproduces_aux_channels() -> None:
    def run(regenerate: bool) -> list[float]:
        torch.manual_seed(44)
        sim = HestonSimulator(
            s0=100.0,
            v0=0.04,
            kappa=1.5,
            theta=0.04,
            xi=0.5,
            rho=-0.7,
            maturity=_MATURITY,
            n_steps=6,
        )
        policy = FeedForwardPolicy(n_features=4, hidden_sizes=(8,))
        config = TrainConfig(n_iterations=3, batch_paths=512, seed=10, regenerate_paths=regenerate)
        return train(
            sim,
            policy,
            EuropeanCall(strike=_STRIKE),
            ProportionalCost(rate=1e-3),
            CVaR(alpha=0.9),
            config,
            feature_map=VarianceFeatures(),
        ).losses

    assert run(False) == run(True)


@pytest.mark.gpu
def test_regeneration_matches_on_the_cuda_backend() -> None:
    from deephedging.market import CudaGBMSimulator, kernels_available

    if not kernels_available():
        pytest.skip("CUDA kernel toolchain unavailable")

    def run(regenerate: bool) -> list[float]:
        torch.manual_seed(45)
        sim = CudaGBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=8)
        policy = FeedForwardPolicy(hidden_sizes=(8, 8)).to("cuda")
        config = TrainConfig(n_iterations=3, batch_paths=4096, seed=11, regenerate_paths=regenerate)
        return train(
            sim,
            policy,
            EuropeanCall(strike=_STRIKE),
            ProportionalCost(rate=1e-3),
            CVaR(alpha=0.9),
            config,
        ).losses

    stored, regenerated = run(False), run(True)
    assert all(abs(a - b) < 1e-6 for a, b in zip(stored, regenerated, strict=True))


@pytest.mark.gpu
@pytest.mark.slow
def test_regeneration_cuts_peak_memory_at_scale() -> None:
    from deephedging.market import CudaGBMSimulator, kernels_available

    if not kernels_available():
        pytest.skip("CUDA kernel toolchain unavailable")

    def peak(regenerate: bool) -> int:
        torch.manual_seed(46)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sim = CudaGBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=30)
        policy = FeedForwardPolicy(hidden_sizes=(64, 64)).to("cuda")
        config = TrainConfig(
            n_iterations=2,
            batch_paths=262_144,
            seed=12,
            regenerate_paths=regenerate,
            checkpoint_steps=regenerate,
        )
        train(
            sim,
            policy,
            EuropeanCall(strike=_STRIKE),
            ProportionalCost(rate=1e-3),
            CVaR(alpha=0.95),
            config,
        )
        return torch.cuda.max_memory_allocated()

    stored, regenerated = peak(False), peak(True)
    assert regenerated < 0.5 * stored


def test_evaluation_never_routes_through_regeneration() -> None:
    torch.manual_seed(47)
    sim = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=6)
    policy = FeedForwardPolicy(hidden_sizes=(8,))
    config = TrainConfig(n_iterations=2, batch_paths=256, seed=13, regenerate_paths=True)
    train(
        sim,
        policy,
        EuropeanCall(strike=_STRIKE),
        ProportionalCost(rate=1e-3),
        CVaR(alpha=0.9),
        config,
    )
    from deephedging.market import NoiseSpec
    from deephedging.training import hedge_pnl

    state = sim.simulate(1024, noise=NoiseSpec(seed=99))
    with torch.no_grad():
        pnl = hedge_pnl(state, policy, EuropeanCall(strike=_STRIKE), ProportionalCost(rate=1e-3))
    assert pnl.shape == (1024,)
    assert torch.isfinite(pnl).all()


def test_regeneration_requires_a_seed() -> None:
    sim = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=5)
    policy = FeedForwardPolicy(hidden_sizes=(8,))
    config = TrainConfig(n_iterations=1, batch_paths=8, regenerate_paths=True)
    with pytest.raises(ValueError, match="seed"):
        train(
            sim,
            policy,
            EuropeanCall(strike=_STRIKE),
            ProportionalCost(rate=1e-3),
            CVaR(alpha=0.9),
            config,
        )


def test_regeneration_excludes_graph_capture() -> None:
    sim = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=5)
    policy = FeedForwardPolicy(hidden_sizes=(8,))
    config = TrainConfig(
        n_iterations=1, batch_paths=8, seed=1, regenerate_paths=True, graph_episode=True
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        train(
            sim,
            policy,
            EuropeanCall(strike=_STRIKE),
            ProportionalCost(rate=1e-3),
            CVaR(alpha=0.9),
            config,
        )
