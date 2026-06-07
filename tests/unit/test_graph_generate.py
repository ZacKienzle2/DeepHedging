"""Tests for in-graph batch generation.

The offset kernels read the shifted Philox subsequence from device
memory at launch, so a captured training graph redraws a fresh batch on
every replay by updating one element in place. These tests pin the
offset kernels bitwise against the host-argument kernels, the captured
training loop against eager training on the same streams, the freshness
of replayed batches, and the guards that keep the mode away from
configurations where a frozen offset would silently train on one batch
forever.
"""

import pytest
import torch

from deephedging import (
    CVaR,
    EuropeanCall,
    FeedForwardPolicy,
    GBMSimulator,
    NoiseSpec,
    ProportionalCost,
    TrainConfig,
    train,
)

_SIGMA = 0.2
_MATURITY = 30 / 365
_STRIKE = 100.0


def _require_kernels() -> None:
    from deephedging.market import kernels_available

    if not kernels_available():
        pytest.skip("CUDA kernel toolchain unavailable")


@pytest.mark.gpu
def test_offset_kernels_match_host_argument_kernels_bitwise() -> None:
    _require_kernels()
    from deephedging.market import CudaGBMSimulator, CudaHestonSimulator

    gbm = CudaGBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=30)
    heston = CudaHestonSimulator(
        s0=100.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, maturity=0.25, n_steps=30
    )
    for sim in (gbm, heston):
        scalar = sim.simulate(4096, noise=NoiseSpec(seed=7, stream=5))
        offset = torch.tensor([5 << 32], dtype=torch.int64, device="cuda")
        via_offset = sim.simulate_with_offset(4096, seed=7, offset=offset)
        assert torch.equal(scalar.spot, via_offset.spot)
        offset.fill_(9 << 32)
        redrawn = sim.simulate_with_offset(4096, seed=7, offset=offset)
        named = sim.simulate(4096, noise=NoiseSpec(seed=7, stream=9))
        assert torch.equal(redrawn.spot, named.spot)


@pytest.mark.gpu
def test_generated_graph_matches_eager_training() -> None:
    _require_kernels()
    from deephedging.market import CudaGBMSimulator

    def run(graphed: bool) -> list[float]:
        torch.manual_seed(48)
        sim = CudaGBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=8)
        policy = FeedForwardPolicy(hidden_sizes=(8, 8)).to("cuda")
        config = TrainConfig(
            n_iterations=5,
            batch_paths=4096,
            seed=14,
            graph_episode=graphed,
            graph_generate=graphed,
        )
        return train(
            sim,
            policy,
            EuropeanCall(strike=_STRIKE),
            ProportionalCost(rate=1e-3),
            CVaR(alpha=0.9),
            config,
        ).losses

    eager, generated = run(False), run(True)
    assert abs(eager[0] - generated[0]) < 1e-6
    assert abs(eager[1] - generated[1]) < 1e-6
    assert all(abs(a - b) < 1e-2 for a, b in zip(eager, generated, strict=True))


@pytest.mark.gpu
def test_replays_draw_fresh_batches() -> None:
    _require_kernels()
    from deephedging.market import CudaGBMSimulator

    torch.manual_seed(49)
    sim = CudaGBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=8)
    policy = FeedForwardPolicy(hidden_sizes=(8, 8)).to("cuda")
    config = TrainConfig(
        n_iterations=6,
        batch_paths=4096,
        seed=15,
        lr=0.0,
        graph_episode=True,
        graph_generate=True,
    )
    losses = train(
        sim,
        policy,
        EuropeanCall(strike=_STRIKE),
        ProportionalCost(rate=1e-3),
        CVaR(alpha=0.9),
        config,
    ).losses
    assert len(set(losses)) > 1


def test_graph_generate_requires_graph_episode() -> None:
    with pytest.raises(ValueError, match="graph_episode"):
        TrainConfig(n_iterations=1, batch_paths=8, seed=1, graph_generate=True)


def test_graph_generate_requires_a_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        TrainConfig(n_iterations=1, batch_paths=8, graph_episode=True, graph_generate=True)


@pytest.mark.gpu
def test_graph_generate_rejects_simulators_without_offsets() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    sim = GBMSimulator(s0=100.0, sigma=_SIGMA, maturity=_MATURITY, n_steps=5, device="cuda")
    policy = FeedForwardPolicy(hidden_sizes=(4,)).to("cuda")
    config = TrainConfig(
        n_iterations=1, batch_paths=8, seed=1, graph_episode=True, graph_generate=True
    )
    with pytest.raises(ValueError, match="simulate_with_offset"):
        train(
            sim,
            policy,
            EuropeanCall(strike=_STRIKE),
            ProportionalCost(rate=1e-3),
            CVaR(alpha=0.9),
            config,
        )
