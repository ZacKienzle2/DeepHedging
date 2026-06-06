"""Parity tests for the fused CUDA path generators.

Parity against the eager simulators is distributional, never bitwise;
the backends use different RNG algorithms by design. Replay within the
CUDA backend is exact.
"""

import math

import pytest
import torch

from deephedging.market import (
    CudaGBMSimulator,
    CudaHestonSimulator,
    GBMSimulator,
    NoiseSpec,
    kernels_available,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not kernels_available(), reason="CUDA kernels unavailable"),
]


def _cuda_gbm() -> CudaGBMSimulator:
    return CudaGBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=50)


def _cuda_heston() -> CudaHestonSimulator:
    return CudaHestonSimulator(
        s0=100.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, maturity=1.0, n_steps=100
    )


def test_gbm_shape_initial_value_and_replay() -> None:
    sim = _cuda_gbm()
    state = sim.simulate(4096, noise=NoiseSpec(seed=7))
    assert state.spot.shape == (51, 4096)
    assert state.spot.device.type == "cuda"
    assert torch.all(state.spot[0] == 100.0)
    replay = sim.simulate(4096, noise=NoiseSpec(seed=7))
    assert torch.equal(state.spot, replay.spot)
    other = sim.simulate(4096, noise=NoiseSpec(seed=7).child(1))
    assert not torch.equal(state.spot, other.spot)


def test_gbm_matches_analytic_log_moments() -> None:
    sim = _cuda_gbm()
    state = sim.simulate(400_000, noise=NoiseSpec(seed=11))
    log_returns = torch.log(state.terminal / 100.0)
    assert abs(float(log_returns.mean()) - (-0.5 * 0.2**2)) < 5e-3
    assert abs(float(log_returns.std()) - 0.2) < 5e-3


def test_gbm_distribution_matches_eager() -> None:
    cuda_state = _cuda_gbm().simulate(400_000, noise=NoiseSpec(seed=13))
    eager = GBMSimulator(s0=100.0, sigma=0.2, maturity=1.0, n_steps=50, device="cuda")
    eager_state = eager.simulate(400_000, noise=NoiseSpec(seed=13))
    for t in (10, 25, 50):
        a = torch.log(cuda_state.spot[t] / 100.0)
        b = torch.log(eager_state.spot[t] / 100.0)
        assert abs(float(a.mean()) - float(b.mean())) < 6e-3
        assert abs(float(a.std()) - float(b.std())) < 6e-3


def test_heston_shape_channels_and_replay() -> None:
    sim = _cuda_heston()
    state = sim.simulate(4096, noise=NoiseSpec(seed=17))
    assert state.spot.shape == (101, 4096)
    variance = state.aux["variance"]
    assert variance.shape == (101, 4096)
    assert torch.all(variance[0] == 0.04)
    assert torch.all(variance >= 0.0)
    replay = sim.simulate(4096, noise=NoiseSpec(seed=17))
    assert torch.equal(state.spot, replay.spot)
    assert torch.equal(variance, replay.aux["variance"])


def test_heston_zero_xi_degenerates_to_gbm_marginals() -> None:
    sim = CudaHestonSimulator(
        s0=100.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.0, rho=0.0, maturity=1.0, n_steps=100
    )
    state = sim.simulate(400_000, noise=NoiseSpec(seed=19))
    log_returns = torch.log(state.terminal / 100.0)
    assert abs(float(log_returns.mean()) - (-0.5 * 0.04)) < 5e-3
    assert abs(float(log_returns.std()) - math.sqrt(0.04)) < 5e-3


def test_heston_terminal_mean_and_skew_sign() -> None:
    state = _cuda_heston().simulate(400_000, noise=NoiseSpec(seed=23))
    terminal = state.terminal
    standard_error = float(terminal.std()) / math.sqrt(terminal.shape[0])
    assert abs(float(terminal.mean()) - 100.0) < max(5.0 * standard_error, 0.3)
    log_returns = torch.log(terminal / 100.0)
    centred = log_returns - log_returns.mean()
    assert float((centred**3).mean()) < 0.0


def test_gbm_folds_match_grid_bitwise() -> None:
    sim = _cuda_gbm()
    noise = NoiseSpec(seed=163)
    state = sim.simulate(8192, noise=noise)
    folds = sim.simulate_folds(8192, noise=noise)
    assert torch.equal(folds.terminal, state.terminal)
    assert torch.equal(folds.running_max, state.spot.max(dim=0).values)
    assert torch.equal(folds.running_min, state.spot.min(dim=0).values)


def test_heston_folds_match_grid_bitwise() -> None:
    sim = _cuda_heston()
    noise = NoiseSpec(seed=167)
    state = sim.simulate(8192, noise=noise)
    folds = sim.simulate_folds(8192, noise=noise)
    assert torch.equal(folds.terminal, state.terminal)
    assert torch.equal(folds.running_max, state.spot.max(dim=0).values)
    assert torch.equal(folds.running_min, state.spot.min(dim=0).values)


def test_pricer_fold_route_matches_grid_route() -> None:
    from deephedging.instruments import EuropeanCall, UpAndOutCall
    from deephedging.pricing import MonteCarloPricer

    sim = _cuda_gbm()
    pricer = MonteCarloPricer(n_paths=100_000, seed=173)
    for payoff in (EuropeanCall(strike=100.0), UpAndOutCall(strike=100.0, barrier=125.0)):
        fast = pricer.price(payoff, sim)
        grid_state = sim.simulate(100_000, noise=NoiseSpec(seed=173))
        grid_values = payoff(grid_state.spot)
        assert abs(fast.value - float(grid_values.mean())) < 1e-6
        assert fast.provenance == "monte-carlo"


def test_stream_bounds_enforced() -> None:
    sim = _cuda_gbm()
    with pytest.raises(ValueError):
        sim.simulate(16, noise=NoiseSpec(seed=1, stream=1 << 32))


def test_graphed_episode_matches_eager_training() -> None:
    from deephedging import (
        CVaR,
        EuropeanCall,
        FeedForwardPolicy,
        ProportionalCost,
        TrainConfig,
        train,
    )

    sim = CudaGBMSimulator(s0=100.0, sigma=0.2, maturity=0.25, n_steps=10)
    payoff = EuropeanCall(strike=100.0)
    cost = ProportionalCost(rate=1e-3)

    def run(graphed: bool, n_iterations: int) -> list[float]:
        torch.manual_seed(31)
        policy = FeedForwardPolicy(hidden_sizes=(16,)).to("cuda")
        config = TrainConfig(
            n_iterations=n_iterations,
            batch_paths=2048,
            lr=1e-3,
            seed=5,
            graph_episode=graphed,
        )
        return train(sim, policy, payoff, cost, CVaR(alpha=0.9), config, premium=4.0).losses

    first_eager = run(graphed=False, n_iterations=1)[0]
    first_graphed = run(graphed=True, n_iterations=1)[0]
    assert abs(first_eager - first_graphed) < 1e-5 * max(1.0, abs(first_eager))

    def run_checkpointed() -> float:
        torch.manual_seed(31)
        policy = FeedForwardPolicy(hidden_sizes=(16,)).to("cuda")
        config = TrainConfig(
            n_iterations=1,
            batch_paths=2048,
            lr=1e-3,
            seed=5,
            graph_episode=True,
            checkpoint_steps=True,
        )
        return train(sim, policy, payoff, cost, CVaR(alpha=0.9), config, premium=4.0).losses[0]

    first_checkpointed = run_checkpointed()
    assert abs(first_graphed - first_checkpointed) < 1e-5 * max(1.0, abs(first_graphed))

    from deephedging import VarianceFeatures

    heston = CudaHestonSimulator(
        s0=100.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, maturity=0.25, n_steps=10
    )

    def run_heston(graphed: bool) -> float:
        torch.manual_seed(33)
        policy = FeedForwardPolicy(n_features=4, hidden_sizes=(16,)).to("cuda")
        config = TrainConfig(
            n_iterations=1, batch_paths=2048, lr=1e-3, seed=6, graph_episode=graphed
        )
        result = train(
            heston,
            policy,
            payoff,
            cost,
            CVaR(alpha=0.9),
            config,
            premium=4.0,
            feature_map=VarianceFeatures(),
        )
        return result.losses[0]

    heston_eager = run_heston(graphed=False)
    heston_graphed = run_heston(graphed=True)
    assert abs(heston_eager - heston_graphed) < 1e-5 * max(1.0, abs(heston_eager))

    eager_losses = run(graphed=False, n_iterations=25)
    graphed_losses = run(graphed=True, n_iterations=25)
    for eager_value, graphed_value in zip(eager_losses, graphed_losses, strict=True):
        assert abs(eager_value - graphed_value) < 5e-2 * max(1.0, abs(eager_value))
    assert graphed_losses[-1] < graphed_losses[0]
