"""Throughput benchmarks for generation, pricing and training.

pytest-benchmark calibrates the rounds, reports the median with its spread,
and compares a run against a saved one with ``--benchmark-compare``, which
is what the hand-written timing loop and baseline gate used to approximate.
Each timed call ends in a device synchronisation, so a CUDA measurement
covers the kernels it launched rather than their enqueueing alone.
"""

from collections.abc import Callable
from typing import TypedDict

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from deephedging import (
    CVaR,
    EuropeanCall,
    FeedForwardPolicy,
    GBMSimulator,
    HestonSimulator,
    NoiseSpec,
    ProportionalCost,
    TrainConfig,
    UpAndOutCall,
    hedge_pnl,
    train,
)
from deephedging.market import CudaGBMSimulator, CudaHestonSimulator, kernels_available
from deephedging.pricing import MonteCarloPricer

_STEPS = 30
_PATHS = {"cpu": 8192, "cuda": 65_536}


class _HestonParameters(TypedDict):
    s0: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float


_HESTON: _HestonParameters = {
    "s0": 100.0,
    "v0": 0.04,
    "kappa": 1.5,
    "theta": 0.04,
    "xi": 0.5,
    "rho": -0.7,
}
_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
_NOISE = NoiseSpec(seed=1)

fused = pytest.mark.skipif(
    not torch.cuda.is_available() or not kernels_available(),
    reason="fused kernels need CUDA and a host compiler",
)


def _synchronised(device: str, fn: Callable[[], object]) -> Callable[[], None]:
    def run() -> None:
        fn()
        if device == "cuda":
            torch.cuda.synchronize()

    return run


@pytest.mark.parametrize("device", _DEVICES)
def test_gbm_generation(benchmark: BenchmarkFixture, device: str) -> None:
    simulator = GBMSimulator(
        s0=100.0, sigma=0.2, maturity=0.25, n_steps=_STEPS, device=device
    )
    benchmark(
        _synchronised(device, lambda: simulator.simulate(_PATHS[device], noise=_NOISE))
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_heston_generation(benchmark: BenchmarkFixture, device: str) -> None:
    simulator = HestonSimulator(**_HESTON, maturity=0.25, n_steps=_STEPS, device=device)
    benchmark(
        _synchronised(device, lambda: simulator.simulate(_PATHS[device], noise=_NOISE))
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_eager_train_step(benchmark: BenchmarkFixture, device: str) -> None:
    simulator = GBMSimulator(
        s0=100.0, sigma=0.2, maturity=0.25, n_steps=_STEPS, device=device
    )
    policy = FeedForwardPolicy(hidden_sizes=(64, 64)).to(device)
    risk = CVaR(alpha=0.95).to(device)
    optimizer = torch.optim.Adam([*policy.parameters(), *risk.parameters()], lr=1e-3)
    payoff = EuropeanCall(strike=100.0)
    cost = ProportionalCost(rate=1e-3)

    def step() -> None:
        state = simulator.simulate(_PATHS[device], noise=_NOISE)
        loss = risk(-hedge_pnl(state, policy, payoff, cost, premium=4.0))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    benchmark(_synchronised(device, step))


@pytest.mark.gpu
@fused
@pytest.mark.parametrize("model", ["gbm", "heston"])
def test_fused_generation(benchmark: BenchmarkFixture, model: str) -> None:
    simulator = (
        CudaGBMSimulator(s0=100.0, sigma=0.2, maturity=0.25, n_steps=_STEPS)
        if model == "gbm"
        else CudaHestonSimulator(**_HESTON, maturity=0.25, n_steps=_STEPS)
    )
    benchmark(
        _synchronised("cuda", lambda: simulator.simulate(_PATHS["cuda"], noise=_NOISE))
    )


@pytest.mark.gpu
@fused
@pytest.mark.parametrize("precision", ["fp32", "tf32", "bf16"])
def test_generated_training(
    benchmark: BenchmarkFixture, monkeypatch: pytest.MonkeyPatch, precision: str
) -> None:
    if precision == "tf32":
        monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32")
    simulator = CudaHestonSimulator(**_HESTON, maturity=0.25, n_steps=_STEPS)
    config = TrainConfig(
        n_iterations=200,
        batch_paths=_PATHS["cuda"],
        seed=1,
        amp=precision == "bf16",
        graph_episode=True,
        graph_generate=True,
    )

    def run() -> None:
        train(
            simulator,
            FeedForwardPolicy(hidden_sizes=(64, 64)).to("cuda"),
            EuropeanCall(strike=100.0),
            ProportionalCost(rate=1e-3),
            CVaR(alpha=0.95),
            config,
            premium=4.0,
        )

    benchmark.pedantic(_synchronised("cuda", run), rounds=3, warmup_rounds=1)


@pytest.mark.gpu
@fused
@pytest.mark.parametrize("route", ["fold", "grid"])
def test_barrier_pricing(benchmark: BenchmarkFixture, route: str) -> None:
    simulator = CudaHestonSimulator(**_HESTON, maturity=1.0, n_steps=250)
    barrier = UpAndOutCall(strike=100.0, barrier=130.0)
    pricer = MonteCarloPricer(n_paths=_PATHS["cuda"], seed=1)

    def fold() -> None:
        pricer.price(barrier, simulator)

    def grid() -> None:
        barrier(simulator.simulate(_PATHS["cuda"], noise=_NOISE).spot).mean()

    benchmark(_synchronised("cuda", fold if route == "fold" else grid))
