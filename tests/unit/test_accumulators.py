"""Tests pinning the streaming folds against the materialised oracle."""

import torch

from deephedging.accumulators import (
    BarrierAliveAccumulator,
    RunningMaxAccumulator,
    fold_path,
)
from deephedging.instruments import UpAndOutCall
from deephedging.market import GBMSimulator, MarketState, NoiseSpec


def _state(n_paths: int = 20_000, seed: int = 29) -> MarketState:
    sim = GBMSimulator(s0=100.0, sigma=0.3, maturity=1.0, n_steps=50)
    return sim.simulate(n_paths, noise=NoiseSpec(seed=seed))


def test_running_max_fold_matches_cummax_oracle() -> None:
    state = _state()
    streamed = fold_path(state, RunningMaxAccumulator())
    oracle = state.running_max(state.n_steps)
    assert torch.equal(streamed, oracle)


def test_barrier_fold_matches_payoff_indicator() -> None:
    state = _state()
    barrier = 115.0
    alive = fold_path(state, BarrierAliveAccumulator(barrier=barrier))
    payoff = UpAndOutCall(strike=100.0, barrier=barrier)
    vanilla = torch.clamp(state.terminal - 100.0, min=0.0)
    assert torch.equal(payoff(state.spot), vanilla * alive)


def test_barrier_fold_knocks_at_inception() -> None:
    spot = torch.tensor([[100.0, 100.0], [90.0, 110.0], [95.0, 105.0]])
    state = MarketState.from_spot(spot)
    alive = fold_path(state, BarrierAliveAccumulator(barrier=100.0))
    assert torch.equal(alive, torch.zeros(2))
