# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import functools

import torch
from hypothesis import given
from hypothesis import strategies as st

import deephedging
import deephedging.bsde
from deephedging.bsde import BackwardConfig, BSDEProblem, DiscountGenerator, ZeroGenerator


@given(
    problem=st.builds(
        BSDEProblem,
        dim=st.integers(1, 3),
        x0=st.floats(50.0, 150.0),
        sigma=st.floats(0.05, 0.5),
        maturity=st.floats(0.1, 2.0),
        n_steps=st.integers(1, 4),
        terminal=st.just(functools.partial(torch.sum, dim=-1)),
        generator=st.one_of(
            st.just(ZeroGenerator()), st.builds(DiscountGenerator, rate=st.floats(0.0, 0.1))
        ),
        mu=st.floats(-0.05, 0.1),
    ),
    config=st.builds(
        BackwardConfig,
        american=st.one_of(st.just(False), st.booleans()),
        batch_paths=st.one_of(st.just(16), st.integers(1, 64)),
        device=st.just("cpu"),
        first_iterations=st.one_of(st.just(2), st.integers(1, 3)),
        hidden_sizes=st.one_of(st.just((8, 8)), st.lists(st.integers(1, 8), max_size=2).map(tuple)),
        iterations=st.one_of(st.just(1), st.integers(1, 3)),
        lr=st.one_of(st.just(0.001), st.floats(1e-4, 1e-2)),
        seed=st.one_of(st.none(), st.none(), st.integers(0, 2**31)),
    ),
)
def test_fuzz_solve_backward(
    problem: deephedging.BSDEProblem, config: deephedging.bsde.BackwardConfig
) -> None:
    deephedging.bsde.solve_backward(problem=problem, config=config)
