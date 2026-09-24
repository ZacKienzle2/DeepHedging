# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import torch
from hypothesis import given
from hypothesis import strategies as st

import deephedging
import deephedging.features
from deephedging import DefaultFeatures, GBMSimulator, LevelFeatures, NoiseSpec

_PATHS, _STEPS = 16, 8


@given(
    self=st.builds(
        LevelFeatures,
        base=st.just(DefaultFeatures()),
        codes=st.sampled_from((1, 2, 4, 8, 16))
        .flatmap(
            lambda count: st.lists(st.floats(-3.0, 3.0), min_size=count, max_size=count)
        )
        .map(tuple),
    ),
    state=st.integers(0, 2**31).map(
        lambda seed: GBMSimulator(
            s0=100.0, sigma=0.2, maturity=1.0, n_steps=_STEPS
        ).simulate(_PATHS, noise=NoiseSpec(seed=seed))
    ),
    t=st.integers(0, _STEPS - 1),
    tau=st.floats(0.0, 1.0).map(torch.tensor),
    position=st.lists(st.floats(-2.0, 2.0), min_size=_PATHS, max_size=_PATHS).map(
        torch.tensor
    ),
)
def test_fuzz_level_features_call(
    self: LevelFeatures,
    state: deephedging.MarketState,
    t: int,
    tau: torch.Tensor,
    position: torch.Tensor,
) -> None:
    deephedging.features.LevelFeatures.__call__(
        self=self, state=state, t=t, tau=tau, position=position
    )
