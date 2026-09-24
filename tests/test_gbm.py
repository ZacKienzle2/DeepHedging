# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.


from hypothesis import given
from hypothesis import strategies as st

import deephedging
import deephedging.market.gbm
from deephedging import GBMSimulator, NoiseSpec


@given(
    self=st.builds(
        GBMSimulator,
        s0=st.floats(1.0, 500.0),
        sigma=st.floats(0.0, 1.0),
        maturity=st.floats(0.01, 5.0),
        n_steps=st.integers(1, 64),
        mu=st.floats(-0.2, 0.2),
        sampler=st.sampled_from(("pseudo", "sobol")),
    ),
    n_paths=st.integers(1, 256),
    noise=st.one_of(
        st.none(),
        st.builds(
            NoiseSpec,
            seed=st.integers(0, 2**31),
            stream=st.one_of(st.just(0), st.integers(0, 2**20)),
        ),
    ),
)
def test_fuzz_gbm_simulator_simulate(
    self: GBMSimulator, n_paths: int, noise: deephedging.NoiseSpec | None
) -> None:
    deephedging.market.gbm.GBMSimulator.simulate(
        self=self, n_paths=n_paths, noise=noise
    )


@given(n_steps=st.integers(1, 128))
def test_fuzz_brownian_bridge_matrix(n_steps: int) -> None:
    deephedging.market.gbm.brownian_bridge_matrix(n_steps=n_steps)
