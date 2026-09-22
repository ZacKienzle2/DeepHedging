# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.


from hypothesis import given
from hypothesis import strategies as st

import deephedging
import deephedging.market.rough_bergomi
from deephedging import NoiseSpec, RoughBergomiSimulator

_HURST = st.floats(0.02, 0.48)
_RHO = st.floats(-0.99, 0.99)
_MATURITY = st.floats(0.01, 3.0)
_STEPS = st.integers(1, 32)


@given(
    self=st.builds(
        RoughBergomiSimulator,
        s0=st.floats(1.0, 500.0),
        xi0=st.floats(0.001, 1.0),
        hurst=_HURST,
        eta=st.floats(0.0, 3.0),
        rho=_RHO,
        maturity=_MATURITY,
        n_steps=_STEPS,
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
def test_fuzz_rough_bergomi_simulator_simulate(
    self: RoughBergomiSimulator, n_paths: int, noise: deephedging.NoiseSpec | None
) -> None:
    deephedging.market.rough_bergomi.RoughBergomiSimulator.simulate(
        self=self, n_paths=n_paths, noise=noise
    )


@given(hurst=_HURST, rho=_RHO, maturity=_MATURITY, n_steps=_STEPS)
def test_fuzz_rough_bergomi_factor(hurst: float, rho: float, maturity: float, n_steps: int) -> None:
    deephedging.market.rough_bergomi.rough_bergomi_factor(
        hurst=hurst, rho=rho, maturity=maturity, n_steps=n_steps
    )
