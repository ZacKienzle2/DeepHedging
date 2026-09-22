# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import torch
from hypothesis import given
from hypothesis import strategies as st

import deephedging
import deephedging.baselines


@given(
    jump_intensity=st.floats(0.0, 5.0),
    jump_mean=st.floats(-0.5, 0.5),
    jump_vol=st.floats(0.0, 0.5),
    sigma=st.floats(0.05, 0.8),
    spot=st.floats(50.0, 150.0),
    strike=st.floats(50.0, 150.0),
    tau=st.floats(0.05, 3.0),
)
def test_equivalent_merton_call_price_per_term_merton_call_price(
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    sigma: float,
    spot: float,
    strike: float,
    tau: float,
) -> None:
    result_merton_call_price_per_term = deephedging.baselines.merton_call_price_per_term(
        spot=spot,
        strike=strike,
        sigma=sigma,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_vol=jump_vol,
        tau=tau,
    )
    result_merton_call_price = deephedging.merton_call_price(
        spot=spot,
        strike=strike,
        sigma=sigma,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_vol=jump_vol,
        tau=tau,
    )
    torch.testing.assert_close(result_merton_call_price, result_merton_call_price_per_term)
