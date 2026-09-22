# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.


import torch
from hypothesis import given
from hypothesis import strategies as st

import deephedging
import deephedging.baselines


@given(
    rate=st.floats(-0.05, 0.1),
    sigma=st.floats(0.02, 2.0),
    spot=st.one_of(st.just(100.0), st.floats(50.0, 200.0)),
    strike=st.one_of(st.just(100.0), st.floats(25.0, 400.0)),
    tau=st.floats(0.02, 5.0),
)
def test_equivalent_bs_call_price_by_mpmath_bs_call_price(
    rate: float,
    sigma: float,
    spot: float,
    strike: float,
    tau: float,
) -> None:
    result_bs_call_price_by_mpmath = deephedging.baselines.bs_call_price_by_mpmath(
        spot=spot, strike=strike, sigma=sigma, tau=tau, rate=rate
    )
    result_bs_call_price = deephedging.bs_call_price(
        spot=spot, strike=strike, sigma=sigma, tau=tau, rate=rate
    )
    torch.testing.assert_close(
        result_bs_call_price, result_bs_call_price_by_mpmath, rtol=1e-9, atol=1e-280
    )
