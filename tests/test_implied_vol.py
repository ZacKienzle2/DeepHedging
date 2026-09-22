# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import torch
from hypothesis import given
from hypothesis import strategies as st

import deephedging.baselines
import deephedging.calibration
from deephedging import bs_call_price

_STRIKES = torch.linspace(100.0, 200.0, 10, dtype=torch.float64)


@given(
    n_iterations=st.integers(2, 4),
    prices=st.lists(st.floats(0.1, 1.0), min_size=10, max_size=10).map(
        lambda sigmas: bs_call_price(100.0, _STRIKES, torch.tensor(sigmas), 0.5)
    ),
    spot=st.just(100.0),
    strikes=st.just(_STRIKES),
    tau=st.just(0.5),
)
def test_equivalent_implied_vol_by_mpmath_implied_vol(
    n_iterations: int,
    prices: torch.Tensor,
    spot: float,
    strikes: torch.Tensor,
    tau: float,
) -> None:
    result_implied_vol_by_mpmath = deephedging.baselines.implied_vol_by_mpmath(
        prices=prices, spot=spot, strikes=strikes, tau=tau
    )
    result_implied_vol = deephedging.calibration.implied_vol(
        prices=prices, spot=spot, strikes=strikes, tau=tau, n_iterations=n_iterations
    )
    torch.testing.assert_close(
        result_implied_vol, result_implied_vol_by_mpmath, rtol=1e-12, atol=0.0
    )
