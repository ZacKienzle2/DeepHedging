# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.


import torch
from hypothesis import given
from hypothesis import strategies as st

import deephedging.risk.multi_level
from deephedging import CVaR, Entropic, MultiLevelRisk

_LEVELS = 4
_MEASURES = st.lists(
    st.one_of(
        st.builds(Entropic, st.floats(0.1, 5.0)), st.builds(CVaR, st.floats(0.5, 0.99))
    ),
    min_size=_LEVELS,
    max_size=_LEVELS,
).map(MultiLevelRisk)
_LOSS = (
    st.integers(1, 64)
    .flatmap(
        lambda rows: st.lists(
            st.floats(-10.0, 10.0), min_size=rows * _LEVELS, max_size=rows * _LEVELS
        )
    )
    .map(lambda values: torch.tensor(values, dtype=torch.float64))
)


@given(
    self=_MEASURES,
    loss=_LOSS,
    weights=st.none(),
)
def test_fuzz_multi_level_risk_forward(
    self: MultiLevelRisk, loss: torch.Tensor, weights: torch.Tensor | None
) -> None:
    deephedging.risk.multi_level.MultiLevelRisk.forward(
        self=self, loss=loss, weights=weights
    )


@given(
    self=_MEASURES,
    loss=_LOSS,
    weights=st.none(),
)
def test_fuzz_multi_level_risk_warm_start(
    self: MultiLevelRisk, loss: torch.Tensor, weights: torch.Tensor | None
) -> None:
    deephedging.risk.multi_level.MultiLevelRisk.warm_start(
        self=self, loss=loss, weights=weights
    )
