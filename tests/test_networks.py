# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

from hypothesis import given
from hypothesis import strategies as st

import deephedging.networks


@given(
    in_features=st.integers(1, 64),
    hidden_sizes=st.lists(st.integers(1, 64), max_size=4).map(tuple),
    out_features=st.integers(1, 64),
)
def test_fuzz_mlp(in_features: int, hidden_sizes: tuple[int, ...], out_features: int) -> None:
    deephedging.networks.mlp(
        in_features=in_features, hidden_sizes=hidden_sizes, out_features=out_features
    )
