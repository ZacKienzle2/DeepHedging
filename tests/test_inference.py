# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.


import collections.abc
import functools

import torch
from hypothesis import given
from hypothesis import strategies as st

import deephedging
import deephedging.baselines


@given(
    confidence=st.floats(0.5, 0.99),
    metric=st.sampled_from(
        (
            functools.partial(torch.mean, dim=-1),
            functools.partial(deephedging.expected_shortfall, alpha=0.9),
        )
    ),
    n_resamples=st.integers(1, 64),
    pnl=st.lists(st.floats(-10.0, 10.0), min_size=2, max_size=64).map(
        lambda values: torch.tensor(values, dtype=torch.float64)
    ),
    seed=st.integers(0, 2**31),
)
def test_equivalent_bootstrap_metric_per_resample_bootstrap_metric(
    confidence: float,
    metric: collections.abc.Callable[[torch.Tensor], torch.Tensor],
    n_resamples: int,
    pnl: torch.Tensor,
    seed: int,
) -> None:
    result_bootstrap_metric_per_resample = (
        deephedging.baselines.bootstrap_metric_per_resample(
            pnl=pnl,
            metric=metric,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
    )
    result_bootstrap_metric = deephedging.bootstrap_metric(
        pnl=pnl,
        metric=metric,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    assert result_bootstrap_metric_per_resample == result_bootstrap_metric, (
        result_bootstrap_metric_per_resample,
        result_bootstrap_metric,
    )
