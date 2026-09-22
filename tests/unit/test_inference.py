"""Tests for bootstrap confidence intervals and paired comparison."""

import pytest
import torch

from deephedging.evaluation import bootstrap_metric, expected_shortfall, paired_bootstrap


def _sample(n: int = 20_000, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(n, generator=generator, dtype=torch.float64)


def _mean(sample: torch.Tensor) -> torch.Tensor:
    return sample.mean(dim=-1)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    pnl = _sample()
    result = bootstrap_metric(pnl, _mean, n_resamples=400, seed=1)
    assert result.low < result.estimate < result.high
    assert abs(result.estimate - float(pnl.mean())) < 1e-9
    assert abs(result.estimate) < 0.05


def test_bootstrap_is_reproducible_with_seed() -> None:
    pnl = _sample()
    first = bootstrap_metric(pnl, lambda sample: expected_shortfall(sample, 0.95), seed=7)
    second = bootstrap_metric(pnl, lambda sample: expected_shortfall(sample, 0.95), seed=7)
    assert (first.low, first.high) == (second.low, second.high)


def test_bootstrap_interval_narrows_with_more_paths() -> None:
    wide = bootstrap_metric(_sample(n=4_000), _mean, n_resamples=300, seed=2)
    narrow = bootstrap_metric(_sample(n=80_000), _mean, n_resamples=300, seed=2)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_chunked_resamples_match_one_by_one_evaluation() -> None:
    pnl = _sample(n=5_000)
    batched = expected_shortfall(pnl.expand(3, -1), 0.95)
    single = expected_shortfall(pnl, 0.95)
    assert torch.equal(batched, single.expand(3))


def test_paired_bootstrap_detects_a_dominant_strategy() -> None:
    first = _sample()
    second = first - 0.5
    result = paired_bootstrap(
        first, second, lambda sample: expected_shortfall(sample, 0.9), n_resamples=300, seed=4
    )
    assert result.difference < 0.0
    assert result.high < 0.0
    assert result.probability_first_lower > 0.99


def test_inference_rejects_invalid_inputs() -> None:
    pnl = _sample(n=100)
    with pytest.raises(ValueError, match=r"confidence must be in \(0, 1\)"):
        bootstrap_metric(pnl, _mean, confidence=1.0)
    with pytest.raises(ValueError, match="pnl must be 1-dimensional"):
        bootstrap_metric(pnl.reshape(10, 10), _mean)
    with pytest.raises(ValueError, match="paired samples must match in shape"):
        paired_bootstrap(pnl, pnl[:50], _mean)
    with pytest.raises(ValueError, match="metric must reduce the last dimension"):
        bootstrap_metric(pnl, lambda sample: sample.mean())
