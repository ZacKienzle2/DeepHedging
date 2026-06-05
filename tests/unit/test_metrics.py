"""Tests for evaluation metrics."""

import pytest
import torch

from deephedging.evaluation import expected_shortfall, pnl_summary


def _pnl_sample(n: int = 50_000, seed: int = 31) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(n, generator=generator, dtype=torch.float64)


def test_pnl_summary_keys_and_consistency() -> None:
    pnl = _pnl_sample()
    summary = pnl_summary(pnl)
    assert set(summary) == {"mean", "std", "es_95", "es_99"}
    assert summary["es_95"] == pytest.approx(float(expected_shortfall(pnl, 0.95)))
    assert summary["es_99"] == pytest.approx(float(expected_shortfall(pnl, 0.99)))


def test_expected_shortfall_monotone_in_alpha() -> None:
    pnl = _pnl_sample()
    es_low = float(expected_shortfall(pnl, alpha=0.9))
    es_high = float(expected_shortfall(pnl, alpha=0.99))
    assert es_high > es_low


def test_expected_shortfall_exceeds_mean_loss() -> None:
    pnl = _pnl_sample()
    assert float(expected_shortfall(pnl, alpha=0.95)) > float((-pnl).mean())


def test_expected_shortfall_rejects_invalid_alpha() -> None:
    pnl = _pnl_sample(n=100)
    with pytest.raises(ValueError):
        expected_shortfall(pnl, alpha=0.0)
    with pytest.raises(ValueError):
        expected_shortfall(pnl, alpha=1.0)
