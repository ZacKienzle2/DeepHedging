"""Tests for risk measures."""

import math

import pytest
import torch

from deephedging.evaluation import expected_shortfall
from deephedging.risk import CVaR, Entropic


def _normal_sample(n: int = 400_000, seed: int = 11) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(n, generator=generator, dtype=torch.float64)


def test_cvar_objective_upper_bounds_analytic_cvar() -> None:
    alpha = 0.95
    loss = _normal_sample()
    measure = CVaR(alpha=alpha)
    analytic = math.exp(-0.5 * 1.6448536**2) / (math.sqrt(2 * math.pi) * (1 - alpha))
    with torch.no_grad():
        objective = float(measure(loss))
    assert objective >= analytic - 0.02


def test_cvar_threshold_optimisation_recovers_analytic_value() -> None:
    alpha = 0.95
    z_alpha = 1.6448536
    analytic = math.exp(-0.5 * z_alpha**2) / (math.sqrt(2 * math.pi) * (1 - alpha))
    loss = _normal_sample()
    measure = CVaR(alpha=alpha)
    optimizer = torch.optim.Adam(measure.parameters(), lr=0.05)
    for _ in range(500):
        objective = measure(loss)
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        assert abs(float(measure(loss)) - analytic) < 0.02
        assert abs(float(measure.threshold) - z_alpha) < 0.05


def test_cvar_matches_empirical_expected_shortfall_at_optimum() -> None:
    loss = _normal_sample(n=100_000)
    empirical = float(expected_shortfall(-loss, alpha=0.9))
    measure = CVaR(alpha=0.9)
    with torch.no_grad():
        measure.threshold.copy_(torch.quantile(loss, 0.9))
    assert abs(float(measure(loss)) - empirical) < 1e-6


def test_cvar_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        CVaR(alpha=0.0)
    with pytest.raises(ValueError):
        CVaR(alpha=1.0)


def test_entropic_matches_analytic_normal_value() -> None:
    risk_aversion = 0.5
    loss = _normal_sample(n=1_200_000)
    measure = Entropic(risk_aversion=risk_aversion)
    analytic = 0.5 * risk_aversion
    assert abs(float(measure(loss)) - analytic) < 3e-3


def test_entropic_increases_with_risk_aversion() -> None:
    loss = _normal_sample(n=100_000)
    low = float(Entropic(risk_aversion=0.5)(loss))
    high = float(Entropic(risk_aversion=2.0)(loss))
    assert high > low


def test_entropic_rejects_invalid_risk_aversion() -> None:
    with pytest.raises(ValueError):
        Entropic(risk_aversion=0.0)


def test_entropic_stable_for_large_losses() -> None:
    loss = torch.tensor([1e4, 2e4, 3e4], dtype=torch.float64)
    value = float(Entropic(risk_aversion=1.0)(loss))
    assert math.isfinite(value)
    assert value > 2.9e4
