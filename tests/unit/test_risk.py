"""Tests for risk measures."""

import math

import pytest
import torch

from deephedging.evaluation import expected_shortfall
from deephedging.risk import CVaR, Entropic, MeanVariance, SpectralRisk


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


def test_spectral_cvar_converges_to_analytic_cvar() -> None:
    alpha = 0.95
    loss = _normal_sample()
    analytic = math.exp(-0.5 * 1.6448536**2) / (math.sqrt(2 * math.pi) * (1 - alpha))
    measure = SpectralRisk.conditional_value_at_risk(alpha)
    with torch.no_grad():
        assert abs(float(measure(loss)) - analytic) < 0.02


def test_spectral_cvar_tracks_empirical_expected_shortfall() -> None:
    loss = _normal_sample(n=200_000)
    empirical = float(expected_shortfall(-loss, alpha=0.9))
    measure = SpectralRisk.conditional_value_at_risk(0.9)
    with torch.no_grad():
        assert abs(float(measure(loss)) - empirical) < 5e-3


def test_spectral_exponential_recovers_mean_in_the_risk_neutral_limit() -> None:
    loss = _normal_sample(n=100_000)
    measure = SpectralRisk.exponential(risk_aversion=1e-4)
    with torch.no_grad():
        assert abs(float(measure(loss)) - float(loss.mean())) < 1e-3


def test_spectral_exponential_increases_with_risk_aversion() -> None:
    loss = _normal_sample(n=100_000)
    with torch.no_grad():
        low = float(SpectralRisk.exponential(risk_aversion=0.5)(loss))
        high = float(SpectralRisk.exponential(risk_aversion=4.0)(loss))
    assert high > low


def test_spectral_risk_is_positively_homogeneous_and_translation_invariant() -> None:
    loss = _normal_sample(n=50_000)
    measure = SpectralRisk.exponential(risk_aversion=2.0)
    with torch.no_grad():
        base = float(measure(loss))
        scaled = float(measure(3.0 * loss))
        shifted = float(measure(loss + 5.0))
    assert abs(scaled - 3.0 * base) < 1e-6
    assert abs(shifted - (base + 5.0)) < 1e-6


def test_spectral_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        SpectralRisk.exponential(risk_aversion=0.0)
    with pytest.raises(ValueError):
        SpectralRisk.conditional_value_at_risk(alpha=1.0)


def test_mean_variance_reduces_to_mean_without_aversion() -> None:
    loss = _normal_sample(n=10_000)
    measure = MeanVariance(risk_aversion=0.0)
    with torch.no_grad():
        assert abs(float(measure(loss)) - float(loss.mean())) < 1e-9


def test_mean_variance_increases_with_aversion_on_dispersed_loss() -> None:
    loss = _normal_sample(n=10_000)
    with torch.no_grad():
        low = float(MeanVariance(risk_aversion=0.5)(loss))
        high = float(MeanVariance(risk_aversion=2.0)(loss))
    assert high > low


def test_mean_semivariance_penalises_less_than_full_variance() -> None:
    loss = _normal_sample(n=10_000)
    with torch.no_grad():
        full = float(MeanVariance(risk_aversion=2.0)(loss))
        downside = float(MeanVariance(risk_aversion=2.0, downside=True)(loss))
    assert downside < full


def test_mean_variance_reduces_to_mean_on_constant_loss() -> None:
    loss = torch.full((100,), 3.0, dtype=torch.float64)
    with torch.no_grad():
        assert abs(float(MeanVariance(risk_aversion=5.0)(loss)) - 3.0) < 1e-9
        assert abs(float(MeanVariance(risk_aversion=5.0, downside=True)(loss)) - 3.0) < 1e-9


def test_mean_variance_weighted_matches_unweighted_under_uniform_weights() -> None:
    loss = _normal_sample(n=5_000)
    weights = torch.ones_like(loss)
    measure = MeanVariance(risk_aversion=1.5)
    with torch.no_grad():
        assert abs(float(measure(loss, weights=weights)) - float(measure(loss))) < 1e-9


def test_mean_variance_rejects_negative_aversion() -> None:
    with pytest.raises(ValueError):
        MeanVariance(risk_aversion=-1.0)
