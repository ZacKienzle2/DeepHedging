"""Tests for fixed-strike lookback payoffs."""

import torch

from deephedging.instruments import EuropeanCall, EuropeanPut, LookbackCall, LookbackPut


def _paths(n_steps: int = 20, n_paths: int = 5000, seed: int = 2) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    increments = 0.02 * torch.randn(n_steps, n_paths, generator=generator, dtype=torch.float64)
    log_path = torch.cat([torch.zeros(1, n_paths, dtype=torch.float64), increments.cumsum(dim=0)])
    return 100.0 * torch.exp(log_path)


def test_lookback_call_dominates_european_call() -> None:
    paths = _paths()
    lookback = LookbackCall(strike=100.0)(paths)
    european = EuropeanCall(strike=100.0)(paths)
    assert lookback.shape == (paths.shape[1],)
    assert bool(torch.all(lookback >= european - 1e-9))


def test_lookback_put_dominates_european_put() -> None:
    paths = _paths()
    lookback = LookbackPut(strike=100.0)(paths)
    european = EuropeanPut(strike=100.0)(paths)
    assert bool(torch.all(lookback >= european - 1e-9))


def test_monotone_increasing_path_matches_european_call() -> None:
    path = torch.linspace(80.0, 120.0, 11, dtype=torch.float64).unsqueeze(1)
    assert torch.allclose(LookbackCall(strike=100.0)(path), EuropeanCall(strike=100.0)(path))


def test_constant_path_pays_intrinsic() -> None:
    paths = torch.full((10, 4), 90.0, dtype=torch.float64)
    ten = torch.full((4,), 10.0, dtype=torch.float64)
    assert torch.allclose(LookbackPut(strike=100.0)(paths), ten)
    assert torch.allclose(LookbackCall(strike=80.0)(paths), ten)
