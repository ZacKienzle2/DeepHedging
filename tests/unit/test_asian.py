"""Tests for Asian average-price payoffs."""

import torch

from deephedging.instruments import AsianCall, AsianPut


def _paths(n_steps: int = 20, n_paths: int = 5000, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    increments = 0.02 * torch.randn(n_steps, n_paths, generator=generator, dtype=torch.float64)
    log_path = torch.cat([torch.zeros(1, n_paths, dtype=torch.float64), increments.cumsum(dim=0)])
    return 100.0 * torch.exp(log_path)


def test_geometric_average_never_exceeds_arithmetic() -> None:
    paths = _paths()
    arithmetic = AsianCall(strike=0.0)(paths)
    geometric = AsianCall(strike=0.0, geometric=True)(paths)
    assert bool(torch.all(geometric <= arithmetic + 1e-9))


def test_asian_call_stays_below_the_lookback_and_returns_per_path() -> None:
    paths = _paths()
    asian = AsianCall(strike=100.0)(paths)
    assert asian.shape == (paths.shape[1],)
    lookback = torch.clamp(paths.max(dim=0).values - 100.0, min=0.0)
    assert bool(torch.all(asian <= lookback + 1e-9))


def test_asian_call_put_parity_on_the_average() -> None:
    paths = _paths()
    strike = 100.0
    call = AsianCall(strike=strike)(paths)
    put = AsianPut(strike=strike)(paths)
    assert torch.allclose(call - put, paths.mean(dim=0) - strike, atol=1e-9)


def test_constant_path_pays_intrinsic() -> None:
    paths = torch.full((10, 3), 120.0, dtype=torch.float64)
    twenty = torch.full((3,), 20.0, dtype=torch.float64)
    ten = torch.full((3,), 10.0, dtype=torch.float64)
    assert torch.allclose(AsianCall(strike=100.0)(paths), twenty)
    assert torch.allclose(AsianCall(strike=100.0, geometric=True)(paths), twenty)
    assert torch.allclose(AsianPut(strike=130.0)(paths), ten)
