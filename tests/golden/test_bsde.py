"""Golden tests for the deep BSDE solver against closed forms.

Tolerances are statistical: the solver is trained with small budgets so CI
stays fast, and the references are exact Black-Scholes-type prices.
"""

import math
from collections.abc import Callable

import pytest
import torch

from deephedging.bsde import (
    BSDEConfig,
    BSDEProblem,
    DeepBSDESolver,
    DiscountGenerator,
    train_bsde,
)
from deephedging.evaluation import bs_call_delta, bs_call_price


def _call_terminal(strike: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def terminal(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x[:, 0] - strike, min=0.0)

    return terminal


def _config() -> BSDEConfig:
    return BSDEConfig(n_iterations=800, batch_paths=512, lr=3e-3, seed=7)


def test_one_dim_call_zero_rate_matches_black_scholes() -> None:
    torch.manual_seed(1)
    problem = BSDEProblem(
        dim=1,
        x0=100.0,
        sigma=0.2,
        maturity=1.0,
        n_steps=10,
        terminal=_call_terminal(100.0),
    )
    solver = DeepBSDESolver(dim=1, hidden_sizes=(32, 32))
    result = train_bsde(problem, solver, _config())
    reference = float(bs_call_price(100.0, 100.0, 0.2, 1.0))
    assert abs(result.y0 - reference) < 0.03 * reference
    assert result.final_loss < sum(result.losses[:5]) / 5


def test_one_dim_call_with_rate_matches_black_scholes() -> None:
    torch.manual_seed(2)
    rate = 0.05
    problem = BSDEProblem(
        dim=1,
        x0=100.0,
        sigma=0.2,
        maturity=1.0,
        n_steps=10,
        terminal=_call_terminal(100.0),
        generator=DiscountGenerator(rate=rate),
        mu=rate,
    )
    solver = DeepBSDESolver(dim=1, hidden_sizes=(32, 32))
    result = train_bsde(problem, solver, _config())
    reference = float(bs_call_price(100.0, 100.0, 0.2, 1.0, rate=rate))
    assert abs(result.y0 - reference) < 0.03 * reference


def test_initial_z_approximates_delta_times_diffusion() -> None:
    torch.manual_seed(3)
    sigma = 0.2
    problem = BSDEProblem(
        dim=1,
        x0=100.0,
        sigma=sigma,
        maturity=1.0,
        n_steps=10,
        terminal=_call_terminal(100.0),
    )
    solver = DeepBSDESolver(dim=1, hidden_sizes=(32, 32))
    train_bsde(problem, solver, _config())
    with torch.no_grad():
        z0 = float(solver.z_net(torch.zeros((1, 2)))[0, 0])
    delta = float(bs_call_delta(100.0, 100.0, sigma, 1.0))
    reference = sigma * 100.0 * delta
    assert abs(z0 - reference) < 0.15 * reference


def test_geometric_basket_matches_lognormal_closed_form() -> None:
    torch.manual_seed(4)
    dim, x0, sigma, maturity, strike = 5, 100.0, 0.2, 1.0, 100.0

    def terminal(x: torch.Tensor) -> torch.Tensor:
        geometric = torch.exp(torch.log(x).mean(dim=1))
        return torch.clamp(geometric - strike, min=0.0)

    problem = BSDEProblem(
        dim=dim,
        x0=x0,
        sigma=sigma,
        maturity=maturity,
        n_steps=10,
        terminal=terminal,
    )
    solver = DeepBSDESolver(dim=dim, hidden_sizes=(48, 48))
    result = train_bsde(problem, solver, _config())

    sigma_g = sigma / math.sqrt(dim)
    forward = x0 * math.exp(-0.5 * sigma**2 * maturity + 0.5 * sigma_g**2 * maturity)
    d1 = (math.log(forward / strike) + 0.5 * sigma_g**2 * maturity) / (
        sigma_g * math.sqrt(maturity)
    )
    d2 = d1 - sigma_g * math.sqrt(maturity)
    normal = torch.distributions.Normal(0.0, 1.0)
    reference = forward * float(normal.cdf(torch.tensor(d1))) - strike * float(
        normal.cdf(torch.tensor(d2))
    )
    assert abs(result.y0 - reference) < 0.05 * reference


def test_solver_dim_mismatch_rejected() -> None:
    problem = BSDEProblem(
        dim=2,
        x0=100.0,
        sigma=0.2,
        maturity=1.0,
        n_steps=5,
        terminal=_call_terminal(100.0),
    )
    solver = DeepBSDESolver(dim=1)
    with pytest.raises(ValueError):
        train_bsde(problem, solver, BSDEConfig(n_iterations=1))


def test_invalid_problem_parameters_raise() -> None:
    with pytest.raises(ValueError):
        BSDEProblem(
            dim=0, x0=100.0, sigma=0.2, maturity=1.0, n_steps=5, terminal=_call_terminal(100.0)
        )
    with pytest.raises(ValueError):
        BSDEProblem(
            dim=1, x0=-1.0, sigma=0.2, maturity=1.0, n_steps=5, terminal=_call_terminal(100.0)
        )
