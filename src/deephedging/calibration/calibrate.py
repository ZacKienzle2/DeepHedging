"""Heston surface calibration and the analytic pricer seam.

Calibration minimises vega-weighted price residuals, which agree with
implied-volatility residuals to first order without running a root-find
inside every optimiser step. Weights are computed once from the market
quotes and detached; they are an objective design choice, not a
variable. Parameters are optimised through softplus and tanh transforms
so the optimiser is unconstrained while the model always sees admissible
values, and the COS pricer's truncation bounds are recomputed each
iteration from detached cumulants so the expansion tracks the moving
parameters without contaminating their gradients.
"""

import math
from dataclasses import dataclass, field

import torch

from deephedging.calibration.cf import cos_call_price
from deephedging.calibration.heston_cf import HestonParams, heston_cf, heston_cumulants
from deephedging.calibration.implied_vol import implied_vol
from deephedging.instruments.base import Payoff
from deephedging.instruments.vanilla import EuropeanCall
from deephedging.market.base import PathSimulator
from deephedging.market.heston import HestonSimulator
from deephedging.pricing import PriceEstimate

_RHO_BOUND = 0.999


def _softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(value)


def _softplus_inverse(value: float) -> float:
    return math.log(math.expm1(value))


def _constrain(raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (
        _softplus(raw[0]),
        _softplus(raw[1]),
        _softplus(raw[2]),
        _softplus(raw[3]),
        torch.tanh(raw[4]) * _RHO_BOUND,
    )


def _unconstrain(params: HestonParams) -> torch.Tensor:
    return torch.tensor(
        [
            _softplus_inverse(params.v0),
            _softplus_inverse(params.kappa),
            _softplus_inverse(params.theta),
            _softplus_inverse(params.xi),
            math.atanh(min(max(params.rho / _RHO_BOUND, -0.999), 0.999)),
        ],
        dtype=torch.float64,
    )


def price_surface(
    params: tuple[torch.Tensor, ...],
    s0: float,
    strikes: torch.Tensor,
    tau: float,
    n_terms: int = 160,
) -> torch.Tensor:
    """Prices a strike grid under Heston parameters carried as tensors.

    Args:
        params: ``(v0, kappa, theta, xi, rho)`` scalar tensors, possibly
            requiring gradients.
        s0: Spot price.
        strikes: Strike grid of shape ``(n_strikes,)`` in float64.
        tau: Time to maturity in years.
        n_terms: Number of cosine expansion terms.

    Returns:
        Call prices of shape ``(n_strikes,)``.
    """
    detached = tuple(value.detach() for value in params)
    c1, c2 = heston_cumulants(tau, *detached)
    width = 12.0 * math.sqrt(max(c2, 1e-12))

    def cf(u: torch.Tensor) -> torch.Tensor:
        return heston_cf(u, tau, *params)

    return cos_call_price(cf, s0, strikes, c1 - width, c1 + width, n_terms=n_terms)


@dataclass(frozen=True)
class CalibrationConfig:
    """Hyperparameters for surface calibration.

    Attributes:
        n_iterations: Number of optimiser steps.
        lr: Adam learning rate on the unconstrained parameters.
    """

    n_iterations: int = 400
    lr: float = 5e-2


@dataclass
class CalibrationResult:
    """Outcome of a calibration run.

    Attributes:
        params: Recovered model parameters.
        final_loss: Vega-weighted squared residual at the last iteration.
        losses: Loss recorded at every iteration.
    """

    params: HestonParams
    final_loss: float
    losses: list[float] = field(default_factory=list)


def calibrate_heston(
    market_prices: torch.Tensor,
    s0: float,
    strikes: torch.Tensor,
    taus: tuple[float, ...],
    initial: HestonParams,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Calibrates Heston parameters to a multi-maturity quote surface.

    A single-maturity strip underdetermines the parameters: the smile
    constrains a blend of initial and long-run variance, so distinct
    parameter sets reprice one maturity almost exactly. The term
    structure is what separates ``v0`` from ``theta`` and pins the mean
    reversion, so calibration takes the full surface.

    Args:
        market_prices: Observed call prices of shape
            ``(n_maturities, n_strikes)``.
        s0: Spot price.
        strikes: Strike grid of shape ``(n_strikes,)`` in float64.
        taus: Maturities in years, one per price row.
        initial: Starting parameters.
        config: Optimiser hyperparameters; defaults applied when omitted.

    Returns:
        The recovered parameters with the loss trajectory.

    Raises:
        ValueError: If the price rows disagree with the maturity count.
    """
    if market_prices.shape[0] != len(taus):
        raise ValueError(
            f"market_prices has {market_prices.shape[0]} rows for {len(taus)} maturities"
        )
    settings = config if config is not None else CalibrationConfig()
    weights = []
    for row, tau in zip(market_prices, taus, strict=True):
        market_vols = implied_vol(row, s0, strikes, tau)
        sqrt_tau = math.sqrt(tau)
        d1 = (torch.log(s0 / strikes) + 0.5 * market_vols**2 * tau) / (market_vols * sqrt_tau)
        vega = s0 * torch.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi) * sqrt_tau
        floored = torch.clamp(vega, min=0.05 * float(vega.max()))
        weights.append((1.0 / floored**2).detach())

    raw = _unconstrain(initial).requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=settings.lr)
    losses: list[float] = []
    for _ in range(settings.n_iterations):
        params = _constrain(raw)
        loss = market_prices.new_zeros(())
        for row, tau, weight in zip(market_prices, taus, weights, strict=True):
            model_prices = price_surface(params, s0, strikes, tau)
            loss = loss + (weight * (model_prices - row) ** 2).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        final = _constrain(raw)
    recovered = HestonParams(
        v0=float(final[0]),
        kappa=float(final[1]),
        theta=float(final[2]),
        xi=float(final[3]),
        rho=float(final[4]),
    )
    return CalibrationResult(params=recovered, final_loss=losses[-1], losses=losses)


@dataclass(frozen=True)
class HestonAnalyticPricer:
    """COS-based vanilla pricer implementing the pricing seam.

    Attributes:
        n_terms: Number of cosine expansion terms.
    """

    n_terms: int = 160

    def price(self, payoff: Payoff, simulator: PathSimulator) -> PriceEstimate:
        """Prices a European call under a Heston simulator's parameters.

        Args:
            payoff: A :class:`EuropeanCall`.
            simulator: A :class:`HestonSimulator` supplying the dynamics.

        Returns:
            The COS price with zero standard error.

        Raises:
            TypeError: If the payoff or simulator is outside scope.
        """
        if not isinstance(simulator, HestonSimulator):
            raise TypeError(
                f"analytic pricer requires HestonSimulator, got {type(simulator).__name__}"
            )
        if not isinstance(payoff, EuropeanCall):
            raise TypeError(f"no closed form for {type(payoff).__name__}")
        params = HestonParams(
            v0=simulator.v0,
            kappa=simulator.kappa,
            theta=simulator.theta,
            xi=simulator.xi,
            rho=simulator.rho,
        )
        strikes = torch.tensor([payoff.strike], dtype=torch.float64)
        value = price_surface(
            params.as_tensors(), simulator.s0, strikes, simulator.maturity, self.n_terms
        )
        return PriceEstimate(value=float(value[0]), standard_error=0.0, provenance="heston-cos")
