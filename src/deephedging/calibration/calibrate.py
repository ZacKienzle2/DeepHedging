"""Heston surface calibration and the analytic pricer seam.

Calibration minimises vega-weighted price residuals, which agree with
implied-volatility residuals to first order without running a root-find
inside every optimiser step. Weights are computed once from the market
quotes and detached; they are an objective design choice, not a
variable. Parameters are optimised through the softplus and tanh bijectors
of ``torch.distributions`` so the optimiser is unconstrained while the model
always sees admissible values, and the COS pricer's truncation bounds are
recomputed each evaluation from detached cumulants so the expansion tracks
the moving parameters without contaminating their gradients. The objective
is smooth, deterministic and five-dimensional, the case quasi-Newton methods
are built for, so L-BFGS with a strong Wolfe line search minimises it: on
the golden surface it reached a loss of 1e-17 in 28 evaluations where 800
Adam steps stopped at 4e-8.
"""

import math
from dataclasses import dataclass, field
from typing import cast

import torch
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    SoftplusTransform,
    TanhTransform,
    Transform,
)

from deephedging.calibration.cf import cos_call_price
from deephedging.calibration.heston_cf import HestonParams, heston_cf, heston_cumulants
from deephedging.calibration.implied_vol import implied_vol
from deephedging.evaluation.black_scholes import bs_call_vega
from deephedging.instruments.base import Payoff
from deephedging.instruments.vanilla import EuropeanCall
from deephedging.market.base import PathSimulator
from deephedging.market.heston import HestonSimulator
from deephedging.pricing import PriceEstimate

_RHO_BOUND = 0.999
_POSITIVE = SoftplusTransform()
_CORRELATION = ComposeTransform([TanhTransform(), AffineTransform(0.0, _RHO_BOUND)])


def _apply(transform: Transform, value: torch.Tensor) -> torch.Tensor:
    return cast("torch.Tensor", transform(value))


def _constrain(raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (*_apply(_POSITIVE, raw[:4]).unbind(), _apply(_CORRELATION, raw[4]))


def _unconstrain(params: HestonParams) -> torch.Tensor:
    positive = torch.tensor(
        [params.v0, params.kappa, params.theta, params.xi], dtype=torch.float64
    ).clamp(min=1e-8)
    rho = torch.tensor([params.rho], dtype=torch.float64).clamp(-(_RHO_BOUND**2), _RHO_BOUND**2)
    return torch.cat((_apply(_POSITIVE.inv, positive), _apply(_CORRELATION.inv, rho)))


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
        n_iterations: Maximum L-BFGS iterations.
        lr: Initial L-BFGS step length, which the line search then adapts.
    """

    n_iterations: int = 100
    lr: float = 1.0


@dataclass
class CalibrationResult:
    """Outcome of a calibration run.

    Attributes:
        params: Recovered model parameters.
        final_loss: Vega-weighted squared residual at the last evaluation.
        losses: Loss recorded at every objective evaluation.
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

    A single-maturity strip underdetermines the parameters. The smile
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

    Quotes whose prices violate the no-arbitrage band invert to NaN and
    receive zero weight, so a handful of bad market points cannot poison
    the objective.

    Returns:
        The recovered parameters with the loss trajectory.

    Raises:
        ValueError: If the price rows disagree with the maturity count,
            if any maturity is non-positive, or if a maturity has no
            invertible quote at all.
    """
    if market_prices.shape[0] != len(taus):
        msg = f"market_prices has {market_prices.shape[0]} rows for {len(taus)} maturities"
        raise ValueError(msg)
    if not taus or any(tau <= 0.0 for tau in taus):
        msg = f"taus must be non-empty and positive, got {taus}"
        raise ValueError(msg)
    settings = config if config is not None else CalibrationConfig()
    weights = []
    for row, tau in zip(market_prices, taus, strict=True):
        market_vols = implied_vol(row, s0, strikes, tau)
        usable = torch.isfinite(market_vols)
        if not bool(usable.any()):
            msg = f"no quote at maturity {tau} inverts to a finite volatility"
            raise ValueError(msg)
        safe_vols = torch.where(usable, market_vols, torch.ones_like(market_vols))
        vega = bs_call_vega(s0, strikes, safe_vols, tau)
        floored = torch.clamp(vega, min=0.05 * float(vega[usable].max()))
        weights.append((usable.to(vega.dtype) / floored**2).detach())

    raw = _unconstrain(initial).requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [raw], lr=settings.lr, max_iter=settings.n_iterations, line_search_fn="strong_wolfe"
    )
    losses: list[float] = []

    def objective() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        params = _constrain(raw)
        loss = market_prices.new_zeros(())
        for row, tau, weight in zip(market_prices, taus, weights, strict=True):
            model_prices = price_surface(params, s0, strikes, tau)
            loss = loss + (weight * (model_prices - row) ** 2).sum()
        loss.backward()
        losses.append(float(loss.detach()))
        return loss

    optimizer.step(objective)
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
            ValueError: If the simulator carries a non-zero drift, which
                the risk-neutral closed form cannot represent.
        """
        if not isinstance(simulator, HestonSimulator):
            msg = f"analytic pricer requires HestonSimulator, got {type(simulator).__name__}"
            raise TypeError(msg)
        if not isinstance(payoff, EuropeanCall):
            msg = f"no closed form for {type(payoff).__name__}"
            raise TypeError(msg)
        if simulator.mu != 0.0:
            msg = (
                "analytic pricer assumes risk-neutral zero-drift dynamics; "
                f"simulator carries mu={simulator.mu}"
            )
            raise ValueError(msg)
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
