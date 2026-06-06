"""Pricing seam unifying closed-form and Monte Carlo backends.

Every backend returns a :class:`PriceEstimate` rather than a bare float,
so the Monte Carlo standard error and the provenance survive the seam.
Calibration objectives and golden tests can then compare backends
like with like instead of treating a noisy estimate as exact.
"""

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from deephedging.evaluation.black_scholes import bs_call_price, bs_put_price
from deephedging.instruments.barrier import UpAndOutCall
from deephedging.instruments.base import Payoff
from deephedging.instruments.vanilla import EuropeanCall, EuropeanPut
from deephedging.market.base import PathSimulator
from deephedging.market.gbm import GBMSimulator
from deephedging.market.noise import NoiseSpec
from deephedging.market.state import PathFolds


@dataclass(frozen=True)
class PriceEstimate:
    """A price with its uncertainty and origin.

    Attributes:
        value: Point estimate of the price.
        standard_error: Sampling standard error; zero for closed forms.
        provenance: Identifier of the producing backend.
    """

    value: float
    standard_error: float
    provenance: str


@runtime_checkable
class Pricer(Protocol):
    """Prices a payoff under a simulator's dynamics."""

    def price(self, payoff: Payoff, simulator: PathSimulator) -> PriceEstimate:
        """Computes the price estimate.

        Args:
            payoff: The liability to price.
            simulator: Market dynamics under the pricing measure.

        Returns:
            The price with uncertainty and provenance.
        """
        ...


@dataclass(frozen=True)
class BlackScholesPricer:
    """Closed-form pricer for vanilla payoffs under lognormal dynamics.

    Attributes:
        rate: Continuously compounded interest rate.
    """

    rate: float = 0.0

    def price(self, payoff: Payoff, simulator: PathSimulator) -> PriceEstimate:
        """Prices a European call or put under GBM dynamics.

        Args:
            payoff: A :class:`EuropeanCall` or :class:`EuropeanPut`.
            simulator: A :class:`GBMSimulator`; its spot, volatility, and
                maturity parameterise the closed form.

        Returns:
            The exact price with zero standard error.

        Raises:
            TypeError: If the payoff or simulator is outside the closed
                form's scope.
            ValueError: If the simulator drift disagrees with the pricing
                rate, which would price under the wrong measure.
        """
        if not isinstance(simulator, GBMSimulator):
            raise TypeError(f"closed form requires GBMSimulator, got {type(simulator).__name__}")
        if simulator.mu != self.rate:
            raise ValueError(
                f"risk-neutral pricing needs simulator drift {self.rate}, got {simulator.mu}"
            )
        if isinstance(payoff, EuropeanCall):
            value = bs_call_price(
                simulator.s0, payoff.strike, simulator.sigma, simulator.maturity, self.rate
            )
        elif isinstance(payoff, EuropeanPut):
            value = bs_put_price(
                simulator.s0, payoff.strike, simulator.sigma, simulator.maturity, self.rate
            )
        else:
            raise TypeError(f"no closed form for {type(payoff).__name__}")
        return PriceEstimate(value=float(value), standard_error=0.0, provenance="black-scholes")


def _payoff_from_folds(payoff: Payoff, folds: PathFolds) -> torch.Tensor | None:
    if isinstance(payoff, EuropeanCall):
        return torch.clamp(folds.terminal - payoff.strike, min=0.0)
    if isinstance(payoff, EuropeanPut):
        return torch.clamp(payoff.strike - folds.terminal, min=0.0)
    if isinstance(payoff, UpAndOutCall):
        alive = (folds.running_max < payoff.barrier).to(folds.terminal.dtype)
        return torch.clamp(folds.terminal - payoff.strike, min=0.0) * alive
    return None


@dataclass(frozen=True)
class MonteCarloPricer:
    """Model-agnostic pricer by discounted Monte Carlo expectation.

    Prices any payoff under any simulator at the cost of sampling noise,
    reported honestly through the standard error. The discount applies
    the pricer's rate over the simulator's maturity, so the estimate
    stays comparable with discounting closed forms away from the
    zero-rate path where the two would otherwise silently disagree.
    When the simulator offers register-resident path folds and the
    payoff is expressible from them, pricing skips the grid entirely;
    the folds consume the same noise stream, so the two routes return
    the same estimate.

    Attributes:
        n_paths: Number of simulated paths.
        seed: Noise seed for a reproducible estimate.
        rate: Continuously compounded discount rate; must equal the
            simulator drift for a risk-neutral price.
    """

    n_paths: int = 200_000
    seed: int = 0
    rate: float = 0.0

    def price(self, payoff: Payoff, simulator: PathSimulator) -> PriceEstimate:
        """Estimates the discounted expectation of the payoff.

        Args:
            payoff: The liability to price.
            simulator: Market dynamics under the pricing measure.

        Returns:
            The discounted sample mean with its standard error.
        """
        values: torch.Tensor | None = None
        fold_source = getattr(simulator, "simulate_folds", None)
        fold_expressible = isinstance(payoff, (EuropeanCall, EuropeanPut, UpAndOutCall))
        if fold_source is not None and fold_expressible:
            folds = fold_source(self.n_paths, noise=NoiseSpec(seed=self.seed))
            values = _payoff_from_folds(payoff, folds)
        if values is None:
            state = simulator.simulate(self.n_paths, noise=NoiseSpec(seed=self.seed))
            values = payoff(state.spot)
        discount = math.exp(-self.rate * simulator.maturity)
        mean = float(values.mean()) * discount
        spread = float(values.std()) / math.sqrt(self.n_paths) * discount
        return PriceEstimate(value=mean, standard_error=spread, provenance="monte-carlo")
