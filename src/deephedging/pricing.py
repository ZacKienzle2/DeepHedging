"""Pricing seam unifying closed-form and Monte Carlo backends.

Every backend returns a :class:`PriceEstimate` rather than a bare float,
so the Monte Carlo standard error and the provenance survive the seam.
Calibration objectives and golden tests can then compare backends
like with like instead of treating a noisy estimate as exact.
"""

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from deephedging.evaluation.black_scholes import bs_call_price, bs_put_price
from deephedging.instruments.base import Payoff
from deephedging.instruments.vanilla import EuropeanCall, EuropeanPut
from deephedging.market.base import PathSimulator
from deephedging.market.gbm import GBMSimulator
from deephedging.market.noise import NoiseSpec


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
        """
        if not isinstance(simulator, GBMSimulator):
            raise TypeError(f"closed form requires GBMSimulator, got {type(simulator).__name__}")
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


@dataclass(frozen=True)
class MonteCarloPricer:
    """Model-agnostic pricer by plain Monte Carlo expectation.

    Prices any payoff under any simulator at the cost of sampling noise,
    reported honestly through the standard error. Uses the simulator's
    own measure; for arbitrage-free prices the simulator drift must be
    risk-neutral.

    Attributes:
        n_paths: Number of simulated paths.
        seed: Noise seed for a reproducible estimate.
    """

    n_paths: int = 200_000
    seed: int = 0

    def price(self, payoff: Payoff, simulator: PathSimulator) -> PriceEstimate:
        """Estimates the undiscounted expectation of the payoff.

        Args:
            payoff: The liability to price.
            simulator: Market dynamics under the pricing measure.

        Returns:
            The sample mean with its standard error.
        """
        state = simulator.simulate(self.n_paths, noise=NoiseSpec(seed=self.seed))
        values = payoff(state.spot)
        mean = float(values.mean())
        spread = float(values.std()) / math.sqrt(self.n_paths)
        return PriceEstimate(value=mean, standard_error=spread, provenance="monte-carlo")
