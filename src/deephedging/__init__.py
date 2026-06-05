"""Deep hedging research framework.

Neural hedging policies trained against convex risk measures over simulated
market paths under realistic frictions.
"""

from deephedging.evaluation import (
    bs_call_delta,
    bs_call_price,
    bs_call_vega,
    bs_put_price,
    delta_hedge_positions,
    expected_shortfall,
    pnl_summary,
)
from deephedging.frictions import CostModel, NoCost, ProportionalCost
from deephedging.instruments import EuropeanCall, EuropeanPut, Payoff, UpAndOutCall
from deephedging.market import GBMSimulator, HestonSimulator, PathSimulator
from deephedging.policies import FeedForwardPolicy, HedgePolicy, RecurrentPolicy
from deephedging.risk import CVaR, Entropic, RiskMeasure
from deephedging.training import TrainConfig, hedge_pnl, pnl_from_positions, train

__version__ = "0.1.0"

__all__ = [
    "CVaR",
    "CostModel",
    "Entropic",
    "EuropeanCall",
    "EuropeanPut",
    "FeedForwardPolicy",
    "GBMSimulator",
    "HedgePolicy",
    "HestonSimulator",
    "NoCost",
    "Payoff",
    "PathSimulator",
    "ProportionalCost",
    "RecurrentPolicy",
    "RiskMeasure",
    "TrainConfig",
    "UpAndOutCall",
    "bs_call_delta",
    "bs_call_price",
    "bs_call_vega",
    "bs_put_price",
    "delta_hedge_positions",
    "expected_shortfall",
    "hedge_pnl",
    "pnl_from_positions",
    "pnl_summary",
    "train",
]
