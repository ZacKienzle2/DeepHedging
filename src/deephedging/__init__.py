"""Deep hedging research framework.

Neural hedging policies trained against convex risk measures over simulated
market paths under realistic frictions.
"""

from deephedging.bsde import (
    BSDEConfig,
    BSDEProblem,
    BSDEResult,
    DeepBSDESolver,
    DiscountGenerator,
    ZeroGenerator,
    train_bsde,
)
from deephedging.features import DefaultFeatures, FeatureMap, RunningMaxFeatures
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
    "BSDEConfig",
    "BSDEProblem",
    "BSDEResult",
    "CVaR",
    "CostModel",
    "DeepBSDESolver",
    "DefaultFeatures",
    "DiscountGenerator",
    "Entropic",
    "FeatureMap",
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
    "RunningMaxFeatures",
    "TrainConfig",
    "UpAndOutCall",
    "ZeroGenerator",
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
    "train_bsde",
]
