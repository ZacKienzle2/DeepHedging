"""Deep hedging research framework.

Neural hedging policies trained against convex risk measures over simulated
market paths under realistic frictions.
"""

from deephedging.accumulators import (
    BarrierAliveAccumulator,
    PathAccumulator,
    RunningMaxAccumulator,
    fold_path,
)
from deephedging.bsde import (
    BSDEConfig,
    BSDEProblem,
    BSDEResult,
    DeepBSDESolver,
    DiscountGenerator,
    ZeroGenerator,
    train_bsde,
)
from deephedging.evaluation import (
    bs_call_delta,
    bs_call_price,
    bs_call_vega,
    bs_put_price,
    delta_hedge_positions,
    expected_shortfall,
    merton_call_price,
    pnl_summary,
)
from deephedging.features import (
    DefaultFeatures,
    FeatureMap,
    MultiAssetFeatures,
    RunningMaxFeatures,
    VarianceFeatures,
)
from deephedging.frictions import CostModel, NoCost, ProportionalCost
from deephedging.instruments import (
    BasketCall,
    EuropeanCall,
    EuropeanPut,
    GeometricBasketCall,
    Payoff,
    UpAndOutCall,
)
from deephedging.market import (
    CorrelatedGBMSimulator,
    GBMSimulator,
    HestonSimulator,
    MarketState,
    MertonSimulator,
    NoiseSpec,
    PathSimulator,
)
from deephedging.market.local_vol import LocalVolSimulator
from deephedging.policies import FeedForwardPolicy, HedgePolicy, RecurrentPolicy
from deephedging.pricing import (
    BlackScholesPricer,
    MonteCarloPricer,
    PriceEstimate,
    Pricer,
)
from deephedging.risk import CVaR, Entropic, RiskMeasure
from deephedging.training import TrainConfig, hedge_pnl, pnl_from_positions, train

__version__ = "0.1.0"

__all__ = [
    "BSDEConfig",
    "BSDEProblem",
    "BSDEResult",
    "BarrierAliveAccumulator",
    "BasketCall",
    "BlackScholesPricer",
    "CVaR",
    "CorrelatedGBMSimulator",
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
    "GeometricBasketCall",
    "HedgePolicy",
    "HestonSimulator",
    "LocalVolSimulator",
    "MarketState",
    "MertonSimulator",
    "MonteCarloPricer",
    "MultiAssetFeatures",
    "NoCost",
    "NoiseSpec",
    "PathAccumulator",
    "Payoff",
    "PathSimulator",
    "PriceEstimate",
    "Pricer",
    "ProportionalCost",
    "RecurrentPolicy",
    "RunningMaxAccumulator",
    "RiskMeasure",
    "RunningMaxFeatures",
    "TrainConfig",
    "UpAndOutCall",
    "VarianceFeatures",
    "ZeroGenerator",
    "bs_call_delta",
    "bs_call_price",
    "bs_call_vega",
    "bs_put_price",
    "delta_hedge_positions",
    "expected_shortfall",
    "fold_path",
    "hedge_pnl",
    "merton_call_price",
    "pnl_from_positions",
    "pnl_summary",
    "train",
    "train_bsde",
]
