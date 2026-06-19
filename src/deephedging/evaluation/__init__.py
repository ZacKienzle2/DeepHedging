"""Analytic baselines and risk metrics for evaluation."""

from deephedging.evaluation.american import binomial_american_put, lsm_american_put
from deephedging.evaluation.black_scholes import (
    bs_call_delta,
    bs_call_price,
    bs_call_vega,
    bs_put_price,
    delta_hedge_positions,
)
from deephedging.evaluation.greeks import Greeks, european_greeks
from deephedging.evaluation.inference import (
    BootstrapInterval,
    PairedComparison,
    bootstrap_metric,
    paired_bootstrap,
)
from deephedging.evaluation.merton import merton_call_price
from deephedging.evaluation.metrics import expected_shortfall, pnl_summary

__all__ = [
    "BootstrapInterval",
    "Greeks",
    "PairedComparison",
    "binomial_american_put",
    "bootstrap_metric",
    "bs_call_delta",
    "bs_call_price",
    "bs_call_vega",
    "bs_put_price",
    "delta_hedge_positions",
    "european_greeks",
    "expected_shortfall",
    "lsm_american_put",
    "merton_call_price",
    "paired_bootstrap",
    "pnl_summary",
]
