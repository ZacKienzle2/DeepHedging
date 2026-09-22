# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.


import torch
from hypothesis import given
from hypothesis import strategies as st

import deephedging
import deephedging.baselines
from deephedging import (
    BidAskCost,
    EuropeanCall,
    EuropeanPut,
    FeedForwardPolicy,
    GBMSimulator,
    MarketState,
    NoCost,
    NoiseSpec,
    NoTransactionBandPolicy,
    PowerLawImpactCost,
    ProportionalCost,
    RecurrentPolicy,
)

_WIDTHS = st.lists(st.integers(1, 8), min_size=1, max_size=2).map(tuple)


def _grid(n_paths: int, n_steps: int, seed: int) -> MarketState:
    simulator = GBMSimulator(
        s0=100.0, sigma=0.2, maturity=1.0, n_steps=n_steps, dtype=torch.float64
    )
    return simulator.simulate(n_paths, noise=NoiseSpec(seed=seed))


@given(
    amp=st.just(False),
    checkpoint_steps=st.booleans(),
    cost_model=st.one_of(
        st.just(NoCost()),
        st.builds(ProportionalCost, rate=st.floats(0.0, 0.01)),
        st.builds(
            BidAskCost,
            bid_half_spread=st.floats(0.0, 0.01),
            ask_half_spread=st.floats(0.0, 0.01),
        ),
        st.builds(
            PowerLawImpactCost, coefficient=st.floats(0.0, 0.01), exponent=st.floats(1.0, 2.0)
        ),
    ),
    feature_map=st.none(),
    liquidate_terminal=st.booleans(),
    payoff=st.one_of(
        st.builds(EuropeanCall, strike=st.floats(80.0, 120.0)),
        st.builds(EuropeanPut, strike=st.floats(80.0, 120.0)),
    ),
    policy=st.one_of(
        st.builds(
            NoTransactionBandPolicy,
            hidden_sizes=_WIDTHS,
            maturity=st.floats(0.1, 2.0),
            n_features=st.just(3),
            sigma=st.floats(0.05, 0.5),
            strike_ratio=st.floats(0.8, 1.2),
        ),
        st.builds(
            FeedForwardPolicy,
            hidden_sizes=_WIDTHS,
            n_features=st.just(3),
            n_outputs=st.just(1),
        ),
        st.builds(
            RecurrentPolicy,
            hidden_size=st.integers(1, 8),
            n_features=st.just(3),
        ),
    ).map(lambda policy: policy.double()),
    premium=st.floats(-10.0, 10.0),
    state=st.builds(
        _grid,
        n_paths=st.integers(1, 16),
        n_steps=st.integers(1, 8),
        seed=st.integers(0, 2**16),
    ),
)
def test_equivalent_hedge_pnl_settled_per_date_hedge_pnl(
    amp: bool,
    checkpoint_steps: bool,
    cost_model: deephedging.CostModel,
    feature_map: deephedging.FeatureMap | None,
    liquidate_terminal: bool,
    payoff: deephedging.Payoff,
    policy: deephedging.HedgePolicy,
    premium: float,
    state: deephedging.MarketState,
) -> None:
    result_hedge_pnl_settled_per_date = deephedging.baselines.hedge_pnl_settled_per_date(
        state=state,
        policy=policy,
        payoff=payoff,
        cost_model=cost_model,
        premium=premium,
        liquidate_terminal=liquidate_terminal,
        checkpoint_steps=checkpoint_steps,
        feature_map=feature_map,
    )
    result_hedge_pnl = deephedging.hedge_pnl(
        state=state,
        policy=policy,
        payoff=payoff,
        cost_model=cost_model,
        premium=premium,
        liquidate_terminal=liquidate_terminal,
        checkpoint_steps=checkpoint_steps,
        feature_map=feature_map,
        amp=amp,
    )
    torch.testing.assert_close(result_hedge_pnl, result_hedge_pnl_settled_per_date)
