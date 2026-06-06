"""Hedging episode simulation."""

from typing import cast

import torch
from torch.utils.checkpoint import checkpoint

from deephedging.features import DefaultFeatures, FeatureMap
from deephedging.frictions.base import CostModel
from deephedging.instruments.base import Payoff
from deephedging.market.state import MarketState
from deephedging.policies.base import HedgePolicy


def hedge_pnl(
    state: MarketState,
    policy: HedgePolicy,
    payoff: Payoff,
    cost_model: CostModel,
    premium: float | torch.Tensor = 0.0,
    liquidate_terminal: bool = False,
    checkpoint_steps: bool = False,
    feature_map: FeatureMap | None = None,
    amp: bool = False,
) -> torch.Tensor:
    """Computes the terminal PnL of a self-financed hedge along paths.

    Runs the sequential time-major episode loop. At each rebalancing date
    the policy maps the feature-map output to a position, trading gains
    accumulate as ``pos_t * (S_{t+1} - S_t)``, and transaction costs are
    charged on every position change including the initial trade from a
    flat book. With a trailing asset axis on the spot grid the policy
    emits one position per asset and gains and costs contract that axis.

    Args:
        state: Simulated market state.
        policy: Hedging policy network.
        payoff: Liability payoff, charged at maturity.
        cost_model: Transaction cost model.
        premium: Premium received for the liability at inception.
        liquidate_terminal: Whether to charge the cost of closing the final
            position at the terminal price.
        checkpoint_steps: Whether to gradient-checkpoint each policy call,
            trading recompute for ``O(1/T)`` activation memory. Validates the
            recompute contract the CUDA backward will rely on.
        feature_map: Observation builder; defaults to
            :class:`~deephedging.features.DefaultFeatures`.
        amp: Whether to run the policy network under bfloat16 autocast.
            Only the network matmuls run in reduced precision; positions
            are cast back so the PnL state, the cost accounting, and the
            risk reduction stay in the path dtype, where systematic
            rounding bias would otherwise survive Monte Carlo averaging.
            Pays off only for networks wide enough to engage tensor
            cores; at the 64-wide default the per-step cast overhead
            exceeds the matmul saving and the benchmark runs faster
            with this off.

    Returns:
        PnL per path of shape ``(n_paths,)``; positive is profit.
    """
    paths = state.spot
    n_steps = state.n_steps
    n_paths = state.n_paths

    def per_path(values: torch.Tensor) -> torch.Tensor:
        return values.sum(dim=-1) if values.dim() == 2 else values

    features_of = feature_map if feature_map is not None else DefaultFeatures()
    taus = torch.arange(n_steps, 0, -1, dtype=paths.dtype, device=paths.device) / n_steps
    position = paths.new_zeros(paths.shape[1:])
    pnl = paths.new_zeros(n_paths) + premium
    hidden: torch.Tensor | None = None
    use_checkpoint = checkpoint_steps and torch.is_grad_enabled()
    device_type = paths.device.type
    for t in range(n_steps):
        spot = paths[t]
        features = features_of(state, t, taus[t], position)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp):
            if use_checkpoint:
                output = checkpoint(policy, features, hidden, use_reentrant=False)
                new_position, hidden = cast(tuple[torch.Tensor, torch.Tensor | None], output)
            else:
                new_position, hidden = policy(features, hidden)
        new_position = new_position.to(paths.dtype)
        pnl = (
            pnl
            + per_path(new_position * (paths[t + 1] - spot))
            - per_path(cost_model(new_position - position, spot))
        )
        position = new_position
    if liquidate_terminal:
        pnl = pnl - per_path(cost_model(position, paths[-1]))
    return pnl - payoff(paths)


def pnl_from_positions(
    state: MarketState,
    positions: torch.Tensor,
    payoff: Payoff,
    cost_model: CostModel,
    premium: float | torch.Tensor = 0.0,
    liquidate_terminal: bool = False,
) -> torch.Tensor:
    """Computes hedged PnL for precomputed positions, fully vectorised.

    Used by analytic baselines (for example the Black-Scholes delta hedge)
    whose positions do not depend on the episode loop, and as a
    cross-validation oracle for :func:`hedge_pnl`.

    Args:
        state: Simulated market state.
        positions: Hedge positions of shape ``(n_steps, n_paths)`` for a
            single asset or ``(n_steps, n_paths, n_assets)`` with the
            trailing asset axis contracted in the gains and cost
            reductions; row ``t`` is held over the interval ``[t, t + 1)``.
        payoff: Liability payoff, charged at maturity.
        cost_model: Transaction cost model.
        premium: Premium received for the liability at inception.
        liquidate_terminal: Whether to charge the cost of closing the final
            position at the terminal price.

    Returns:
        PnL per path of shape ``(n_paths,)``; positive is profit.
    """
    paths = state.spot

    def per_path(values: torch.Tensor) -> torch.Tensor:
        return values.sum(dim=-1) if values.dim() == 2 else values

    gains = per_path((positions * (paths[1:] - paths[:-1])).sum(dim=0))
    initial = positions.new_zeros((1, *positions.shape[1:]))
    trades = torch.diff(positions, dim=0, prepend=initial)
    costs = per_path(cost_model(trades, paths[:-1]).sum(dim=0))
    if liquidate_terminal:
        costs = costs + per_path(cost_model(positions[-1], paths[-1]))
    return premium + gains - costs - payoff(paths)
