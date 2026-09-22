"""Hedging episode simulation.

Only the policy recursion is sequential: the position at a date is an input
to the next date's observation, so the network runs once per date. Trading
gains and transaction costs depend on the whole position path but on no
later decision, so they are computed once over the stacked positions after
the recursion rather than inside it. This is loop distribution in the sense
of Allen and Kennedy's vectorisation algorithm: the statements outside the
dependence cycle leave the loop and run as whole-grid kernels, which cuts
the per-date launch count the eager engine is bound by.
"""

from typing import cast

import torch
from torch.utils.checkpoint import checkpoint

from deephedging.features import DefaultFeatures, FeatureMap
from deephedging.frictions.base import CostModel
from deephedging.instruments.base import Payoff
from deephedging.market.state import MarketState
from deephedging.policies.base import HedgePolicy


def _per_path(values: torch.Tensor) -> torch.Tensor:
    return values.sum(dim=-1) if values.dim() == 2 else values


def _positions(
    state: MarketState,
    policy: HedgePolicy,
    checkpoint_steps: bool,
    feature_map: FeatureMap | None,
    amp: bool,
) -> torch.Tensor:
    paths = state.spot
    n_steps = state.n_steps
    features_of = feature_map if feature_map is not None else DefaultFeatures()
    taus = torch.arange(n_steps, 0, -1, dtype=paths.dtype, device=paths.device) / n_steps
    position = paths.new_zeros(paths.shape[1:])
    hidden: torch.Tensor | None = None
    use_checkpoint = checkpoint_steps and torch.is_grad_enabled()
    device_type = paths.device.type
    held: list[torch.Tensor] = []
    for t in range(n_steps):
        features = features_of(state, t, taus[t], position)
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=amp, cache_enabled=False
        ):
            if use_checkpoint:
                output = checkpoint(policy, features, hidden, use_reentrant=False)
                new_position, hidden = cast("tuple[torch.Tensor, torch.Tensor | None]", output)
            else:
                new_position, hidden = policy(features, hidden)
        position = new_position.to(paths.dtype)
        held.append(position)
    return torch.stack(held)


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

    Runs the time-major policy recursion, where the policy maps the
    feature-map output at each rebalancing date to a position, then
    settles the whole position path at once through
    :func:`pnl_from_positions`. Trading gains accrue as
    ``pos_t * (S_{t+1} - S_t)`` and transaction costs are charged on every
    position change including the initial trade from a flat book. With a
    trailing asset axis on the spot grid the policy emits one position per
    asset and gains and costs contract that axis.

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
            The casts are extra kernels, so an eager loop that is already
            bound by kernel launches gains little; under whole-iteration
            capture the launches are free and the halved activation
            traffic cut the 64-wide generated training benchmark by a
            third. Each autocast region spans one date, so its weight
            cast cache is disabled, which is also what capture requires.

    Returns:
        PnL per path of shape ``(n_paths,)``; positive is profit.
    """
    positions = _positions(state, policy, checkpoint_steps, feature_map, amp)
    return pnl_from_positions(state, positions, payoff, cost_model, premium, liquidate_terminal)


def pnl_from_positions(
    state: MarketState,
    positions: torch.Tensor,
    payoff: Payoff,
    cost_model: CostModel,
    premium: float | torch.Tensor = 0.0,
    liquidate_terminal: bool = False,
) -> torch.Tensor:
    """Computes hedged PnL for a whole position path, fully vectorised.

    Settles the positions :func:`hedge_pnl` produces, and serves analytic
    baselines (for example the Black-Scholes delta hedge) whose positions
    do not depend on the episode loop.

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
    gains = _per_path((positions * torch.diff(paths, dim=0)).sum(dim=0))
    trades = torch.diff(positions, dim=0, prepend=torch.zeros_like(positions[:1]))
    costs = _per_path(cost_model(trades, paths[:-1]).sum(dim=0))
    if liquidate_terminal:
        costs = costs + _per_path(cost_model(positions[-1], paths[-1]))
    return premium + gains - costs - payoff(paths)
