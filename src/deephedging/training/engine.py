"""Hedging episode simulation."""

from typing import cast

import torch
from torch.utils.checkpoint import checkpoint

from deephedging.frictions.base import CostModel
from deephedging.instruments.base import Payoff
from deephedging.policies.base import HedgePolicy


def hedge_pnl(
    paths: torch.Tensor,
    policy: HedgePolicy,
    payoff: Payoff,
    cost_model: CostModel,
    premium: float | torch.Tensor = 0.0,
    liquidate_terminal: bool = False,
    checkpoint_steps: bool = False,
) -> torch.Tensor:
    """Computes the terminal PnL of a self-financed hedge along paths.

    Runs the sequential time-major episode loop: at each rebalancing date the
    policy maps ``(log moneyness, time to maturity, previous position)`` to a
    position, trading gains accumulate as ``pos_t * (S_{t+1} - S_t)``, and
    transaction costs are charged on every position change including the
    initial trade from a flat book.

    Args:
        paths: Price paths of shape ``(n_steps + 1, n_paths)``.
        policy: Hedging policy network.
        payoff: Liability payoff, charged at maturity.
        cost_model: Transaction cost model.
        premium: Premium received for the liability at inception.
        liquidate_terminal: Whether to charge the cost of closing the final
            position at the terminal price.
        checkpoint_steps: Whether to gradient-checkpoint each policy call,
            trading recompute for ``O(1/T)`` activation memory. Validates the
            recompute contract the CUDA backward will rely on.

    Returns:
        PnL per path of shape ``(n_paths,)``; positive is profit.
    """
    n_steps = paths.shape[0] - 1
    n_paths = paths.shape[1]
    spot0 = paths[0]
    taus = torch.arange(n_steps, 0, -1, dtype=paths.dtype, device=paths.device) / n_steps
    position = paths.new_zeros(n_paths)
    pnl = paths.new_zeros(n_paths) + premium
    state: torch.Tensor | None = None
    use_checkpoint = checkpoint_steps and torch.is_grad_enabled()
    for t in range(n_steps):
        spot = paths[t]
        tau = taus[t].expand(n_paths)
        features = torch.stack((torch.log(spot / spot0), tau, position), dim=-1)
        if use_checkpoint:
            output = checkpoint(policy, features, state, use_reentrant=False)
            new_position, state = cast(tuple[torch.Tensor, torch.Tensor | None], output)
        else:
            new_position, state = policy(features, state)
        pnl = pnl + new_position * (paths[t + 1] - spot)
        pnl = pnl - cost_model(new_position - position, spot)
        position = new_position
    if liquidate_terminal:
        pnl = pnl - cost_model(position, paths[-1])
    return pnl - payoff(paths)


def pnl_from_positions(
    paths: torch.Tensor,
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
        paths: Price paths of shape ``(n_steps + 1, n_paths)``.
        positions: Hedge positions of shape ``(n_steps, n_paths)``; row ``t``
            is held over the interval ``[t, t + 1)``.
        payoff: Liability payoff, charged at maturity.
        cost_model: Transaction cost model.
        premium: Premium received for the liability at inception.
        liquidate_terminal: Whether to charge the cost of closing the final
            position at the terminal price.

    Returns:
        PnL per path of shape ``(n_paths,)``; positive is profit.
    """
    gains = (positions * (paths[1:] - paths[:-1])).sum(dim=0)
    initial = positions.new_zeros((1, positions.shape[1]))
    trades = torch.diff(positions, dim=0, prepend=initial)
    costs = cost_model(trades, paths[:-1]).sum(dim=0)
    if liquidate_terminal:
        costs = costs + cost_model(positions[-1], paths[-1])
    return premium + gains - costs - payoff(paths)
