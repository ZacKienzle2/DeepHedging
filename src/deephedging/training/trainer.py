"""Training loop for hedging policies."""

from dataclasses import dataclass, field

import torch

from deephedging.features import FeatureMap
from deephedging.frictions.base import CostModel
from deephedging.instruments.base import Payoff
from deephedging.market.base import PathSimulator
from deephedging.policies.base import HedgePolicy
from deephedging.risk.base import RiskMeasure
from deephedging.training.engine import hedge_pnl


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for the training loop.

    Attributes:
        n_iterations: Number of optimisation steps, each on a fresh batch.
        batch_paths: Simulated paths per step; for CVaR at level ``alpha``
            this should scale like ``1 / (1 - alpha)`` to keep enough tail
            samples per batch.
        lr: Learning rate for the policy parameters.
        risk_lr: Learning rate for risk-measure parameters (the CVaR
            threshold); defaults to ``10 * lr`` for a faster timescale so the
            threshold tracks the moving quantile and keeps the policy
            gradient close to the true CVaR gradient.
        seed: Optional seed for reproducible path generation.
        checkpoint_steps: Whether to gradient-checkpoint the episode loop.
        liquidate_terminal: Whether episodes charge terminal liquidation.
    """

    n_iterations: int = 2000
    batch_paths: int = 4096
    lr: float = 1e-3
    risk_lr: float | None = None
    seed: int | None = None
    checkpoint_steps: bool = False
    liquidate_terminal: bool = False


@dataclass
class TrainResult:
    """Outcome of a training run.

    Attributes:
        losses: Risk objective recorded at every iteration.
    """

    losses: list[float] = field(default_factory=list)


def train(
    simulator: PathSimulator,
    policy: HedgePolicy,
    payoff: Payoff,
    cost_model: CostModel,
    risk_measure: RiskMeasure,
    config: TrainConfig,
    premium: float | torch.Tensor = 0.0,
    feature_map: FeatureMap | None = None,
) -> TrainResult:
    """Trains a hedging policy by SGD on a convex risk measure.

    Paths are generated on the fly per iteration: data is unbounded, nothing
    is stored, and no overfitting to a fixed dataset is possible.

    Args:
        simulator: Market path simulator.
        policy: Hedging policy to optimise.
        payoff: Liability payoff.
        cost_model: Transaction cost model.
        risk_measure: Training objective applied to the loss ``-pnl``.
        config: Training hyperparameters.
        premium: Premium received for the liability at inception.
        feature_map: Observation builder forwarded to the episode engine.

    Returns:
        The recorded training losses.
    """
    device = next(policy.parameters()).device
    generator: torch.Generator | None = None
    if config.seed is not None:
        generator = torch.Generator(device=simulator.device)
        generator.manual_seed(config.seed)
    param_groups: list[dict[str, object]] = [{"params": list(policy.parameters()), "lr": config.lr}]
    risk_params = list(risk_measure.parameters())
    if risk_params:
        risk_lr = config.risk_lr if config.risk_lr is not None else 10.0 * config.lr
        param_groups.append({"params": risk_params, "lr": risk_lr})
    optimizer = torch.optim.Adam(param_groups)
    with torch.no_grad():
        warmup_paths = simulator.simulate(config.batch_paths, generator=generator).to(device)
        warmup_pnl = hedge_pnl(
            warmup_paths,
            policy,
            payoff,
            cost_model,
            premium=premium,
            liquidate_terminal=config.liquidate_terminal,
            feature_map=feature_map,
        )
        risk_measure.warm_start(-warmup_pnl)
    loss_history: list[torch.Tensor] = []
    for _ in range(config.n_iterations):
        paths = simulator.simulate(config.batch_paths, generator=generator).to(device)
        pnl = hedge_pnl(
            paths,
            policy,
            payoff,
            cost_model,
            premium=premium,
            liquidate_terminal=config.liquidate_terminal,
            checkpoint_steps=config.checkpoint_steps,
            feature_map=feature_map,
        )
        loss = risk_measure(-pnl)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach())
    recorded = torch.stack(loss_history).cpu() if loss_history else torch.empty(0)
    return TrainResult(losses=recorded.tolist())
