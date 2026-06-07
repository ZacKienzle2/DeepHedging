"""Training loop for hedging policies."""

import os
import warnings
from dataclasses import dataclass, field
from typing import Protocol, cast

import torch
from torch.utils.checkpoint import checkpoint

from deephedging.features import FeatureMap
from deephedging.frictions.base import CostModel
from deephedging.instruments.base import Payoff
from deephedging.market.base import PathSimulator
from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState
from deephedging.market.tilted import LOG_WEIGHT_CHANNEL
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
        seed: Optional experiment seed; each iteration draws from its own
            addressable noise stream, so any single batch can be replayed.
        checkpoint_steps: Whether to gradient-checkpoint the episode loop.
        liquidate_terminal: Whether episodes charge terminal liquidation.
        compile_policy: Whether to wrap the policy in ``torch.compile``.
            Compiles only the per-step network, never the episode loop,
            because the feature-map indirection and checkpoint branch
            would force graph breaks. Requires a working host compiler
            toolchain; incompatible with ``checkpoint_steps``.
        amp: Whether to run the policy network under bfloat16 autocast;
            see :func:`~deephedging.training.engine.hedge_pnl`.
        graph_episode: Whether to capture the whole episode, forward and
            backward through loss, in one CUDA graph and replay it per
            iteration. The training loop is dispatch-bound at small
            network widths, with the host issuing tens of microsecond
            kernels while the device idles; replay collapses the entire
            iteration into one launch. The episode is the correct
            capture unit because a graphed callable keeps static saved
            activations, so capturing the per-step policy and replaying
            it many times before a single backward corrupts gradients.
            Auxiliary channels such as the Heston variance are passed
            alongside the spot grid in sorted key order. Capture keeps
            a private memory pool that roughly doubles the resident
            activation footprint, and a batch that overflows physical
            device memory degrades silently into host paging at a
            throughput collapse of more than an order of magnitude, so
            the trainer warns when the post-capture reservation
            approaches the device capacity, tightening the threshold on
            Windows where the display driver pages under pressure. Combining capture with
            gradient checkpointing trades replayed recompute for that
            footprint and lifts the batch ceiling. Requires CUDA and a
            fixed batch size; mutually exclusive with compilation and
            autocast.
        regenerate_paths: Whether to drop the path grid after the
            forward pass and regenerate it from its noise stream inside
            the backward. Every simulator replays bitwise from a
            ``NoiseSpec``, so the regenerated grid is the recompute
            contract gradient checkpointing needs, and combining this
            flag with ``checkpoint_steps`` reduces peak activation
            memory to roughly one rebalancing step. The lever matters
            for batches large enough that the grid and its derived
            caches crowd device memory; small batches pay recompute for
            nothing. Requires a seed, because an unseeded recompute
            would draw fresh paths and silently corrupt gradients, and
            excludes ``graph_episode``, whose capture freezes the noise
            pair at capture time so a replayed regeneration would train
            every iteration on the same batch.
        graph_generate: Whether the captured graph also generates its
            batch. The fused kernels offer an offset variant that reads
            the shifted Philox subsequence from device memory at launch
            rather than from a frozen host argument, so updating one
            element in place redraws a fresh batch on every replay and
            the entire training iteration, generation included, becomes
            a single graph launch. Requires ``graph_episode``, a seed
            to address the streams, and a simulator exposing
            ``simulate_with_offset``.
    """

    n_iterations: int = 2000
    batch_paths: int = 4096
    lr: float = 1e-3
    risk_lr: float | None = None
    seed: int | None = None
    checkpoint_steps: bool = False
    liquidate_terminal: bool = False
    compile_policy: bool = False
    amp: bool = False
    graph_episode: bool = False
    regenerate_paths: bool = False
    graph_generate: bool = False

    def __post_init__(self) -> None:
        if self.compile_policy and self.checkpoint_steps:
            raise ValueError("compile_policy and checkpoint_steps are mutually exclusive")
        if self.graph_episode and (self.compile_policy or self.amp):
            raise ValueError("graph_episode excludes compilation and autocast")
        if self.regenerate_paths and self.graph_episode:
            raise ValueError("regenerate_paths and graph_episode are mutually exclusive")
        if self.regenerate_paths and self.seed is None:
            raise ValueError("regenerate_paths requires a seed for deterministic replay")
        if self.graph_generate and not self.graph_episode:
            raise ValueError("graph_generate requires graph_episode")
        if self.graph_generate and self.seed is None:
            raise ValueError("graph_generate requires a seed to address the streams")


class _OffsetSimulator(Protocol):
    """Simulator whose kernel reads the Philox subsequence from the device."""

    def simulate_with_offset(self, n_paths: int, seed: int, offset: torch.Tensor) -> MarketState:
        """Simulates a batch addressed by the device-resident offset."""
        ...


def _importance_weights(state: MarketState) -> torch.Tensor | None:
    log_weight = state.aux.get(LOG_WEIGHT_CHANNEL)
    if log_weight is None:
        return None
    return torch.exp(log_weight.to(torch.float64))


class _EpisodeLoss(torch.nn.Module):
    """Whole-iteration module mapping market tensors to the risk objective.

    Registers the policy and the risk measure as submodules so a CUDA
    graph capture of this module records their forward and backward
    passes and accumulates gradients into their parameters. Auxiliary
    channels are passed positionally in a fixed key order, because graph
    capture requires a stable tensor-only signature.
    """

    def __init__(
        self,
        policy: HedgePolicy,
        payoff: Payoff,
        cost_model: CostModel,
        risk_measure: RiskMeasure,
        feature_map: FeatureMap | None,
        premium: float | torch.Tensor,
        liquidate_terminal: bool,
        aux_keys: tuple[str, ...],
        checkpoint_steps: bool,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.risk_measure = risk_measure
        self.payoff = payoff
        self.cost_model = cost_model
        self.feature_map = feature_map
        self.premium = premium
        self.liquidate_terminal = liquidate_terminal
        self.aux_keys = aux_keys
        self.checkpoint_steps = checkpoint_steps

    def forward(self, spot: torch.Tensor, *aux: torch.Tensor) -> torch.Tensor:
        """Runs one full episode and reduces to the scalar objective.

        Args:
            spot: Price grid of shape ``(n_steps + 1, n_paths)``.
            aux: Auxiliary channels in ``aux_keys`` order.

        Returns:
            Scalar risk objective.
        """
        state = MarketState(spot=spot, aux=dict(zip(self.aux_keys, aux, strict=True)))
        pnl = hedge_pnl(
            state,
            self.policy,
            self.payoff,
            self.cost_model,
            premium=self.premium,
            liquidate_terminal=self.liquidate_terminal,
            checkpoint_steps=self.checkpoint_steps,
            feature_map=self.feature_map,
        )
        return self.risk_measure(-pnl, weights=_importance_weights(state))


@dataclass
class TrainResult:
    """Outcome of a training run.

    Attributes:
        losses: Risk objective recorded at every iteration.
    """

    losses: list[float] = field(default_factory=list)


class _GeneratedEpisodeLoss(torch.nn.Module):
    """Episode loss that generates its own batch inside the capture.

    The simulator kernel reads the shifted Philox subsequence from the
    offset tensor at launch time, so capturing this module records
    generation, the episode forward, and the backward as one replayable
    unit, and updating the offset in place redraws a fresh batch on
    every replay. A host-argument kernel could not do this, because
    capture freezes launch arguments and every replay would train on
    the same paths.
    """

    def __init__(
        self,
        simulator: _OffsetSimulator,
        episode: _EpisodeLoss,
        n_paths: int,
        seed: int,
        aux_keys: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.simulator = simulator
        self.episode = episode
        self.n_paths = n_paths
        self.seed = seed
        self.aux_keys = aux_keys

    def forward(self, offset: torch.Tensor) -> torch.Tensor:
        """Generates the batch the offset names and scores the episode.

        Args:
            offset: One-element int64 CUDA tensor holding the shifted
                subsequence base.

        Returns:
            Scalar risk objective.
        """
        state = self.simulator.simulate_with_offset(self.n_paths, self.seed, offset)
        return self.episode(state.spot, *(state.aux[key] for key in self.aux_keys))


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

    Paths are generated on the fly per iteration. Data is unbounded,
    nothing is stored, and no overfitting to a fixed dataset is possible.
    The risk measure warm-starts its auxiliary state on an initial batch so
    early iterations optimise the intended objective, and is moved to the
    policy device so its parameters never force cross-device synchronisation
    inside the loop.

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

    Raises:
        ValueError: If the config combines mutually exclusive options or
            requests episode capture off the CUDA device.
    """
    device = next(policy.parameters()).device
    if config.graph_episode and device.type != "cuda":
        raise ValueError("graph_episode requires the policy on a CUDA device")
    risk_measure.to(device)
    base_noise = NoiseSpec(seed=config.seed) if config.seed is not None else None
    stepper: HedgePolicy = policy
    if config.compile_policy:
        stepper = torch.compile(policy)  # type: ignore[assignment]

    def batch_state(index: int) -> MarketState:
        noise = base_noise.child(index) if base_noise is not None else None
        return simulator.simulate(config.batch_paths, noise=noise).to(device)

    param_groups: list[dict[str, object]] = [{"params": list(policy.parameters()), "lr": config.lr}]
    risk_params = list(risk_measure.parameters())
    if risk_params:
        risk_lr = config.risk_lr if config.risk_lr is not None else 10.0 * config.lr
        param_groups.append({"params": risk_params, "lr": risk_lr})
    optimizer = torch.optim.Adam(param_groups)
    warmup_state = batch_state(0)
    with torch.no_grad():
        warmup_pnl = hedge_pnl(
            warmup_state,
            stepper,
            payoff,
            cost_model,
            premium=premium,
            liquidate_terminal=config.liquidate_terminal,
            feature_map=feature_map,
            amp=config.amp,
        )
        risk_measure.warm_start(-warmup_pnl, weights=_importance_weights(warmup_state))
    graphed_loss = None
    offset_input = torch.empty(0, dtype=torch.int64)
    aux_keys: tuple[str, ...] = ()
    if config.graph_episode:
        aux_keys = tuple(sorted(warmup_state.aux))
        episode = _EpisodeLoss(
            policy,
            payoff,
            cost_model,
            risk_measure,
            feature_map,
            premium,
            config.liquidate_terminal,
            aux_keys,
            config.checkpoint_steps,
        )
        if config.graph_generate:
            if not hasattr(simulator, "simulate_with_offset"):
                raise ValueError(
                    "graph_generate requires a simulator exposing simulate_with_offset"
                )
            assert config.seed is not None
            offset_input = torch.zeros((1,), dtype=torch.int64, device=device)
            generated = _GeneratedEpisodeLoss(
                cast(_OffsetSimulator, simulator),
                episode,
                config.batch_paths,
                config.seed,
                aux_keys,
            )
            graphed_loss = torch.cuda.make_graphed_callables(generated, (offset_input,))
        else:
            sample_args = (
                warmup_state.spot.clone(),
                *(warmup_state.aux[key].clone() for key in aux_keys),
            )
            graphed_loss = torch.cuda.make_graphed_callables(episode, sample_args)
        reserved = torch.cuda.memory_reserved(device)
        capacity = torch.cuda.get_device_properties(device).total_memory
        headroom = 0.7 if os.name == "nt" else 0.8
        if reserved > headroom * capacity:
            warnings.warn(
                f"graph capture reserves {reserved / 2**30:.1f}GiB of "
                f"{capacity / 2**30:.1f}GiB; batches beyond physical memory "
                "page through the host and collapse throughput",
                stacklevel=2,
            )

    def episode_loss(state: MarketState) -> torch.Tensor:
        pnl = hedge_pnl(
            state,
            stepper,
            payoff,
            cost_model,
            premium=premium,
            liquidate_terminal=config.liquidate_terminal,
            checkpoint_steps=config.checkpoint_steps,
            feature_map=feature_map,
            amp=config.amp,
        )
        return risk_measure(-pnl, weights=_importance_weights(state))

    def regenerated_loss(index: int) -> torch.Tensor:
        def from_noise(*_parameters: torch.Tensor) -> torch.Tensor:
            return episode_loss(batch_state(index))

        return cast(
            torch.Tensor,
            checkpoint(from_noise, *policy.parameters(), *risk_params, use_reentrant=False),
        )

    loss_history: list[torch.Tensor] = []
    for iteration in range(config.n_iterations):
        if graphed_loss is not None and config.graph_generate:
            assert base_noise is not None
            spec = base_noise.child(iteration + 1)
            offset_input.fill_(spec.stream << 32)
            loss = cast(torch.Tensor, graphed_loss(offset_input))
            optimizer.zero_grad(set_to_none=False)
        elif graphed_loss is not None:
            state = batch_state(iteration + 1)
            channels = (state.spot, *(state.aux[key] for key in aux_keys))
            loss = cast(torch.Tensor, graphed_loss(*channels))
            optimizer.zero_grad(set_to_none=False)
        elif config.regenerate_paths:
            loss = regenerated_loss(iteration + 1)
            optimizer.zero_grad(set_to_none=True)
        else:
            loss = episode_loss(batch_state(iteration + 1))
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().clone())
    recorded = torch.stack(loss_history).cpu() if loss_history else torch.empty(0)
    return TrainResult(losses=recorded.tolist())
