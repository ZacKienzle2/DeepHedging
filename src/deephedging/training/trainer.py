"""Training loop for hedging policies."""

import os
import warnings
from collections.abc import Callable, Sequence
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

_WARMUP_ITERATIONS = 3


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
        graph_episode: Whether to capture the whole training iteration,
            the episode forward, the backward, gradient clipping and the
            optimiser step, in one CUDA graph and replay it per iteration.
            This is the whole-network capture of the PyTorch CUDA graphs
            documentation: the optimiser is Adam with ``capturable`` set,
            so its step counters and learning rates live on the device and
            a replay advances them without a host round trip. The training
            loop is dispatch-bound at small network widths, with the host
            issuing tens of microsecond kernels while the device idles, and
            replay collapses the iteration into one launch. Capture runs a
            few warm-up iterations on a side stream first, as the
            documentation requires, and then restores the parameters and
            clears the optimiser state, so the captured run starts from the
            same point as an eager one. Auxiliary channels such as the
            Heston variance are copied into static buffers alongside the
            spot grid in sorted key order. Capture keeps a private memory
            pool that roughly doubles the resident activation footprint,
            and a batch that overflows physical device memory degrades
            silently into host paging at a throughput collapse of more than
            an order of magnitude, so the trainer warns when the
            post-capture reservation approaches the device capacity,
            tightening the threshold on Windows where the display driver
            pages under pressure. Combining capture with gradient
            checkpointing trades replayed recompute for that footprint and
            lifts the batch ceiling. Requires CUDA and a fixed batch size;
            mutually exclusive with compilation.
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
            rather than from a frozen host argument, and the captured
            iteration advances that offset in place after drawing, so each
            replay trains on the next stream with generation included and
            nothing issued from the host but the replay itself. Requires
            ``graph_episode``, a seed to address the streams, and a
            simulator exposing ``simulate_with_offset``.
        grad_clip_norm: Optional ceiling on the global gradient norm across
            the policy and risk parameters, applied between the backward and
            the optimiser step. The tail objectives put almost all of their
            gradient mass on the rare worst paths, so a single extreme batch
            can produce a step that undoes many good ones; clipping bounds
            that step without changing the well-behaved iterations. The
            norm and the rescale are device operations with no host
            synchronisation, so clipping is captured with the rest of the
            iteration. Default off preserves the unclipped behaviour.
        lr_schedule: Optional learning-rate schedule, ``"cosine"`` or
            ``"linear"``, decaying every parameter group proportionally over
            ``n_iterations`` steps. A constant rate large enough to converge
            quickly early tends to oscillate around the optimum late;
            annealing keeps the early speed and settles the tail. Under
            capture the rates are device tensors the scheduler fills in
            place, which is what lets a replayed optimiser step read the
            current rate.
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
    grad_clip_norm: float | None = None
    lr_schedule: str | None = None

    def __post_init__(self) -> None:
        """Rejects field values outside the documented domain."""
        if self.compile_policy and self.checkpoint_steps:
            msg = "compile_policy and checkpoint_steps are mutually exclusive"
            raise ValueError(msg)
        if self.graph_episode and self.compile_policy:
            msg = "graph_episode excludes compilation"
            raise ValueError(msg)
        if self.regenerate_paths and self.graph_episode:
            msg = "regenerate_paths and graph_episode are mutually exclusive"
            raise ValueError(msg)
        if self.regenerate_paths and self.seed is None:
            msg = "regenerate_paths requires a seed for deterministic replay"
            raise ValueError(msg)
        if self.graph_generate and not self.graph_episode:
            msg = "graph_generate requires graph_episode"
            raise ValueError(msg)
        if self.graph_generate and self.seed is None:
            msg = "graph_generate requires a seed to address the streams"
            raise ValueError(msg)
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            msg = f"grad_clip_norm must be positive, got {self.grad_clip_norm}"
            raise ValueError(msg)
        if self.lr_schedule is not None and self.lr_schedule not in (
            "cosine",
            "linear",
        ):
            msg = f"lr_schedule must be 'cosine' or 'linear', got {self.lr_schedule!r}"
            raise ValueError(msg)


class _OffsetSimulator(Protocol):
    """Simulator whose kernel reads the Philox subsequence from the device."""

    def simulate_with_offset(
        self, n_paths: int, seed: int, offset: torch.Tensor
    ) -> MarketState:
        """Simulates a batch addressed by the device-resident offset."""
        ...


@dataclass
class TrainResult:
    """Outcome of a training run.

    Attributes:
        losses: Risk objective recorded at every iteration.
    """

    losses: list[float] = field(default_factory=list)


def _importance_weights(state: MarketState) -> torch.Tensor | None:
    log_weight = state.aux.get(LOG_WEIGHT_CHANNEL)
    if log_weight is None:
        return None
    return torch.exp(log_weight.to(torch.float64))


def _optimizer(
    config: TrainConfig,
    policy_params: list[torch.nn.Parameter],
    risk_params: list[torch.nn.Parameter],
    device: torch.device,
) -> torch.optim.Adam:
    """Builds Adam over the policy and risk parameter groups.

    Fused on CUDA, and capturable with device-tensor learning rates when the
    iteration is captured, so a replayed step reads the scheduler's rate.

    Args:
        config: Training hyperparameters.
        policy_params: Parameters of the hedging policy.
        risk_params: Parameters of the risk measure, possibly empty.
        device: Device holding the parameters.

    Returns:
        The optimiser.
    """

    def rate(value: float) -> float | torch.Tensor:
        return torch.tensor(value, device=device) if config.graph_episode else value

    groups: list[dict[str, object]] = [{"params": policy_params, "lr": rate(config.lr)}]
    if risk_params:
        risk_lr = config.risk_lr if config.risk_lr is not None else 10.0 * config.lr
        groups.append({"params": risk_params, "lr": rate(risk_lr)})
    return torch.optim.Adam(
        groups, fused=device.type == "cuda", capturable=config.graph_episode
    )


def _schedule(
    config: TrainConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Builds the configured learning-rate schedule, if any.

    Args:
        config: Training hyperparameters.
        optimizer: The optimiser whose rates the schedule decays.

    Returns:
        The schedule, or None for a constant rate.
    """
    total = max(1, config.n_iterations)
    if config.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    if config.lr_schedule == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=1e-3, total_iters=total
        )
    return None


def _warn_if_capture_crowds(device: torch.device) -> None:
    """Warns when the capture pool leaves too little device memory free.

    Args:
        device: The CUDA device the graph was captured on.
    """
    reserved = torch.cuda.memory_reserved(device)
    capacity = torch.cuda.get_device_properties(device).total_memory
    headroom = 0.7 if os.name == "nt" else 0.8
    if reserved > headroom * capacity:
        warnings.warn(
            f"graph capture reserves {reserved / 2**30:.1f}GiB of "
            f"{capacity / 2**30:.1f}GiB; batches beyond physical memory "
            "page through the host and collapse throughput",
            stacklevel=3,
        )


def _capture(
    iteration: Callable[[], torch.Tensor],
    parameters: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    """Captures one training iteration as a CUDA graph.

    Follows the whole-network recipe of the PyTorch CUDA graphs notes:
    warm up on a side stream so lazily allocated optimiser state and
    autograd buffers exist before capture, then capture with the gradients
    unset so the backward allocates them from the graph's private pool.
    The warm-up steps move the parameters and fill the Adam moments, so
    both are restored afterwards, which keeps the first replay identical
    to the first eager iteration.

    Args:
        iteration: Runs the forward, the backward and the optimiser step,
            returning the loss.
        parameters: Every parameter the optimiser updates.
        optimizer: The capturable optimiser stepped inside ``iteration``.

    Returns:
        The captured graph and its static loss tensor.
    """
    initial = [parameter.detach().clone() for parameter in parameters]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(_WARMUP_ITERATIONS):
            optimizer.zero_grad(set_to_none=True)
            iteration()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        static_loss = iteration()
    with torch.no_grad():
        for parameter, value in zip(parameters, initial, strict=True):
            parameter.copy_(value)
        for moments in optimizer.state.values():
            for tensor in moments.values():
                tensor.zero_()
    return graph, static_loss


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
    inside the loop. On CUDA the optimiser is PyTorch's fused Adam, which
    updates every parameter in one kernel launch.

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
        msg = "graph_episode requires the policy on a CUDA device"
        raise ValueError(msg)
    if config.graph_generate and not hasattr(simulator, "simulate_with_offset"):
        msg = "graph_generate requires a simulator exposing simulate_with_offset"
        raise ValueError(msg)
    risk_measure.to(device)
    base_noise = NoiseSpec(seed=config.seed) if config.seed is not None else None
    stepper: HedgePolicy = policy
    if config.compile_policy:
        stepper = torch.compile(policy)  # type: ignore[assignment]

    def batch_state(index: int) -> MarketState:
        noise = base_noise.child(index) if base_noise is not None else None
        return simulator.simulate(config.batch_paths, noise=noise).to(device)

    policy_params = list(policy.parameters())
    risk_params = list(risk_measure.parameters())
    optimizer = _optimizer(config, policy_params, risk_params, device)
    scheduler = _schedule(config, optimizer)
    clip_params = policy_params + risk_params
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
        loss: torch.Tensor = risk_measure(-pnl, weights=_importance_weights(state))
        return loss

    def descend(loss: torch.Tensor) -> torch.Tensor:
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(clip_params, config.grad_clip_norm)
        optimizer.step()
        return loss

    def regenerated_loss(index: int) -> torch.Tensor:
        def from_noise(*_parameters: torch.Tensor) -> torch.Tensor:
            return episode_loss(batch_state(index))

        return checkpoint(from_noise, *policy_params, *risk_params, use_reentrant=False)

    graph: torch.cuda.CUDAGraph | None = None
    static_loss = torch.empty(0)
    aux_keys = tuple(sorted(warmup_state.aux))
    static_channels = (
        warmup_state.spot.clone(),
        *(warmup_state.aux[key].clone() for key in aux_keys),
    )
    offset = torch.zeros((1,), dtype=torch.int64, device=device)
    if config.graph_episode:
        if config.graph_generate:
            assert config.seed is not None
            generator = cast("_OffsetSimulator", simulator)
            seed = config.seed

            def captured() -> torch.Tensor:
                state = generator.simulate_with_offset(config.batch_paths, seed, offset)
                offset.add_(1 << 32)
                return descend(episode_loss(state))

        else:

            def captured() -> torch.Tensor:
                spot, *aux = static_channels
                state = MarketState(
                    spot=spot, aux=dict(zip(aux_keys, aux, strict=True))
                )
                return descend(episode_loss(state))

        graph, static_loss = _capture(captured, clip_params, optimizer)
        if base_noise is not None:
            offset.fill_(base_noise.child(1).stream << 32)
        _warn_if_capture_crowds(device)

    loss_history: list[torch.Tensor] = []
    for iteration in range(config.n_iterations):
        if graph is not None:
            if not config.graph_generate:
                state = batch_state(iteration + 1)
                for buffer, channel in zip(
                    static_channels,
                    (state.spot, *(state.aux[key] for key in aux_keys)),
                    strict=True,
                ):
                    buffer.copy_(channel)
            graph.replay()
            loss = static_loss
        else:
            optimizer.zero_grad(set_to_none=True)
            if config.regenerate_paths:
                loss = descend(regenerated_loss(iteration + 1))
            else:
                loss = descend(episode_loss(batch_state(iteration + 1)))
        if scheduler is not None:
            scheduler.step()
        loss_history.append(loss.detach().clone())
    recorded = torch.stack(loss_history).cpu() if loss_history else torch.empty(0)
    return TrainResult(losses=recorded.tolist())
