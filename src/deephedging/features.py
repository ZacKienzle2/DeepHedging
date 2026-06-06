"""Observation construction for hedging policies.

The feature map owns what the policy sees at each rebalancing date,
decoupling observation design from the episode engine: path-functional
features (running extrema, realised variance) and model-specific state
slot in without touching the engine or the policies.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FeatureMap(Protocol):
    """Builds the per-step policy input from the path history."""

    @property
    def n_features(self) -> int:
        """Width of the produced feature vector."""
        ...

    def __call__(
        self, paths: torch.Tensor, t: int, tau: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """Computes features for rebalancing date ``t``.

        Implementations must only read ``paths[: t + 1]``: the policy is
        adapted, and future prices must never leak into its input.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.
            t: Index of the current rebalancing date.
            tau: Scalar tensor, normalised time to maturity at ``t``.
            position: Position held entering ``t``, shape ``(n_paths,)``.

        Returns:
            Features of shape ``(n_paths, n_features)``.
        """
        ...


@dataclass(frozen=True)
class DefaultFeatures:
    """Markovian baseline: log moneyness, time to maturity, position."""

    @property
    def n_features(self) -> int:
        """Width of the produced feature vector."""
        return 3

    def __call__(
        self, paths: torch.Tensor, t: int, tau: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """Computes the baseline feature triple.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.
            t: Index of the current rebalancing date.
            tau: Scalar tensor, normalised time to maturity at ``t``.
            position: Position held entering ``t``, shape ``(n_paths,)``.

        Returns:
            Features of shape ``(n_paths, 3)``.
        """
        spot = paths[t]
        n_paths = spot.shape[0]
        return torch.stack((torch.log(spot / paths[0]), tau.expand(n_paths), position), dim=-1)


@dataclass(frozen=True)
class RunningMaxFeatures:
    """Baseline features plus the normalised running maximum.

    The running maximum is the sufficient path statistic for up-and-out
    contracts: the policy can learn to stop hedging once the barrier is
    breached. Recomputes the maximum from the prefix each step, costing
    ``O(T^2)`` over an episode; acceptable for the research engine, a
    running accumulator in the fused CUDA generator later.
    """

    @property
    def n_features(self) -> int:
        """Width of the produced feature vector."""
        return 4

    def __call__(
        self, paths: torch.Tensor, t: int, tau: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """Computes baseline features plus log running maximum.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.
            t: Index of the current rebalancing date.
            tau: Scalar tensor, normalised time to maturity at ``t``.
            position: Position held entering ``t``, shape ``(n_paths,)``.

        Returns:
            Features of shape ``(n_paths, 4)``.
        """
        spot = paths[t]
        n_paths = spot.shape[0]
        running_max = paths[: t + 1].max(dim=0).values
        return torch.stack(
            (
                torch.log(spot / paths[0]),
                tau.expand(n_paths),
                position,
                torch.log(running_max / paths[0]),
            ),
            dim=-1,
        )
