"""Observation construction for hedging policies.

The feature map owns what the policy sees at each rebalancing date,
decoupling observation design from the episode engine. Path-functional
features and model-specific channels slot in without touching the engine
or the policies. Implementations must only read information available at
the current date because the policy is adapted and future prices must
never leak into its input.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from deephedging.market.state import MarketState


@runtime_checkable
class FeatureMap(Protocol):
    """Builds the per-step policy input from the market state."""

    @property
    def n_features(self) -> int:
        """Width of the produced feature vector."""
        ...

    def __call__(
        self, state: MarketState, t: int, tau: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """Computes features for rebalancing date ``t``.

        Args:
            state: Simulated market state.
            t: Index of the current rebalancing date.
            tau: Scalar tensor, normalised time to maturity at ``t``.
            position: Position held entering ``t``, shape ``(n_paths,)``.

        Returns:
            Features of shape ``(n_paths, n_features)``.
        """
        ...


@dataclass(frozen=True)
class DefaultFeatures:
    """Markovian baseline. Log moneyness, time to maturity, position."""

    @property
    def n_features(self) -> int:
        """Width of the produced feature vector."""
        return 3

    def __call__(
        self, state: MarketState, t: int, tau: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """Computes the baseline feature triple.

        Args:
            state: Simulated market state.
            t: Index of the current rebalancing date.
            tau: Scalar tensor, normalised time to maturity at ``t``.
            position: Position held entering ``t``, shape ``(n_paths,)``.

        Returns:
            Features of shape ``(n_paths, 3)``.
        """
        log_moneyness = state.log_relative(t)
        n_paths = log_moneyness.shape[0]
        return torch.stack((log_moneyness, tau.expand(n_paths), position), dim=-1)


@dataclass(frozen=True)
class RunningMaxFeatures:
    """Baseline features plus the normalised running maximum.

    The running maximum is the sufficient path statistic for up-and-out
    contracts. The policy can learn to stop hedging once the barrier is
    breached. The cumulative maximum is computed once and cached on the
    state, so an episode pays ``O(T)`` rather than ``O(T^2)``.
    """

    @property
    def n_features(self) -> int:
        """Width of the produced feature vector."""
        return 4

    def __call__(
        self, state: MarketState, t: int, tau: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """Computes baseline features plus log running maximum.

        Args:
            state: Simulated market state.
            t: Index of the current rebalancing date.
            tau: Scalar tensor, normalised time to maturity at ``t``.
            position: Position held entering ``t``, shape ``(n_paths,)``.

        Returns:
            Features of shape ``(n_paths, 4)``.
        """
        log_moneyness = state.log_relative(t)
        n_paths = log_moneyness.shape[0]
        return torch.stack(
            (
                log_moneyness,
                tau.expand(n_paths),
                position,
                torch.log(state.running_max(t) / state.spot[0]),
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class MultiAssetFeatures:
    """Per-asset log moneyness and positions plus time to maturity.

    For a trailing-asset spot grid the observation concatenates one log
    moneyness and one held position per asset around the shared time
    feature, so the policy sees the joint cross-sectional state its
    vector position must respond to.

    Attributes:
        n_assets: Number of assets on the spot grid.
    """

    n_assets: int

    @property
    def n_features(self) -> int:
        """Width of the produced feature vector."""
        return 2 * self.n_assets + 1

    def __call__(
        self, state: MarketState, t: int, tau: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """Computes the cross-sectional feature block.

        Args:
            state: Simulated market state with a trailing asset axis.
            t: Index of the current rebalancing date.
            tau: Scalar tensor, normalised time to maturity at ``t``.
            position: Positions held entering ``t``, shape
                ``(n_paths, n_assets)``.

        Returns:
            Features of shape ``(n_paths, 2 * n_assets + 1)``.

        Raises:
            ValueError: If the position width disagrees with ``n_assets``,
                which indicates a policy output width mismatch.
        """
        log_moneyness = state.log_relative(t)
        if position.dim() != 2 or position.shape[-1] != self.n_assets:
            raise ValueError(
                f"position must have shape (n_paths, {self.n_assets}); the policy "
                f"output width must match n_assets, got {tuple(position.shape)}"
            )
        n_paths = log_moneyness.shape[0]
        return torch.cat((log_moneyness, tau.expand(n_paths, 1), position), dim=-1)


@dataclass(frozen=True)
class VarianceFeatures:
    """Baseline features plus the instantaneous variance channel.

    Requires a simulator that exposes a ``variance`` channel, such as
    :class:`~deephedging.market.HestonSimulator`. Variance is the state
    variable a stochastic volatility hedge must condition on; without it
    the policy cannot distinguish calm from stressed regimes at the same
    spot level.
    """

    @property
    def n_features(self) -> int:
        """Width of the produced feature vector."""
        return 4

    def __call__(
        self, state: MarketState, t: int, tau: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """Computes baseline features plus instantaneous variance.

        Args:
            state: Simulated market state.
            t: Index of the current rebalancing date.
            tau: Scalar tensor, normalised time to maturity at ``t``.
            position: Position held entering ``t``, shape ``(n_paths,)``.

        Returns:
            Features of shape ``(n_paths, 4)``.

        Raises:
            KeyError: If the state carries no ``variance`` channel.
        """
        log_moneyness = state.log_relative(t)
        n_paths = log_moneyness.shape[0]
        return torch.stack(
            (
                log_moneyness,
                tau.expand(n_paths),
                position,
                state.aux["variance"][t],
            ),
            dim=-1,
        )
