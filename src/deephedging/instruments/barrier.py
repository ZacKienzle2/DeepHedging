"""Barrier option payoffs."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UpAndOutCall:
    """Up-and-out call, a vanilla call payoff knocked out at a barrier.

    Pays ``max(S_T - K, 0)`` if the path never reaches the barrier and zero
    otherwise. The barrier is monitored discretely on the rebalancing grid,
    so the payoff is the discretely-monitored contract; continuously
    monitored barriers knock out more often and are worth less. Monitoring
    includes inception. A barrier at or below the initial spot makes the
    contract worthless from the start, which the constructor cannot reject
    because the payoff never sees the spot.

    Attributes:
        strike: Strike price K.
        barrier: Knock-out level B, strictly above the strike.
    """

    strike: float
    barrier: float

    def __post_init__(self) -> None:
        if self.barrier <= self.strike:
            raise ValueError(
                f"barrier must exceed strike, got barrier={self.barrier} strike={self.strike}"
            )

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the knocked payoff from the full path.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        alive = paths.max(dim=0).values < self.barrier
        vanilla = torch.clamp(paths[-1] - self.strike, min=0.0)
        return vanilla * alive


@dataclass(frozen=True)
class UpAndInCall:
    """Up-and-in call, a vanilla call that activates only past a barrier.

    Pays ``max(S_T - K, 0)`` only if the path reaches the barrier and zero
    otherwise, the knock-in complement of :class:`UpAndOutCall`. Holding both
    at one strike and barrier reconstructs the vanilla call, the in-out
    parity the test suite pins. The barrier is monitored discretely on the
    rebalancing grid, inception included.

    Attributes:
        strike: Strike price K.
        barrier: Knock-in level B, strictly above the strike.
    """

    strike: float
    barrier: float

    def __post_init__(self) -> None:
        if self.barrier <= self.strike:
            raise ValueError(
                f"barrier must exceed strike, got barrier={self.barrier} strike={self.strike}"
            )

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the knocked-in payoff from the full path.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        knocked_in = paths.max(dim=0).values >= self.barrier
        vanilla = torch.clamp(paths[-1] - self.strike, min=0.0)
        return vanilla * knocked_in


@dataclass(frozen=True)
class DownAndOutCall:
    """Down-and-out call, a vanilla call knocked out at a lower barrier.

    Pays ``max(S_T - K, 0)`` if the path never falls to the barrier and zero
    otherwise. The barrier sits below the spot, so a level at or above the
    initial spot makes the contract worthless from the start, which the
    constructor cannot reject because the payoff never sees the spot.
    Monitoring is discrete on the rebalancing grid, inception included.

    Attributes:
        strike: Strike price K.
        barrier: Knock-out level B, a positive level below the spot.
    """

    strike: float
    barrier: float

    def __post_init__(self) -> None:
        if self.barrier <= 0.0:
            raise ValueError(f"barrier must be positive, got {self.barrier}")

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the knocked payoff from the full path.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        alive = paths.min(dim=0).values > self.barrier
        vanilla = torch.clamp(paths[-1] - self.strike, min=0.0)
        return vanilla * alive


@dataclass(frozen=True)
class DownAndInCall:
    """Down-and-in call, a vanilla call that activates only below a barrier.

    Pays ``max(S_T - K, 0)`` only if the path falls to the barrier, the
    knock-in complement of :class:`DownAndOutCall`, and the two together at
    one strike and barrier reconstruct the vanilla call. Monitoring is
    discrete on the rebalancing grid, inception included.

    Attributes:
        strike: Strike price K.
        barrier: Knock-in level B, a positive level below the spot.
    """

    strike: float
    barrier: float

    def __post_init__(self) -> None:
        if self.barrier <= 0.0:
            raise ValueError(f"barrier must be positive, got {self.barrier}")

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the knocked-in payoff from the full path.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        knocked_in = paths.min(dim=0).values <= self.barrier
        vanilla = torch.clamp(paths[-1] - self.strike, min=0.0)
        return vanilla * knocked_in


@dataclass(frozen=True)
class DoubleKnockOutCall:
    """Call knocked out by either an upper or a lower barrier.

    Pays ``max(S_T - K, 0)`` only while the path stays strictly inside the
    corridor and zero once it touches either side, so the contract is worth
    no more than the single-barrier knock-outs and cheaper still. Monitoring
    is discrete on the rebalancing grid, inception included.

    Attributes:
        strike: Strike price K.
        lower_barrier: Lower knock-out level, positive and below the upper.
        upper_barrier: Upper knock-out level, above the lower.
    """

    strike: float
    lower_barrier: float
    upper_barrier: float

    def __post_init__(self) -> None:
        if self.lower_barrier <= 0.0:
            raise ValueError(f"lower_barrier must be positive, got {self.lower_barrier}")
        if self.upper_barrier <= self.lower_barrier:
            raise ValueError(
                f"upper_barrier must exceed lower_barrier, got upper={self.upper_barrier} "
                f"lower={self.lower_barrier}"
            )

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the corridor-knocked payoff from the full path.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        alive = (paths.min(dim=0).values > self.lower_barrier) & (
            paths.max(dim=0).values < self.upper_barrier
        )
        vanilla = torch.clamp(paths[-1] - self.strike, min=0.0)
        return vanilla * alive
