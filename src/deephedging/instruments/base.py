"""Payoff interface."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Payoff(Protocol):
    """Maps full price paths to a terminal payoff per path.

    Payoffs are composable layers deliberately decoupled from path
    generation: research iterates on instruments far faster than on
    simulators, and path-dependent payoffs need access to the whole path.
    """

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the payoff.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        ...
