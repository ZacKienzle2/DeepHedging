"""Hedging policy interface."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class HedgePolicy(nn.Module, ABC):
    """Maps per-step market state to a hedge position.

    Policies are pure ``state -> action`` modules, ignorant of the simulator
    and the risk measure, so that either can be swapped independently.
    Recurrent policies thread their hidden state through the optional
    ``state`` argument; feedforward policies ignore it.
    """

    @abstractmethod
    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Computes the hedge position for one rebalancing date.

        Args:
            features: Per-path state of shape ``(n_paths, n_features)``;
                features are ``(log moneyness, time to maturity, previous
                position)``.
            state: Recurrent hidden state from the previous step, or ``None``
                at the first step.

        Returns:
            Tuple of the position per path, shape ``(n_paths,)``, and the
            hidden state to carry to the next step (``None`` for stateless
            policies).
        """
