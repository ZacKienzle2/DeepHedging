"""Recurrent hedging policy."""

import torch
from torch import nn

from deephedging.policies.base import HedgePolicy


class RecurrentPolicy(HedgePolicy):
    """GRU-cell policy carrying hidden state across rebalancing dates.

    Suited to incomplete-information settings where the optimal hedge
    depends on the path history, not only the current state.

    Attributes:
        cell: The recurrent cell.
        head: Linear readout from hidden state to position.
        hidden_size: Width of the recurrent hidden state.
    """

    def __init__(self, n_features: int = 3, hidden_size: int = 32) -> None:
        """Initialises the policy network.

        Args:
            n_features: Number of input features per path.
            hidden_size: Width of the recurrent hidden state.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(n_features, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Computes the hedge position for one rebalancing date.

        Args:
            features: Per-path state of shape ``(n_paths, n_features)``.
            state: Hidden state of shape ``(n_paths, hidden_size)``, or
                ``None`` at the first step.

        Returns:
            Tuple of the position per path, shape ``(n_paths,)``, and the
            updated hidden state.
        """
        if state is None:
            state = features.new_zeros((features.shape[0], self.hidden_size))
        hidden = self.cell(features, state)
        return self.head(hidden).squeeze(-1), hidden
