"""Feedforward hedging policy."""

import torch
from torch import nn

from deephedging.policies.base import HedgePolicy


class FeedForwardPolicy(HedgePolicy):
    """Shared-weight MLP applied independently at each rebalancing date.

    Attributes:
        net: The underlying MLP.
    """

    def __init__(
        self,
        n_features: int = 3,
        hidden_sizes: tuple[int, ...] = (32, 32),
        n_outputs: int = 1,
    ) -> None:
        """Initialises the policy network.

        Args:
            n_features: Number of input features per path.
            hidden_sizes: Widths of the hidden layers.
            n_outputs: Number of hedge positions per path; one output per
                asset for multi-asset books.
        """
        super().__init__()
        self.n_outputs = n_outputs
        layers: list[nn.Module] = []
        width = n_features
        for size in hidden_sizes:
            layers.append(nn.Linear(width, size))
            layers.append(nn.SiLU())
            width = size
        layers.append(nn.Linear(width, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Computes the hedge position for one rebalancing date.

        Args:
            features: Per-path state of shape ``(n_paths, n_features)``.
            state: Ignored; present for interface compatibility.

        Returns:
            Tuple of the position per path, shape ``(n_paths,)`` for a
            single output or ``(n_paths, n_outputs)`` otherwise, and
            ``None``.
        """
        output = self.net(features)
        if self.n_outputs == 1:
            return output.squeeze(-1), None
        return output, None
