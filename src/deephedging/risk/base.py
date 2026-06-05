"""Risk measure interface."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class RiskMeasure(nn.Module, ABC):
    """Maps a sample of losses to a scalar risk value.

    Losses follow the convention that larger values are worse. Risk measures
    are ``nn.Module`` subclasses so that auxiliary variables (such as the
    CVaR threshold) are registered parameters: they are optimised jointly
    with the policy and synchronised by DDP, where per-rank empirical
    statistics would silently disagree.
    """

    @abstractmethod
    def forward(self, loss: torch.Tensor) -> torch.Tensor:
        """Evaluates the risk of a loss sample.

        Args:
            loss: Loss per path of shape ``(n_paths,)``.

        Returns:
            Scalar risk value.
        """
