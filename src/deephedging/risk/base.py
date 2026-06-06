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
    def forward(self, loss: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluates the risk of a loss sample.

        Args:
            loss: Loss per path of shape ``(n_paths,)``.
            weights: Optional likelihood ratios of shape ``(n_paths,)``
                from importance sampling; expectations are taken under
                the original measure as weighted means.

        Returns:
            Scalar risk value.
        """

    def warm_start(self, loss: torch.Tensor, weights: torch.Tensor | None = None) -> None:
        """Initialises auxiliary parameters from an initial loss sample.

        Default is a no-op; risk measures with auxiliary state (the CVaR
        threshold) override it so training does not spend its early
        iterations dragging the auxiliary variable from an arbitrary
        initial value to the right scale.

        Args:
            loss: Loss per path of shape ``(n_paths,)``.
            weights: Optional likelihood ratios matching ``loss``.
        """
