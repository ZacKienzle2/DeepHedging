"""Several risk levels trained by one policy."""

from collections.abc import Sequence
from typing import override

import torch
from torch import nn

from deephedging.risk.base import RiskMeasure


class MultiLevelRisk(RiskMeasure):
    """Average of risk measures, each over its own interleaved share of paths.

    Murray et al. (2022, Section 2.3) train hedging policies for many risk
    aversions at once with one network by adding the risk level to the agent
    state. Path ``i`` belongs to level ``i mod K``, which
    :class:`~deephedging.features.LevelFeatures` shows the policy, and the
    objective is the mean over levels of each level's risk of its paths.
    A policy conditioned on the level can minimise every term separately,
    so one training run yields the policy of every level. Each term keeps
    its own measure, so an entropic level keeps its stable log-sum-exp and a
    CVaR level its own learned threshold.

    Attributes:
        measures: One risk measure per level.
    """

    def __init__(self, measures: Sequence[RiskMeasure]) -> None:
        """Initialises the combined measure.

        Args:
            measures: One risk measure per level, in level order.

        Raises:
            ValueError: If ``measures`` is empty.
        """
        super().__init__()
        if not measures:
            msg = "measures must name at least one level"
            raise ValueError(msg)
        self.measures = nn.ModuleList(measures)

    def _levels(
        self, loss: torch.Tensor, weights: torch.Tensor | None
    ) -> list[tuple[RiskMeasure, torch.Tensor, torch.Tensor | None]]:
        count = len(self.measures)
        if loss.dim() != 1 or loss.shape[0] % count != 0:
            msg = f"loss must be 1-dimensional with a length divisible by {count}"
            raise ValueError(msg)
        columns = loss.view(-1, count).T
        weight_columns = weights.view(-1, count).T if weights is not None else None
        return [
            (
                measure,
                columns[level],
                weight_columns[level] if weight_columns is not None else None,
            )
            for level, measure in enumerate(self.measures)
            if isinstance(measure, RiskMeasure)
        ]

    @override
    def forward(
        self, loss: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Evaluates the mean of the per-level risks.

        Args:
            loss: Loss per path of shape ``(n_paths,)``, with ``n_paths``
                divisible by the number of levels.
            weights: Optional likelihood ratios of shape ``(n_paths,)``.

        Returns:
            Scalar mean of the level risks.
        """
        risks = [
            measure(column, weight)
            for measure, column, weight in self._levels(loss, weights)
        ]
        return torch.stack(risks).mean()

    @override
    def warm_start(
        self, loss: torch.Tensor, weights: torch.Tensor | None = None
    ) -> None:
        """Warm-starts each level's measure on its own paths.

        Args:
            loss: Loss per path of shape ``(n_paths,)``.
            weights: Optional likelihood ratios of shape ``(n_paths,)``.
        """
        for measure, column, weight in self._levels(loss, weights):
            measure.warm_start(column, weight)
