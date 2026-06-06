"""Entropic risk measure."""

import math

import torch

from deephedging.risk.base import RiskMeasure


class Entropic(RiskMeasure):
    """Entropic risk ``log(E[exp(a * L)]) / a`` for risk aversion ``a``.

    Computed via ``logsumexp`` for numerical stability; a naive
    ``exp().mean().log()`` overflows for large losses or risk aversion.

    The gradient weights samples by ``softmax(a * loss)``, so for large
    risk aversion the effective gradient sample size collapses toward the
    single worst path; scale the batch size with ``risk_aversion`` or the
    updates become extremely high variance.

    Attributes:
        risk_aversion: Positive risk-aversion coefficient.
    """

    def __init__(self, risk_aversion: float = 1.0) -> None:
        """Initialises the risk measure.

        Args:
            risk_aversion: Positive risk-aversion coefficient.

        Raises:
            ValueError: If ``risk_aversion`` is not positive.
        """
        super().__init__()
        if risk_aversion <= 0.0:
            raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
        self.risk_aversion = risk_aversion

    def forward(self, loss: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluates the entropic risk of a loss sample.

        Args:
            loss: Loss per path of shape ``(n_paths,)``.
            weights: Optional likelihood ratios of shape ``(n_paths,)``;
                the exponential moment becomes their weighted mean inside
                the stable log-sum-exp.

        Returns:
            Scalar entropic risk.

        Raises:
            ValueError: If ``loss`` is not one-dimensional.
        """
        if loss.dim() != 1:
            raise ValueError(f"loss must be 1-dimensional, got shape {tuple(loss.shape)}")
        a = self.risk_aversion
        n = loss.shape[0]
        exponent = a * loss
        if weights is not None:
            exponent = exponent + torch.log(torch.clamp(weights, min=1e-300))
        return (torch.logsumexp(exponent, dim=0) - math.log(n)) / a
