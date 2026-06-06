"""Conditional value-at-risk via the Rockafellar-Uryasev formulation."""

import torch
from torch import nn

from deephedging.risk.base import RiskMeasure


class CVaR(RiskMeasure):
    """Rockafellar-Uryasev CVaR objective with a learned VaR threshold.

    Minimising ``w + E[(L - w)_+] / (1 - alpha)`` jointly over the threshold
    ``w`` and the policy parameters equals minimising ``CVaR_alpha(L)``; the
    identity is pointwise in the policy and therefore survives nonconvex
    networks. The threshold is a registered parameter updated by SGD rather
    than an empirical batch quantile: per-batch inner minimisation is
    optimistically biased (Jensen), and a parameter stays synchronised under
    DDP. Only roughly ``(1 - alpha) * batch`` paths carry gradient, so batch
    sizes should scale like ``1 / (1 - alpha)``.

    Attributes:
        alpha: Confidence level in (0, 1), e.g. 0.95.
        threshold: Learned VaR estimate, a scalar parameter.
    """

    def __init__(self, alpha: float = 0.95) -> None:
        """Initialises the risk measure.

        Args:
            alpha: Confidence level in (0, 1).

        Raises:
            ValueError: If ``alpha`` is outside (0, 1).
        """
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.threshold = nn.Parameter(torch.zeros(()))

    def warm_start(self, loss: torch.Tensor) -> None:
        """Initialises the threshold at the empirical alpha-quantile.

        Without this, a threshold starting at zero lags the true VaR for
        hundreds of iterations and the indicator fires on far more than the
        target tail fraction, so early training optimises something between
        the mean loss and the intended CVaR.

        Args:
            loss: Loss per path of shape ``(n_paths,)``.
        """
        with torch.no_grad():
            self.threshold.copy_(torch.quantile(loss, self.alpha))

    def forward(self, loss: torch.Tensor) -> torch.Tensor:
        """Evaluates the Rockafellar-Uryasev objective.

        Args:
            loss: Loss per path of shape ``(n_paths,)``.

        Returns:
            Scalar objective; an upper bound on ``CVaR_alpha`` that is tight
            when the threshold equals the alpha-quantile of the loss.

        Raises:
            ValueError: If ``loss`` is not one-dimensional.
        """
        if loss.dim() != 1:
            raise ValueError(f"loss must be 1-dimensional, got shape {tuple(loss.shape)}")
        excess = torch.relu(loss - self.threshold)
        return self.threshold + excess.mean() / (1.0 - self.alpha)
