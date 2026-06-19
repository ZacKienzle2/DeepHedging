"""Mean-variance and mean-semivariance risk measures."""

import torch

from deephedging.risk.base import RiskMeasure


class MeanVariance(RiskMeasure):
    """Markowitz objective ``E[L] + c * Var(L)`` over the loss sample.

    The classical mean-variance trade-off and the baseline against
    which the coherent objectives are judged. It penalises dispersion
    symmetrically, so it charges for upside as well as downside and is not a
    coherent risk measure. The downside variant restricts the penalty to
    deviations above the mean, which is the relevant asymmetry for a hedger
    who fears losses but not windfalls; it recovers Markowitz semivariance.

    Both forms are smooth in the loss, so the gradient is far lower variance
    than the tail objectives and the batch size need not scale with any
    confidence level. The variance is the plug-in (biased) estimator, which
    is the consistent choice for an objective evaluated on large batches.

    Attributes:
        risk_aversion: Non-negative weight on the variance term.
        downside: Whether to penalise only above-mean deviations.
    """

    def __init__(self, risk_aversion: float = 1.0, downside: bool = False) -> None:
        """Initialises the risk measure.

        Args:
            risk_aversion: Non-negative weight on the variance term; zero
                recovers the plain expectation.
            downside: If true, penalise only deviations worse than the mean.

        Raises:
            ValueError: If ``risk_aversion`` is negative.
        """
        super().__init__()
        if risk_aversion < 0.0:
            raise ValueError(f"risk_aversion must be non-negative, got {risk_aversion}")
        self.risk_aversion = risk_aversion
        self.downside = downside

    def forward(self, loss: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluates the mean-variance objective.

        Args:
            loss: Loss per path of shape ``(n_paths,)``.
            weights: Optional likelihood ratios of shape ``(n_paths,)``; the
                mean and variance become their weighted counterparts, keeping
                the objective unbiased under a tilted sampler.

        Returns:
            Scalar objective ``mean + risk_aversion * variance``.

        Raises:
            ValueError: If ``loss`` is not one-dimensional.
        """
        if loss.dim() != 1:
            raise ValueError(f"loss must be 1-dimensional, got shape {tuple(loss.shape)}")
        if weights is None:
            mean = loss.mean()
            deviation = loss - mean
            if self.downside:
                deviation = torch.relu(deviation)
            variance = (deviation * deviation).mean()
        else:
            total = weights.sum()
            mean = (weights * loss).sum() / total
            deviation = loss - mean
            if self.downside:
                deviation = torch.relu(deviation)
            variance = (weights * deviation * deviation).sum() / total
        return mean + self.risk_aversion * variance
