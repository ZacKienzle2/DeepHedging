"""Spectral (distortion) risk measures."""

import math
from collections.abc import Callable

import torch

from deephedging.risk.base import RiskMeasure


class SpectralRisk(RiskMeasure):
    """Distortion risk measure ``integral_0^1 F_inv(u) dg(u)``.

    A spectral risk measure reweights the loss quantiles by a non-negative,
    non-decreasing spectrum and averages them. Writing it as a Choquet
    integral against a distortion ``g`` (a non-decreasing map with
    ``g(0) = 0`` and ``g(1) = 1``) makes the empirical estimator exact for
    any spectrum: the weight on each order statistic is the distortion
    increment across its probability mass, so no quadrature error enters and
    the indicator distortion reproduces CVaR to machine precision.

    Unlike CVaR, which discards every loss below its threshold, a smooth
    spectrum weights the whole distribution, so the gradient does not
    collapse onto the ``(1 - alpha)`` tail fraction and the objective stays
    well conditioned at small batch sizes. The measure is coherent whenever
    the distortion is concave, which the provided families satisfy.

    The distortion is evaluated on cumulative probabilities, which are
    constants under the policy, so the gradient flows only through the sorted
    loss values exactly as it does for an empirical quantile.

    Attributes:
        name: Identifier for the spectrum, recorded in provenance.
    """

    distortion: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self, distortion: Callable[[torch.Tensor], torch.Tensor], name: str = "custom"
    ) -> None:
        """Initialises the risk measure from a distortion function.

        Args:
            distortion: Non-decreasing map from cumulative probability in
                ``[0, 1]`` to ``[0, 1]`` with ``g(0) = 0`` and ``g(1) = 1``,
                applied elementwise to a tensor of probabilities.
            name: Identifier for the spectrum.
        """
        super().__init__()
        self.distortion = distortion
        self.name = name

    @classmethod
    def exponential(cls, risk_aversion: float) -> "SpectralRisk":
        """Builds the exponential spectral risk measure.

        The spectrum ``phi(u) = k exp(-k(1 - u)) / (1 - exp(-k))`` weights
        worse quantiles geometrically, recovering the expectation as the
        aversion vanishes and concentrating on the worst path as it grows.

        Args:
            risk_aversion: Positive aversion coefficient ``k``.

        Returns:
            A concave-distortion spectral risk measure.

        Raises:
            ValueError: If ``risk_aversion`` is not positive.
        """
        if risk_aversion <= 0.0:
            raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
        k = risk_aversion
        normaliser = math.expm1(k)

        def distortion(u: torch.Tensor) -> torch.Tensor:
            return torch.expm1(k * u) / normaliser

        return cls(distortion, name=f"exponential(k={risk_aversion})")

    @classmethod
    def conditional_value_at_risk(cls, alpha: float) -> "SpectralRisk":
        """Builds the spectral form of CVaR at level ``alpha``.

        The distortion ``g(u) = max(0, (u - alpha) / (1 - alpha))`` places
        uniform weight on the worst ``1 - alpha`` mass, so the measure equals
        the empirical expected shortfall. It exists to validate the spectral
        estimator against the closed-form CVaR and as a fixed-spectrum
        alternative to the learned-threshold :class:`~deephedging.risk.CVaR`.

        Args:
            alpha: Confidence level in ``(0, 1)``.

        Returns:
            A spectral risk measure equal to expected shortfall at ``alpha``.

        Raises:
            ValueError: If ``alpha`` is outside ``(0, 1)``.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        def distortion(u: torch.Tensor) -> torch.Tensor:
            return torch.clamp((u - alpha) / (1.0 - alpha), min=0.0, max=1.0)

        return cls(distortion, name=f"cvar(alpha={alpha})")

    def forward(self, loss: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluates the spectral risk of a loss sample.

        Args:
            loss: Loss per path of shape ``(n_paths,)``.
            weights: Optional likelihood ratios of shape ``(n_paths,)``; the
                quantile levels are taken in the weighted empirical
                distribution, keeping the measure unbiased under a tilted
                sampler.

        Returns:
            Scalar spectral risk, the distortion-weighted mean of the order
            statistics.

        Raises:
            ValueError: If ``loss`` is not one-dimensional.
        """
        if loss.dim() != 1:
            raise ValueError(f"loss must be 1-dimensional, got shape {tuple(loss.shape)}")
        order = torch.argsort(loss)
        ordered_loss = loss[order]
        if weights is None:
            masses = loss.new_full(loss.shape, 1.0 / loss.shape[0])
        else:
            ordered_weights = weights[order]
            masses = ordered_weights / ordered_weights.sum()
        upper = torch.cumsum(masses, dim=0)
        lower = upper - masses
        spectral_weights = self.distortion(upper) - self.distortion(lower)
        return (spectral_weights * ordered_loss).sum()
