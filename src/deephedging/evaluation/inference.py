"""Bootstrap confidence intervals and paired significance for PnL metrics."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

MetricFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class BootstrapInterval:
    """A point estimate with a percentile bootstrap confidence interval.

    Attributes:
        estimate: The metric evaluated on the full sample.
        low: Lower confidence bound.
        high: Upper confidence bound.
        confidence: Two-sided coverage of the interval.
    """

    estimate: float
    low: float
    high: float
    confidence: float


@dataclass(frozen=True)
class PairedComparison:
    """Paired bootstrap comparison of a metric between two strategies.

    Attributes:
        difference: ``metric(first) - metric(second)`` on the full sample.
        low: Lower confidence bound of the difference.
        high: Upper confidence bound of the difference.
        confidence: Two-sided coverage of the interval.
        probability_first_lower: Bootstrap share of resamples where the first
            metric is the smaller; for a loss-like metric this is the evidence
            that the first strategy dominates.
    """

    difference: float
    low: float
    high: float
    confidence: float
    probability_first_lower: float


def bootstrap_metric(
    pnl: torch.Tensor,
    metric: MetricFunction,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile bootstrap interval for a scalar metric of a PnL sample.

    A lone point estimate of a tail metric hides its sampling error, which is
    largest exactly where the tail is thinnest; resampling the paths with
    replacement turns that hidden error into a reported interval. The
    percentile form is used because the skewed PnL distribution violates the
    normality a standard-error interval would assume.

    Args:
        pnl: PnL per path of shape ``(n_paths,)``.
        metric: Map from a PnL sample to a scalar tensor.
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided coverage in ``(0, 1)``.
        seed: Seed for the resampling generator, so the interval replays.

    Returns:
        The point estimate and its confidence bounds.

    Raises:
        ValueError: If ``pnl`` is not one-dimensional, ``confidence`` is
            outside ``(0, 1)``, or ``n_resamples`` is not positive.
    """
    _validate_sample(pnl, confidence, n_resamples)
    n_paths = pnl.shape[0]
    generator = torch.Generator(device=pnl.device).manual_seed(seed)
    estimates = pnl.new_empty(n_resamples)
    for index in range(n_resamples):
        draw = torch.randint(n_paths, (n_paths,), generator=generator, device=pnl.device)
        estimates[index] = metric(pnl[draw])
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(metric(pnl)),
        low=float(torch.quantile(estimates, tail)),
        high=float(torch.quantile(estimates, 1.0 - tail)),
        confidence=confidence,
    )


def paired_bootstrap(
    first_pnl: torch.Tensor,
    second_pnl: torch.Tensor,
    metric: MetricFunction,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Paired bootstrap of ``metric(first) - metric(second)`` on common paths.

    The two samples must be aligned on the same paths, as common random
    numbers produce: resampling one shared index set for both cancels the
    path noise they share and isolates the strategy difference, which is the
    comparison with the tightest interval and the correct one when the same
    market drives both. For a loss-like metric, where lower is better,
    ``probability_first_lower`` is the bootstrap evidence that the first
    strategy beats the second.

    Args:
        first_pnl: PnL per path of the first strategy, shape ``(n_paths,)``.
        second_pnl: PnL per path of the second strategy, same paths and shape.
        metric: Map from a PnL sample to a scalar tensor.
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided coverage in ``(0, 1)``.
        seed: Seed for the resampling generator.

    Returns:
        The difference, its confidence bounds, and the dominance probability.

    Raises:
        ValueError: If the samples disagree in shape, are not one-dimensional,
            ``confidence`` is outside ``(0, 1)``, or ``n_resamples`` is not
            positive.
    """
    _validate_sample(first_pnl, confidence, n_resamples)
    if first_pnl.shape != second_pnl.shape:
        raise ValueError(
            f"paired samples must match in shape, got {tuple(first_pnl.shape)} "
            f"and {tuple(second_pnl.shape)}"
        )
    n_paths = first_pnl.shape[0]
    generator = torch.Generator(device=first_pnl.device).manual_seed(seed)
    differences = first_pnl.new_empty(n_resamples)
    for index in range(n_resamples):
        draw = torch.randint(n_paths, (n_paths,), generator=generator, device=first_pnl.device)
        differences[index] = metric(first_pnl[draw]) - metric(second_pnl[draw])
    tail = (1.0 - confidence) / 2.0
    return PairedComparison(
        difference=float(metric(first_pnl) - metric(second_pnl)),
        low=float(torch.quantile(differences, tail)),
        high=float(torch.quantile(differences, 1.0 - tail)),
        confidence=confidence,
        probability_first_lower=float((differences < 0.0).double().mean()),
    )


def _validate_sample(pnl: torch.Tensor, confidence: float, n_resamples: int) -> None:
    if pnl.dim() != 1:
        raise ValueError(f"pnl must be 1-dimensional, got shape {tuple(pnl.shape)}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
