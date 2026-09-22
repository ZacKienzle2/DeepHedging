"""Bootstrap confidence intervals and paired significance for PnL metrics.

The resamples are drawn as one index matrix per chunk and the metric reduces
its last dimension, so every resample in a chunk is evaluated by one call.
This is the vectorised statistic contract of ``scipy.stats.bootstrap``,
which replaces a Python loop issuing one gather and one metric per resample.
Chunks bound the index matrix at ``_CHUNK_ELEMENTS`` entries.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

MetricFunction = Callable[[torch.Tensor], torch.Tensor]

_CHUNK_ELEMENTS = 1 << 22


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
        metric: Reduces the last dimension of a PnL batch to one value per
            leading index, as ``expected_shortfall`` and
            ``lambda sample: sample.mean(dim=-1)`` do.
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided coverage in ``(0, 1)``.
        seed: Seed for the resampling generator, so the interval replays.

    Returns:
        The point estimate and its confidence bounds.

    Raises:
        ValueError: If ``pnl`` is not one-dimensional, ``confidence`` is
            outside ``(0, 1)``, ``n_resamples`` is not positive, or the
            metric does not reduce the last dimension.
    """
    if (msg := _sample_error(pnl, confidence, n_resamples)) is not None:
        raise ValueError(msg)
    (estimates,) = _resampled(metric, (pnl,), n_resamples, seed)
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
    numbers produce. Resampling one shared index set for both cancels the
    path noise they share and isolates the strategy difference, which is the
    comparison with the tightest interval and the correct one when the same
    market drives both. For a loss-like metric, where lower is better,
    ``probability_first_lower`` is the bootstrap evidence that the first
    strategy beats the second.

    Args:
        first_pnl: PnL per path of the first strategy, shape ``(n_paths,)``.
        second_pnl: PnL per path of the second strategy, same paths and shape.
        metric: Reduces the last dimension of a PnL batch to one value per
            leading index, as for :func:`bootstrap_metric`.
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided coverage in ``(0, 1)``.
        seed: Seed for the resampling generator.

    Returns:
        The difference, its confidence bounds, and the dominance probability.

    Raises:
        ValueError: If the samples disagree in shape, are not one-dimensional,
            ``confidence`` is outside ``(0, 1)``, ``n_resamples`` is not
            positive, or the metric does not reduce the last dimension.
    """
    if (msg := _sample_error(first_pnl, confidence, n_resamples)) is not None:
        raise ValueError(msg)
    if first_pnl.shape != second_pnl.shape:
        msg = (
            f"paired samples must match in shape, got {tuple(first_pnl.shape)} "
            f"and {tuple(second_pnl.shape)}"
        )
        raise ValueError(msg)
    first, second = _resampled(metric, (first_pnl, second_pnl), n_resamples, seed)
    differences = first - second
    tail = (1.0 - confidence) / 2.0
    return PairedComparison(
        difference=float(metric(first_pnl) - metric(second_pnl)),
        low=float(torch.quantile(differences, tail)),
        high=float(torch.quantile(differences, 1.0 - tail)),
        confidence=confidence,
        probability_first_lower=float((differences < 0.0).double().mean()),
    )


def _resampled(
    metric: MetricFunction, samples: tuple[torch.Tensor, ...], n_resamples: int, seed: int
) -> list[torch.Tensor]:
    n_paths = samples[0].shape[0]
    device = samples[0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = max(1, _CHUNK_ELEMENTS // n_paths)
    chunks: list[list[torch.Tensor]] = [[] for _ in samples]
    for start in range(0, n_resamples, rows):
        count = min(rows, n_resamples - start)
        draw = torch.randint(n_paths, (count, n_paths), generator=generator, device=device)
        for chunk, sample in zip(chunks, samples, strict=True):
            values = metric(sample[draw])
            if values.shape != (count,):
                msg = (
                    f"metric must reduce the last dimension to shape ({count},), "
                    f"got {tuple(values.shape)}"
                )
                raise ValueError(msg)
            chunk.append(values)
    return [torch.cat(chunk) for chunk in chunks]


def _sample_error(pnl: torch.Tensor, confidence: float, n_resamples: int) -> str | None:
    if pnl.dim() != 1:
        return f"pnl must be 1-dimensional, got shape {tuple(pnl.shape)}"
    if not 0.0 < confidence < 1.0:
        return f"confidence must be in (0, 1), got {confidence}"
    if n_resamples <= 0:
        return f"n_resamples must be positive, got {n_resamples}"
    return None
