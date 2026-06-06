"""Risk metrics for evaluating hedged PnL distributions."""

import torch


def expected_shortfall(
    pnl: torch.Tensor, alpha: float = 0.95, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Empirical expected shortfall (CVaR) of the loss ``-pnl``.

    Uses the Rockafellar-Uryasev form evaluated at the empirical quantile, so
    the number matches the training objective at its optimum. The plug-in
    estimator is finite-sample biased; use large evaluation samples and keep
    it out of training losses. The estimate is clamped at the sample maximum:
    when ``(1 - alpha) * n_paths`` falls below one, quantile interpolation
    plus the ``1 / (1 - alpha)`` rescaling can otherwise overshoot the worst
    observed loss, which no true tail mean can exceed.

    Args:
        pnl: PnL per path of shape ``(n_paths,)``; positive is profit.
        alpha: Confidence level in (0, 1).
        weights: Optional likelihood ratios from importance sampling;
            evaluating a tilted sample without them silently estimates
            the wrong measure, so tilted states must pass their ratios.

    Returns:
        Scalar expected shortfall of the loss distribution.

    Raises:
        ValueError: If ``alpha`` is outside (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    loss = -pnl
    if weights is None:
        var = torch.quantile(loss, alpha)
        excess = torch.relu(loss - var)
        return torch.minimum(var + excess.mean() / (1.0 - alpha), loss.max())
    order = torch.argsort(loss)
    cumulative = torch.cumsum(weights[order], dim=0) / weights.sum()
    position = min(int(torch.searchsorted(cumulative, alpha)), loss.shape[0] - 1)
    var = loss[order][position]
    excess = weights * torch.relu(loss - var)
    return torch.minimum(var + excess.mean() / (1.0 - alpha), loss.max())


def pnl_summary(pnl: torch.Tensor, alphas: tuple[float, ...] = (0.95, 0.99)) -> dict[str, float]:
    """Summarises a hedged PnL distribution.

    Args:
        pnl: PnL per path of shape ``(n_paths,)``.
        alphas: Confidence levels for expected shortfall.

    Returns:
        Mapping with mean, standard deviation, and expected shortfall at each
        requested level (keyed ``es_95`` style).
    """
    summary = {
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
    }
    for alpha in alphas:
        key = f"es_{int(alpha * 100 + 0.5)}"
        summary[key] = float(expected_shortfall(pnl, alpha))
    return summary
