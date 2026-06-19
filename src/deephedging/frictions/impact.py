"""Nonlinear market-impact transaction costs."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PowerLawImpactCost:
    """Market impact priced as ``coefficient * price * |trade| ** exponent``.

    Proportional costs charge a fixed fraction of notional and so ignore that
    a larger order moves the price against itself. A power law restores that
    feedback. The square-root law, exponent 1.5 on the traded quantity and
    the default, matches the empirically observed impact of meta-orders,
    while exponent 2.0 recovers the Almgren-Chriss temporary impact that is
    linear in the trading rate. The coefficient absorbs the asset liquidity
    scale.

    The cost is smooth in the trade and zero at the origin, so it adds a
    convex penalty on large rebalances without breaking the gradient the
    policy trains on. The exponent is held at or above one because a sublinear
    impact would price an infinite marginal cost at the origin and reward
    splitting every trade into vanishing pieces.

    Attributes:
        coefficient: Non-negative impact coefficient.
        exponent: Power on the traded quantity, at least one.
    """

    coefficient: float
    exponent: float = 1.5

    def __post_init__(self) -> None:
        if self.coefficient < 0.0:
            raise ValueError(f"coefficient must be non-negative, got {self.coefficient}")
        if self.exponent < 1.0:
            raise ValueError(f"exponent must be at least one, got {self.exponent}")

    def __call__(self, trade: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """Computes the market-impact cost of a trade.

        Args:
            trade: Signed position change.
            price: Execution price.

        Returns:
            ``coefficient * price * |trade| ** exponent`` elementwise.
        """
        return self.coefficient * price * trade.abs() ** self.exponent
