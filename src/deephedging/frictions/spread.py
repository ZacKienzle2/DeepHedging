"""Bid-ask spread transaction costs."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BidAskCost:
    """Directional cost from crossing a bid-ask spread.

    A single proportional rate cannot express a market that charges more to
    buy than to sell, the asymmetry of a one-sided book or a borrow-
    constrained short. Buys execute at the ask and pay ``ask_half_spread``,
    sells execute at the bid and pay ``bid_half_spread``, each as a fraction
    of the price. Equal half-spreads recover the symmetric proportional cost,
    so this strictly generalises it.

    Attributes:
        bid_half_spread: Half-spread paid when selling, as a fraction of price.
        ask_half_spread: Half-spread paid when buying, as a fraction of price.
    """

    bid_half_spread: float
    ask_half_spread: float

    def __post_init__(self) -> None:
        if self.bid_half_spread < 0.0 or self.ask_half_spread < 0.0:
            raise ValueError(
                "half-spreads must be non-negative, got "
                f"bid={self.bid_half_spread} ask={self.ask_half_spread}"
            )

    def __call__(self, trade: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """Computes the spread cost of a trade.

        Args:
            trade: Signed position change; positive is a buy.
            price: Execution price.

        Returns:
            ``price * (ask_half_spread * buy + bid_half_spread * sell)``, where
            ``buy`` and ``sell`` are the positive and negative parts of the
            trade.
        """
        buy = torch.relu(trade)
        sell = torch.relu(-trade)
        return price * (self.ask_half_spread * buy + self.bid_half_spread * sell)
