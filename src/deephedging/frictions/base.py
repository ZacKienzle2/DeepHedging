"""Transaction cost interface."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class CostModel(Protocol):
    """Cost of executing a signed trade at the prevailing price."""

    def __call__(self, trade: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """Computes the cost of a trade.

        Args:
            trade: Signed position change; arbitrary broadcastable shape.
            price: Execution price, broadcastable against ``trade``.

        Returns:
            Non-negative cost, elementwise over the broadcast shape.
        """
        ...
