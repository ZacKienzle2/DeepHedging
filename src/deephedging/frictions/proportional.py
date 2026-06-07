"""Proportional transaction costs."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProportionalCost:
    """Cost proportional to traded notional: ``rate * price * |trade|``.

    Attributes:
        rate: Proportional cost rate, e.g. 1e-3 for 10 basis points.
    """

    rate: float

    def __post_init__(self) -> None:
        if self.rate < 0.0:
            raise ValueError(f"rate must be non-negative, got {self.rate}")

    def __call__(self, trade: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """Computes proportional cost of a trade.

        Args:
            trade: Signed position change.
            price: Execution price.

        Returns:
            ``rate * price * |trade|`` elementwise.
        """
        return self.rate * price * trade.abs()


@dataclass(frozen=True)
class PerAssetProportionalCost:
    """Proportional cost with one rate per asset on the trailing axis.

    A single rate misprices multi-instrument books whose assets trade
    at different liquidity, most visibly a variance swap whose price
    scale is hundreds of times below the spot. One rate per asset
    broadcasts against the trailing axis the engine contracts, so the
    per-step cost is the rate-weighted traded notional summed over
    assets. The rate tensor is materialised per call, which keeps the
    model stateless but means episode graph capture would bake one
    allocation per step into the graph; prefer the scalar model inside
    captured episodes until the rates are cached per device.

    Attributes:
        rates: Proportional cost rate per asset.
    """

    rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.rates:
            raise ValueError("rates must contain at least one asset")
        if any(rate < 0.0 for rate in self.rates):
            raise ValueError(f"rates must be non-negative, got {self.rates}")

    def __call__(self, trade: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """Computes the per-asset proportional cost of a trade.

        Args:
            trade: Signed position changes of shape ``(n_paths, n_assets)``.
            price: Execution prices of shape ``(n_paths, n_assets)``.

        Returns:
            ``rates * price * |trade|`` elementwise.

        Raises:
            ValueError: If the trailing axis disagrees with the number
                of rates.
        """
        if trade.shape[-1] != len(self.rates):
            raise ValueError(
                f"trade has {trade.shape[-1]} assets, cost model has {len(self.rates)} rates"
            )
        rates = price.new_tensor(self.rates)
        return rates * price * trade.abs()


@dataclass(frozen=True)
class NoCost:
    """Frictionless execution."""

    def __call__(self, trade: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """Returns zero cost with the broadcast shape of the inputs.

        Args:
            trade: Signed position change.
            price: Execution price.

        Returns:
            Zeros with the broadcast shape of ``trade`` and ``price``.
        """
        return torch.zeros_like(trade * price)
