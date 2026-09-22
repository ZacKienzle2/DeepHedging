"""No-transaction-band hedging policy.

Under proportional costs the optimal policy holds inside a band around
the model hedge and trades to the nearest edge outside it. Building
that structure into the network turns a hard exploration problem into
learning two non-negative widths, an inductive bias that trains faster
and generalises better than a free-form position network whenever the
band assumption holds. The Black-Scholes delta anchors the band, so
the policy degenerates gracefully to the model hedge as both widths
shrink to zero.

The band is a function of the market state alone. Davis, Panas and
Zariphopoulou (1993) and Whalley and Wilmott (1997, eq. 3.10) find the
buy and sell boundaries as curves in (S, t), and the network of Imaki
et al. (2021, eq. 12) accordingly omits the held position from its
input. The held position enters only the clamp, so the widths of every
date can be computed in one call before the clamp recursion runs.
"""

import math
from typing import cast, override

import torch
from torch import nn

from deephedging.networks import mlp
from deephedging.policies.base import HedgePolicy

_HELD = 2


class NoTransactionBandPolicy(HedgePolicy):
    """Network-widened no-transaction band around the model delta.

    The network maps the market columns of the observation, every column
    but the held position, to two widths through a softplus, the band is
    the model delta minus the lower width to the model delta plus the
    upper width, and the position is the held inventory clamped to the
    band. Gradients reach the widths exactly when their
    edge binds, which is the no-transaction-band training trick of
    Imaki and co-authors. The anchor delta is computed from the
    observation, so the policy needs the market volatility, horizon,
    and strike ratio that define it.

    Attributes:
        sigma: Volatility anchoring the model delta.
        maturity: Episode horizon in years.
        strike_ratio: Strike over initial spot.
        net: Width network ending in two outputs.
    """

    def __init__(
        self,
        sigma: float,
        maturity: float,
        strike_ratio: float = 1.0,
        n_features: int = 3,
        hidden_sizes: tuple[int, ...] = (32, 32),
    ) -> None:
        """Initialises the policy network.

        Args:
            sigma: Volatility anchoring the model delta.
            maturity: Episode horizon in years.
            strike_ratio: Strike over initial spot.
            n_features: Number of input features per path; the first
                must be log moneyness against the initial spot, the
                second the normalised time to maturity, and the third
                the held position, which the band network does not see.
            hidden_sizes: Hidden layer widths of the band network.

        Raises:
            ValueError: If sigma, maturity, or strike_ratio is not
                positive.
        """
        super().__init__()
        if sigma <= 0.0:
            msg = f"sigma must be positive, got {sigma}"
            raise ValueError(msg)
        if maturity <= 0.0:
            msg = f"maturity must be positive, got {maturity}"
            raise ValueError(msg)
        if strike_ratio <= 0.0:
            msg = f"strike_ratio must be positive, got {strike_ratio}"
            raise ValueError(msg)
        self.sigma = sigma
        self.maturity = maturity
        self.strike_ratio = strike_ratio
        self.log_strike_ratio = math.log(strike_ratio)
        market = [column for column in range(n_features) if column != _HELD]
        self.register_buffer("market_columns", torch.tensor(market), persistent=False)
        self.net = mlp(len(market), hidden_sizes, 2)

    def bands(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the band edges from the market columns.

        Args:
            features: Observations of shape ``(..., n_features)``, laid out
                as for :meth:`forward`; the held position is ignored, so
                any leading batch shape, such as every date at once, works.

        Returns:
            The lower and upper edges, each of the leading shape.
        """
        log_moneyness = features[..., 0] - self.log_strike_ratio
        remaining = torch.clamp(features[..., 1] * self.maturity, min=1e-8)
        scale = self.sigma * torch.sqrt(remaining)
        delta = torch.special.ndtr((log_moneyness + 0.5 * self.sigma**2 * remaining) / scale)
        market = features.index_select(-1, cast("torch.Tensor", self.market_columns))
        widths = nn.functional.softplus(self.net(market))
        return delta - widths[..., 0], delta + widths[..., 1]

    @override
    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Clamps the held position into the learned band.

        Args:
            features: Per-path state of shape ``(n_paths, n_features)``
                whose first three columns are log moneyness, normalised
                time to maturity, and held position.
            state: Ignored; present for interface compatibility.

        Returns:
            Tuple of the position per path, shape ``(n_paths,)``, and
            ``None``.
        """
        lower, upper = self.bands(features)
        return torch.clamp(features[..., _HELD], lower, upper), None
