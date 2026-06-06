"""No-transaction-band hedging policy.

Under proportional costs the optimal policy holds inside a band around
the model hedge and trades to the nearest edge outside it. Building
that structure into the network turns a hard exploration problem into
learning two non-negative widths, an inductive bias that trains faster
and generalises better than a free-form position network whenever the
band assumption holds. The Black-Scholes delta anchors the band, so
the policy degenerates gracefully to the model hedge as both widths
shrink to zero.
"""

import math

import torch
from torch import nn

from deephedging.policies.base import HedgePolicy


class NoTransactionBandPolicy(HedgePolicy):
    """Network-widened no-transaction band around the model delta.

    The network maps the observation to two widths through a softplus,
    the band is the model delta minus the lower width to the model
    delta plus the upper width, and the position is the held inventory
    clamped to the band. Gradients reach the widths exactly when their
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
                the held position.
            hidden_sizes: Hidden layer widths of the band network.

        Raises:
            ValueError: If sigma, maturity, or strike_ratio is not
                positive.
        """
        super().__init__()
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {maturity}")
        if strike_ratio <= 0.0:
            raise ValueError(f"strike_ratio must be positive, got {strike_ratio}")
        self.sigma = sigma
        self.maturity = maturity
        self.strike_ratio = strike_ratio
        self.log_strike_ratio = math.log(strike_ratio)
        layers: list[nn.Module] = []
        width = n_features
        for size in hidden_sizes:
            layers.append(nn.Linear(width, size))
            layers.append(nn.SiLU())
            width = size
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

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
        log_moneyness = features[..., 0] - self.log_strike_ratio
        remaining = torch.clamp(features[..., 1] * self.maturity, min=1e-8)
        scale = self.sigma * torch.sqrt(remaining)
        delta = torch.special.ndtr((log_moneyness + 0.5 * self.sigma**2 * remaining) / scale)
        widths = nn.functional.softplus(self.net(features))
        lower = delta - widths[..., 0]
        upper = delta + widths[..., 1]
        held = features[..., 2]
        return torch.maximum(torch.minimum(held, upper), lower), None
