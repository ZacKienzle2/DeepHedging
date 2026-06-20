"""Permutation-equivariant deep-set hedging policy."""

import torch
from torch import nn

from deephedging.policies.base import HedgePolicy

_TOKEN_FEATURES = 3


class DeepSetPolicy(HedgePolicy):
    """Permutation-equivariant multi-asset policy in the Deep Sets family.

    A feedforward policy over a flat multi-asset feature vector relearns the
    hedge for every permutation of the asset ordering and grows its first
    layer with the asset count. This policy shares one encoder across assets
    and conditions each asset's position on a pooled, order-invariant summary
    of the book, so the map is equivariant under asset relabelling and the
    parameter count is independent of how many assets trade.

    It pairs with :class:`~deephedging.features.MultiAssetFeatures`, whose
    vector lays out the per-asset log moneyness, the shared time to maturity,
    and the per-asset positions. One token per asset is reconstructed from
    that layout as its own log moneyness and position alongside the shared
    time, the encoder maps every token to a latent, mean pooling forms the
    order-invariant summary, and the head reads each asset's position from its
    own latent concatenated with the summary.

    Attributes:
        n_assets: Number of assets the policy hedges.
        encoder: Shared per-asset encoder.
        head: Readout from an asset latent and the pooled summary.
    """

    def __init__(
        self,
        n_assets: int,
        hidden_sizes: tuple[int, ...] = (32, 32),
        latent_size: int = 32,
    ) -> None:
        """Initialises the policy network.

        Args:
            n_assets: Number of assets on the book.
            hidden_sizes: Widths of the shared encoder's hidden layers.
            latent_size: Width of the per-asset latent and the pooled summary.

        Raises:
            ValueError: If ``n_assets`` is not positive.
        """
        super().__init__()
        if n_assets < 1:
            raise ValueError(f"n_assets must be at least 1, got {n_assets}")
        self.n_assets = n_assets
        encoder_layers: list[nn.Module] = []
        width = _TOKEN_FEATURES
        for size in hidden_sizes:
            encoder_layers.append(nn.Linear(width, size))
            encoder_layers.append(nn.SiLU())
            width = size
        encoder_layers.append(nn.Linear(width, latent_size))
        self.encoder = nn.Sequential(*encoder_layers)
        self.head = nn.Sequential(
            nn.Linear(2 * latent_size, latent_size),
            nn.SiLU(),
            nn.Linear(latent_size, 1),
        )

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Computes the per-asset hedge positions for one rebalancing date.

        Args:
            features: Multi-asset features of shape ``(n_paths, 2 * n_assets
                + 1)`` laid out as :class:`~deephedging.features.MultiAssetFeatures`
                produces them.
            state: Ignored; present for interface compatibility.

        Returns:
            Tuple of the positions per path, shape ``(n_paths, n_assets)``,
            and ``None``.

        Raises:
            ValueError: If the feature width disagrees with ``n_assets``.
        """
        expected = 2 * self.n_assets + 1
        if features.shape[-1] != expected:
            raise ValueError(
                f"features must have width {expected} for {self.n_assets} assets, "
                f"got {features.shape[-1]}"
            )
        log_moneyness = features[:, : self.n_assets]
        tau = features[:, self.n_assets : self.n_assets + 1]
        position = features[:, self.n_assets + 1 :]
        tokens = torch.stack((log_moneyness, position, tau.expand(-1, self.n_assets)), dim=-1)
        latent = self.encoder(tokens)
        pooled = latent.mean(dim=1, keepdim=True).expand(-1, self.n_assets, -1)
        combined = torch.cat((latent, pooled), dim=-1)
        return self.head(combined).squeeze(-1), None
