"""Asian (average-price) option payoffs."""

from dataclasses import dataclass

import torch


def _average(paths: torch.Tensor, geometric: bool) -> torch.Tensor:
    if geometric:
        return torch.exp(torch.log(paths).mean(dim=0))
    return paths.mean(dim=0)


@dataclass(frozen=True)
class AsianCall:
    """Average-price call paying ``max(A - K, 0)`` on the path average ``A``.

    The average is taken over the whole monitoring grid, inception included,
    so the contract depends on the realised trajectory rather than the
    terminal price alone. Averaging damps the variance of the underlying, so
    an Asian option is worth less than its European counterpart and is the
    standard hedge against terminal-price manipulation. The geometric mean
    never exceeds the arithmetic mean, so the geometric variant is the
    cheaper of the two and the one with a closed form under GBM.

    The geometric average takes the log of every price and therefore assumes
    a strictly positive path, which the diffusive simulators guarantee.

    Attributes:
        strike: Strike price K.
        geometric: Whether to average geometrically rather than arithmetically.
    """

    strike: float
    geometric: bool = False

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the average-price call payoff from the full path.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        return torch.clamp(_average(paths, self.geometric) - self.strike, min=0.0)


@dataclass(frozen=True)
class AsianPut:
    """Average-price put paying ``max(K - A, 0)`` on the path average ``A``.

    The averaging conventions and the positivity assumption of the geometric
    variant match :class:`AsianCall`.

    Attributes:
        strike: Strike price K.
        geometric: Whether to average geometrically rather than arithmetically.
    """

    strike: float
    geometric: bool = False

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """Computes the average-price put payoff from the full path.

        Args:
            paths: Price paths of shape ``(n_steps + 1, n_paths)``.

        Returns:
            Payoff per path of shape ``(n_paths,)``.
        """
        return torch.clamp(self.strike - _average(paths, self.geometric), min=0.0)
