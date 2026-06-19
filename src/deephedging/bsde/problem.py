"""BSDE problem specification.

The scope is semilinear parabolic PDEs whose generator ``f(t, x, y, z)``
is uniformly Lipschitz in ``(y, z)``, the Pardoux-Peng class where the
terminal-matching loss is a genuine a posteriori error certificate.
Fully nonlinear problems (gamma constraints, uncertain volatility) and
singular control (proportional transaction costs) need 2BSDE or reflected
BSDE machinery and are deliberately out of scope.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Generator(Protocol):
    """Driver ``f(t, x, y, z)`` of the backward equation."""

    def __call__(
        self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the generator.

        Args:
            t: Scalar time tensor.
            x: Forward state of shape ``(n_paths, dim)``.
            y: Value process of shape ``(n_paths,)``.
            z: Volatility process of shape ``(n_paths, dim)``.

        Returns:
            Generator value per path of shape ``(n_paths,)``.
        """
        ...


@dataclass(frozen=True)
class ZeroGenerator:
    """Trivial generator ``f = 0``: ``Y_0`` is the plain expectation of ``g``."""

    def __call__(
        self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Returns zeros per path.

        Args:
            t: Scalar time tensor.
            x: Forward state of shape ``(n_paths, dim)``.
            y: Value process of shape ``(n_paths,)``.
            z: Volatility process of shape ``(n_paths, dim)``.

        Returns:
            Zeros of shape ``(n_paths,)``.
        """
        return torch.zeros_like(y)


@dataclass(frozen=True)
class DiscountGenerator:
    """Linear discounting generator ``f = -r y``.

    With the forward drift set to the same rate (risk-neutral dynamics),
    the solved ``Y_0`` is the discounted expectation, the no-friction
    arbitrage-free price.

    Attributes:
        rate: Continuously compounded interest rate.
    """

    rate: float

    def __call__(
        self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates ``-r y`` per path.

        Args:
            t: Scalar time tensor.
            x: Forward state of shape ``(n_paths, dim)``.
            y: Value process of shape ``(n_paths,)``.
            z: Volatility process of shape ``(n_paths, dim)``.

        Returns:
            ``-rate * y`` of shape ``(n_paths,)``.
        """
        return -self.rate * y


@dataclass(frozen=True)
class BSDEProblem:
    """A decoupled FBSDE with lognormal forward dynamics.

    The forward state is a ``dim``-dimensional GBM with independent drivers,
    evolved exactly in log space: ``dX^i = mu X^i dt + sigma X^i dW^i``.
    The backward equation is ``dY = -f(t, X, Y, Z) dt + Z . dW`` with
    terminal condition ``Y_T = g(X_T)``.

    Attributes:
        dim: Dimension of the forward state.
        x0: Initial value of every coordinate of the forward state.
        sigma: Volatility shared by all coordinates.
        maturity: Horizon in years.
        n_steps: Number of Euler steps for the backward process.
        terminal: Terminal condition ``g``; maps ``(n_paths, dim)`` to
            ``(n_paths,)``.
        generator: Driver of the backward equation; must be uniformly
            Lipschitz in ``(y, z)``.
        mu: Drift shared by all coordinates.
    """

    dim: int
    x0: float
    sigma: float
    maturity: float
    n_steps: int
    terminal: Callable[[torch.Tensor], torch.Tensor]
    generator: Generator = field(default_factory=ZeroGenerator)
    mu: float = 0.0

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"dim must be at least 1, got {self.dim}")
        if self.x0 <= 0.0:
            raise ValueError(f"x0 must be positive, got {self.x0}")
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")
