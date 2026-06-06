"""Heston characteristic function in the trap-safe formulation.

The textbook Heston characteristic function takes the principal branch
of a complex logarithm whose argument winds around the origin as the
frequency or the maturity grows, producing discontinuities and wrong
prices, the failure named the little Heston trap by Albrecher, Mayer,
Schoutens, and Tistaert. The Gatheral formulation below selects the
negative root of the discriminant, whose log argument stays in the
right half plane on the pricing strip, so the function is continuous
everywhere the calibrator can wander. Everything is composed from exp,
log, sqrt, and arithmetic on complex128 tensors, so the parameters
carry autograd for calibration gradients.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HestonParams:
    """Heston model parameters under the risk-neutral measure with zero rate.

    Attributes:
        v0: Initial variance.
        kappa: Mean-reversion speed of the variance.
        theta: Long-run variance level.
        xi: Volatility of variance.
        rho: Correlation between price and variance drivers.
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 < 0.0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta < 0.0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if self.xi <= 0.0:
            raise ValueError(f"xi must be positive, got {self.xi}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")

    def as_tensors(self) -> tuple[torch.Tensor, ...]:
        """Returns the parameters as float64 scalar tensors.

        Returns:
            Tuple of ``(v0, kappa, theta, xi, rho)`` tensors.
        """
        return tuple(
            torch.tensor(value, dtype=torch.float64)
            for value in (self.v0, self.kappa, self.theta, self.xi, self.rho)
        )


def heston_cf(
    u: torch.Tensor,
    tau: float,
    v0: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    xi: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Characteristic function of the log return ``log(S_tau / S_0)``.

    Uses the Gatheral negative-root form; the martingale property
    ``cf(-i) == 1`` holds by construction and is pinned in tests.

    Args:
        u: Real frequencies of shape ``(n_terms,)``.
        tau: Time to maturity in years.
        v0: Initial variance as a scalar tensor.
        kappa: Mean-reversion speed as a scalar tensor.
        theta: Long-run variance as a scalar tensor.
        xi: Volatility of variance as a scalar tensor.
        rho: Driver correlation as a scalar tensor.

    Returns:
        Complex tensor of shape ``(n_terms,)``.
    """
    u_complex = u.to(torch.complex128)
    iu = 1j * u_complex
    beta = kappa - rho * xi * iu
    discriminant = torch.sqrt(beta**2 + xi**2 * (iu + u_complex**2))
    ratio = (beta - discriminant) / (beta + discriminant)
    exp_term = torch.exp(-discriminant * tau)
    log_term = torch.log((1.0 - ratio * exp_term) / (1.0 - ratio))
    c_term = (kappa * theta / xi**2) * ((beta - discriminant) * tau - 2.0 * log_term)
    d_term = ((beta - discriminant) / xi**2) * (1.0 - exp_term) / (1.0 - ratio * exp_term)
    return torch.exp(c_term + d_term * v0)


def heston_cumulants(
    tau: float,
    v0: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    xi: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[float, float]:
    """First two cumulants of the log return for COS truncation.

    The truncation interval is an accuracy device rather than part of
    the price's value contract, so callers detach these from autograd.

    Args:
        tau: Time to maturity in years.
        v0: Initial variance as a scalar tensor.
        kappa: Mean-reversion speed as a scalar tensor.
        theta: Long-run variance as a scalar tensor.
        xi: Volatility of variance as a scalar tensor.
        rho: Driver correlation as a scalar tensor.

    Returns:
        Tuple of ``(c1, c2)`` as floats.
    """
    decay = math.exp(-float(kappa) * tau)
    k, t, x, r, v = (float(kappa), float(theta), float(xi), float(rho), float(v0))
    c1 = (1.0 - decay) * (t - v) / (2.0 * k) - 0.5 * t * tau
    c2 = (
        x * tau * k * decay * (v - t) * (8.0 * k * r - 4.0 * x)
        + k * r * x * (1.0 - decay) * (16.0 * t - 8.0 * v)
        + 2.0 * t * k * tau * (-4.0 * k * r * x + x**2 + 4.0 * k**2)
        + x**2 * ((t - 2.0 * v) * math.exp(-2.0 * k * tau) + t * (6.0 * decay - 7.0) + 2.0 * v)
        + 8.0 * k**2 * (v - t) * (1.0 - decay)
    ) / (8.0 * k**3)
    return c1, c2
