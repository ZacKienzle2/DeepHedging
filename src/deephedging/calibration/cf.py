"""COS-method vanilla pricing from a characteristic function.

The Fang-Oosterlee expansion prices a whole strike grid as one batched
contraction against the characteristic function evaluated at a fixed
set of real frequencies. Nothing is adaptive and every operation is a
fixed-shape tensor op, so the price is differentiable in the model
parameters by plain autograd, which is what surface calibration
gradients require. Payoff coefficients for the call are closed form,
so no inner quadrature exists to corrupt gradients.
"""

import math
from collections.abc import Callable

import torch

CharacteristicFn = Callable[[torch.Tensor], torch.Tensor]


def _chi(omega: torch.Tensor, a: float, b: float) -> torch.Tensor:
    upper = omega * (b - a)
    lower = omega * (0.0 - a)
    return (
        torch.cos(upper) * math.exp(b)
        - torch.cos(lower)
        + omega * torch.sin(upper) * math.exp(b)
        - omega * torch.sin(lower)
    ) / (1.0 + omega**2)


def _psi(omega: torch.Tensor, a: float, b: float) -> torch.Tensor:
    upper = omega * (b - a)
    lower = omega * (0.0 - a)
    values = torch.empty_like(omega)
    values[1:] = (torch.sin(upper[1:]) - torch.sin(lower[1:])) / omega[1:]
    values[0] = b
    return values


def cos_call_price(
    characteristic_fn: CharacteristicFn,
    s0: float,
    strikes: torch.Tensor,
    a: float,
    b: float,
    n_terms: int = 160,
) -> torch.Tensor:
    """Prices European calls by the COS expansion under a zero rate.

    Args:
        characteristic_fn: Characteristic function of the log return
            ``log(S_tau / S_0)``, mapping real frequencies to complex
            values.
        s0: Initial spot price.
        strikes: Strike grid of shape ``(n_strikes,)`` in float64.
        a: Lower truncation bound for the log return density.
        b: Upper truncation bound for the log return density.
        n_terms: Number of cosine expansion terms.

    Returns:
        Call prices of shape ``(n_strikes,)`` in float64.
    """
    k = torch.arange(n_terms, dtype=torch.float64)
    omega = k * math.pi / (b - a)
    payoff_coefficients = (2.0 / (b - a)) * (_chi(omega, a, b) - _psi(omega, a, b))
    cf_values = characteristic_fn(omega)
    x = torch.log(s0 / strikes)
    phase = torch.exp(1j * omega.unsqueeze(0) * (x.unsqueeze(1) - a))
    terms = (cf_values.unsqueeze(0) * phase).real * payoff_coefficients.unsqueeze(0)
    terms[:, 0] = terms[:, 0] * 0.5
    return strikes * terms.sum(dim=1)
