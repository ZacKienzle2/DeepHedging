"""Option Greeks by automatic differentiation."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

PriceFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Greeks:
    """First- and second-order sensitivities of an option price.

    Attributes:
        delta: Sensitivity to the spot, ``dPrice/dSpot``.
        gamma: Curvature in the spot, ``d2Price/dSpot2``.
        vega: Sensitivity to volatility, ``dPrice/dSigma``.
        theta: Time decay, ``-dPrice/dTau``; negative for a long option whose
            value erodes as maturity approaches.
        rho: Sensitivity to the rate, ``dPrice/dRate``.
    """

    delta: torch.Tensor
    gamma: torch.Tensor
    vega: torch.Tensor
    theta: torch.Tensor
    rho: torch.Tensor


def european_greeks(
    price_fn: PriceFunction,
    spot: torch.Tensor | float,
    sigma: torch.Tensor | float,
    tau: torch.Tensor | float,
    rate: torch.Tensor | float = 0.0,
) -> Greeks:
    """Greeks of a price function by automatic differentiation.

    Differentiating the pricing map is exact to machine precision, where a
    finite-difference bump trades truncation error against catastrophic
    cancellation and has to be retuned for each Greek and each regime. The
    inputs broadcast to a common shape and are differentiated elementwise, so
    one call returns a whole sensitivity surface rather than a single point.

    The price function must be written in differentiable tensor operations of
    its four arguments and accept them at the broadcast shape. Closed forms
    expressed in plain tensor algebra qualify; a function that branches on a
    scalar argument does not vectorise and must be called per point. A Monte
    Carlo price qualifies only through a pathwise estimator, never a
    score-function one, whose gradient targets a different quantity.

    Args:
        price_fn: Map ``(spot, sigma, tau, rate) -> price`` in tensor ops.
        spot: Spot price; scalar or tensor.
        sigma: Volatility; scalar or tensor.
        tau: Time to maturity; scalar or tensor.
        rate: Continuously compounded interest rate; scalar or tensor.

    Returns:
        The :class:`Greeks` evaluated elementwise on the broadcast inputs.
    """
    spot_t, sigma_t, tau_t, rate_t = (
        torch.as_tensor(value, dtype=torch.float64) for value in (spot, sigma, tau, rate)
    )
    spot_v, sigma_v, tau_v, rate_v = (
        tensor.clone().requires_grad_(True)
        for tensor in torch.broadcast_tensors(spot_t, sigma_t, tau_t, rate_t)
    )
    price = price_fn(spot_v, sigma_v, tau_v, rate_v)
    delta, vega, decay, rho = torch.autograd.grad(
        price.sum(), (spot_v, sigma_v, tau_v, rate_v), create_graph=True
    )
    gamma = torch.autograd.grad(delta.sum(), spot_v)[0]
    return Greeks(
        delta=delta.detach(),
        gamma=gamma.detach(),
        vega=vega.detach(),
        theta=(-decay).detach(),
        rho=rho.detach(),
    )
