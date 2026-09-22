"""Deep backward dynamic programming of Hure, Pham and Warin.

The DBDP1 scheme (Hure, Pham and Warin, 2020, eq. 3.6) solves the BSDE one
date at a time from maturity backwards. At date ``i`` a network pair
``(U_i, Z_i)`` minimises ``E|U_(i+1)(X_(i+1)) - F(t_i, X_i, U_i, Z_i)|^2``
with ``F = y - f(t, x, y, z) dt + z . dW``, the Euler step of the backward
equation, against the already fitted value at the next date. Each fit
starts from the next date's weights, the warm start the authors credit
with avoiding the poor local minima of the global deep BSDE loss and with
cutting the iterations of every fit after the first. The reflected variant
(their RDBDP, eq. 3.10) replaces each fitted value by ``max(U_i, g)``,
which solves the optimal stopping problem of an American claim.

The forward state is a lognormal diffusion, so ``X_i`` is drawn exactly at
date ``i`` and ``X_(i+1)`` one exact step later. Each fit samples fresh
pairs rather than a stored path set, and memory does not grow with the
number of dates, the limitation the authors note for the global scheme.
"""

import copy
import math
from dataclasses import dataclass, field

import torch
from torch import nn

from deephedging.bsde.problem import BSDEProblem
from deephedging.market.noise import NoiseSpec
from deephedging.networks import mlp


class BackwardPair(nn.Module):
    """Value and volatility networks of one date, as one two-headed network.

    The network maps ``log(x / x0)`` to ``1 + dim`` outputs, the value
    ``U_i`` and the ``dim`` components of ``Z_i``, so both heads share one
    chain of matrix products (Hure, Pham and Warin allow either layout,
    Remark 3.1).

    Attributes:
        net: The two-headed network.
    """

    def __init__(self, dim: int, hidden_sizes: tuple[int, ...]) -> None:
        """Initialises the network.

        Args:
            dim: Dimension of the forward state.
            hidden_sizes: Widths of the hidden layers.
        """
        super().__init__()
        self.net = mlp(dim, hidden_sizes, 1 + dim)

    def forward(self, log_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the value and volatility heads.

        Args:
            log_x: Log forward state relative to ``x0``, shape
                ``(n_paths, dim)``.

        Returns:
            The value of shape ``(n_paths,)`` and ``Z`` of shape
            ``(n_paths, dim)``.
        """
        output = self.net(log_x)
        return output[:, 0], output[:, 1:]


@dataclass(frozen=True)
class BackwardConfig:
    """Hyperparameters of the backward scheme.

    Attributes:
        first_iterations: Optimisation steps at the last date, fitted from
            scratch against the terminal condition.
        iterations: Optimisation steps at each earlier date, warm-started
            from the next date's network.
        batch_paths: Sampled pairs per step.
        lr: Learning rate.
        hidden_sizes: Widths of the hidden layers of each date's network.
        american: Whether to reflect each value on the terminal payoff,
            the RDBDP scheme for optimal stopping.
        seed: Optional experiment seed; each step draws its own
            addressable noise stream.
        device: Device of the networks and samples.
    """

    first_iterations: int = 1000
    iterations: int = 200
    batch_paths: int = 1024
    lr: float = 1e-3
    hidden_sizes: tuple[int, ...] = (32, 32)
    american: bool = False
    seed: int | None = None
    device: str = "cpu"


@dataclass
class BackwardResult:
    """Outcome of the backward scheme.

    Attributes:
        y0: Solved value at inception.
        losses: Final local loss at each date, in date order.
        networks: Fitted network of each date, in date order.
    """

    y0: float
    losses: list[float] = field(default_factory=list)
    networks: list[BackwardPair] = field(default_factory=list)


def solve_backward(problem: BSDEProblem, config: BackwardConfig) -> BackwardResult:
    """Solves a BSDE by deep backward dynamic programming.

    Args:
        problem: The BSDE problem to solve.
        config: Hyperparameters of the scheme.

    Returns:
        The value at inception, the local losses and the fitted networks.
    """
    dt = problem.maturity / problem.n_steps
    drift = problem.mu - 0.5 * problem.sigma**2
    base_noise = NoiseSpec(seed=config.seed) if config.seed is not None else None
    stride = max(config.first_iterations, config.iterations)
    network = BackwardPair(problem.dim, config.hidden_sizes).to(config.device)
    following: BackwardPair | None = None
    losses: list[float] = []
    networks: list[BackwardPair] = []
    for date in reversed(range(problem.n_steps)):
        if following is not None:
            network = copy.deepcopy(following).requires_grad_(True)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=config.lr, fused=config.device.startswith("cuda")
        )
        time = date * dt
        steps = config.first_iterations if following is None else config.iterations
        loss = torch.zeros(())
        for step in range(steps):
            noise = base_noise.child(date * stride + step) if base_noise is not None else None
            generator = noise.torch_generator(config.device) if noise is not None else None
            draws = torch.randn(
                (2, config.batch_paths, problem.dim), device=config.device, generator=generator
            )
            log_x = drift * time + problem.sigma * math.sqrt(time) * draws[0]
            increment = math.sqrt(dt) * draws[1]
            log_next = log_x + drift * dt + problem.sigma * increment
            x = problem.x0 * torch.exp(log_x)
            with torch.no_grad():
                x_next = problem.x0 * torch.exp(log_next)
                target = problem.terminal(x_next)
                if following is not None:
                    value, _ = following(log_next)
                    target = torch.maximum(value, target) if config.american else value
            value, z = network(log_x)
            step_time = torch.tensor(time, device=config.device)
            estimate = (
                value
                - problem.generator(step_time, x, value, z) * dt
                + torch.linalg.vecdot(z, increment)
            )
            loss = nn.functional.mse_loss(estimate, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
        following = network.requires_grad_(False)
        networks.append(following)
    with torch.no_grad():
        origin = torch.zeros((1, problem.dim), device=config.device)
        value, _ = networks[-1](origin)
        y0 = value[0]
        if config.american:
            y0 = torch.maximum(y0, problem.terminal(problem.x0 * torch.exp(origin))[0])
    return BackwardResult(y0=float(y0), losses=losses[::-1], networks=networks[::-1])
