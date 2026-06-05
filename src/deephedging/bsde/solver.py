"""Deep BSDE solver in the style of Han, Jentzen, and E.

Parameterises ``Y_0`` as a learned scalar and ``Z_t`` as a single network
shared across time steps, evolves ``Y`` forward by the Euler scheme, and
minimises the terminal mismatch ``E|Y_T - g(X_T)|^2``. For a generator
uniformly Lipschitz in ``(y, z)`` the achieved loss is an a posteriori
bound on the ``(Y, Z)`` error up to a constant ``C ~ exp(c L^2 T)``, so
the final loss is reported alongside the price.
"""

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from deephedging.bsde.problem import BSDEProblem


class DeepBSDESolver(nn.Module):
    """Learned ``(Y_0, Z)`` pair for a fixed BSDE problem.

    Attributes:
        y0: Learned scalar value of the solution at inception.
        z_net: Shared network mapping ``(t, log(x / x0))`` to ``Z_t``.
    """

    def __init__(self, dim: int, hidden_sizes: tuple[int, ...] = (64, 64)) -> None:
        """Initialises the solver networks.

        Args:
            dim: Dimension of the forward state.
            hidden_sizes: Widths of the hidden layers of the ``Z`` network.
        """
        super().__init__()
        self.dim = dim
        self.y0 = nn.Parameter(torch.zeros(()))
        layers: list[nn.Module] = []
        width = dim + 1
        for size in hidden_sizes:
            layers.append(nn.Linear(width, size))
            layers.append(nn.SiLU())
            width = size
        layers.append(nn.Linear(width, dim))
        self.z_net = nn.Sequential(*layers)

    def forward(
        self, problem: BSDEProblem, n_paths: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulates the coupled forward-backward system.

        Args:
            problem: The BSDE problem to integrate.
            n_paths: Number of Monte Carlo paths.
            generator: Optional RNG for reproducible draws.

        Returns:
            Tuple of ``Y_T`` and ``g(X_T)``, each of shape ``(n_paths,)``.
        """
        device = self.y0.device
        dtype = self.y0.dtype
        dt = problem.maturity / problem.n_steps
        sqrt_dt = math.sqrt(dt)
        noise = torch.randn(
            (problem.n_steps, n_paths, problem.dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        dw = sqrt_dt * noise
        log_x = torch.zeros((n_paths, problem.dim), dtype=dtype, device=device)
        y = self.y0.expand(n_paths)
        drift = (problem.mu - 0.5 * problem.sigma**2) * dt
        for k in range(problem.n_steps):
            t = torch.full((n_paths, 1), k * dt, dtype=dtype, device=device)
            x = problem.x0 * torch.exp(log_x)
            z = self.z_net(torch.cat((t, log_x), dim=1))
            f = problem.generator(t[0, 0], x, y, z)
            y = y - f * dt + (z * dw[k]).sum(dim=1)
            log_x = log_x + drift + problem.sigma * dw[k]
        terminal = problem.terminal(problem.x0 * torch.exp(log_x))
        return y, terminal


@dataclass(frozen=True)
class BSDEConfig:
    """Hyperparameters for training a deep BSDE solver.

    Attributes:
        n_iterations: Number of optimisation steps, each on fresh paths.
        batch_paths: Simulated paths per step.
        lr: Learning rate for the ``Z`` network.
        y0_lr: Learning rate for the scalar ``Y_0``; defaults to ``10 * lr``
            so the price estimate tracks faster than the hedge surface.
        seed: Optional seed for reproducible path generation.
    """

    n_iterations: int = 1500
    batch_paths: int = 512
    lr: float = 1e-3
    y0_lr: float | None = None
    seed: int | None = None


@dataclass
class BSDEResult:
    """Outcome of a deep BSDE training run.

    Attributes:
        y0: Solved value at inception (the price for pricing generators).
        final_loss: Terminal-matching loss at the last iteration; an
            a posteriori certificate of the ``(Y, Z)`` error up to a
            Lipschitz-dependent constant.
        losses: Loss recorded at every iteration.
    """

    y0: float
    final_loss: float
    losses: list[float] = field(default_factory=list)


def train_bsde(
    problem: BSDEProblem,
    solver: DeepBSDESolver,
    config: BSDEConfig,
) -> BSDEResult:
    """Trains a deep BSDE solver by SGD on the terminal-matching loss.

    ``Y_0`` is warm-started at the Monte Carlo mean of the terminal
    condition, which is exact for the zero generator and a good initial
    guess for Lipschitz pricing generators.

    Args:
        problem: The BSDE problem to solve.
        solver: The solver whose parameters are optimised.
        config: Training hyperparameters.

    Returns:
        The solved ``y0``, the final loss certificate, and the loss history.
    """
    if solver.dim != problem.dim:
        raise ValueError(f"solver dim {solver.dim} != problem dim {problem.dim}")
    device = solver.y0.device
    generator: torch.Generator | None = None
    if config.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed)
    with torch.no_grad():
        _, terminal = solver(problem, config.batch_paths, generator=generator)
        solver.y0.copy_(terminal.mean())
    y0_lr = config.y0_lr if config.y0_lr is not None else 10.0 * config.lr
    optimizer = torch.optim.Adam(
        [
            {"params": list(solver.z_net.parameters()), "lr": config.lr},
            {"params": [solver.y0], "lr": y0_lr},
        ]
    )
    loss_history: list[torch.Tensor] = []
    for _ in range(config.n_iterations):
        y_terminal, target = solver(problem, config.batch_paths, generator=generator)
        loss = torch.mean((y_terminal - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach())
    losses = [float(value) for value in loss_history]
    return BSDEResult(y0=float(solver.y0.detach()), final_loss=losses[-1], losses=losses)
