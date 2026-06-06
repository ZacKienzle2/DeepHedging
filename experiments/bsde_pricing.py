"""Deep BSDE pricing accuracy against dimension.

The deep BSDE method prices by solving the backward equation of the
pricing PDE with a learned gradient network, so its cost grows linearly
in dimension where grid methods grow exponentially. The geometric
basket call is the standard high-dimensional golden because the
geometric average of driftless lognormals is itself lognormal, giving
an exact Black-Scholes reference in any dimension. The solver drives
independent Brownian motions, so the effective basket volatility is the
single-asset volatility shrunk by the square root of dimension, and the
geometric average is not a martingale; the convexity correction enters
the forward exactly. Each run records the solved price, the closed
form, and the terminal-matching loss certificate.

    uv run python experiments/bsde_pricing.py [--smoke]
"""

import argparse
import math
import time

import torch

from deephedging.bsde import BSDEConfig, BSDEProblem, DeepBSDESolver, train_bsde
from deephedging.experiment import ExperimentRecord, append_record, capture_provenance, load_records

DIMENSIONS = (1, 5, 20, 50)
SIGMA = 0.2
MATURITY = 1.0
N_STEPS = 20
STRIKE = 100.0
SPOT = 100.0
RESULTS = "experiments/bsde_pricing.jsonl"


def geometric_basket_reference(dim: int) -> float:
    """Prices the geometric basket call in closed form.

    The geometric average of independent driftless lognormals is
    lognormal with volatility shrunk by the square root of dimension
    and a convexity drift below the martingale forward.

    Args:
        dim: Number of assets.

    Returns:
        The exact call price.
    """
    sigma_g = SIGMA / math.sqrt(dim)
    forward = SPOT * math.exp(-0.5 * (SIGMA**2 - sigma_g**2) * MATURITY)
    d1 = (math.log(forward / STRIKE) + 0.5 * sigma_g**2 * MATURITY) / (
        sigma_g * math.sqrt(MATURITY)
    )
    d2 = d1 - sigma_g * math.sqrt(MATURITY)
    normal = torch.distributions.Normal(0.0, 1.0)
    return forward * float(normal.cdf(torch.tensor(d1))) - STRIKE * float(
        normal.cdf(torch.tensor(d2))
    )


def geometric_terminal(x: torch.Tensor) -> torch.Tensor:
    """Computes the geometric basket call payoff.

    Args:
        x: Terminal prices of shape ``(n_paths, dim)``.

    Returns:
        Payoff per path of shape ``(n_paths,)``.
    """
    geometric = torch.exp(torch.log(x).mean(dim=1))
    return torch.clamp(geometric - STRIKE, min=0.0)


def main() -> None:
    """Runs the dimension sweep and prints the accuracy table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    iterations = 50 if arguments.smoke else 3000
    batch_paths = 256 if arguments.smoke else 1024
    seeds = (1,) if arguments.smoke else (1, 2, 3)
    dimensions = (1, 5) if arguments.smoke else DIMENSIONS

    results_path = RESULTS.replace(".jsonl", "_smoke.jsonl") if arguments.smoke else RESULTS
    completed = {record.name for record in load_records(results_path)}
    for dim in dimensions:
        reference = geometric_basket_reference(dim)
        hidden = (64, 64) if dim <= 5 else (128, 128)
        for seed in seeds:
            run_name = f"bsde/dim{dim}/seed{seed}"
            if run_name in completed:
                print(f"skip {run_name} (already recorded)")
                continue
            torch.manual_seed(seed)
            problem = BSDEProblem(
                dim=dim,
                x0=SPOT,
                sigma=SIGMA,
                maturity=MATURITY,
                n_steps=N_STEPS,
                terminal=geometric_terminal,
            )
            solver = DeepBSDESolver(dim=dim, hidden_sizes=hidden).to(arguments.device)
            config = BSDEConfig(
                n_iterations=iterations, batch_paths=batch_paths, lr=1e-3, seed=seed
            )
            started = time.perf_counter()
            result = train_bsde(problem, solver, config)
            duration = time.perf_counter() - started
            parameters = sum(p.numel() for p in solver.parameters())
            relative_error = abs(result.y0 - reference) / reference
            append_record(
                results_path,
                ExperimentRecord(
                    name=run_name,
                    provenance=capture_provenance(),
                    train_config={
                        "n_iterations": config.n_iterations,
                        "batch_paths": config.batch_paths,
                        "lr": config.lr,
                        "seed": config.seed,
                    },
                    setup={
                        "dim": dim,
                        "sigma": SIGMA,
                        "maturity": MATURITY,
                        "n_steps": N_STEPS,
                        "strike": STRIKE,
                        "hidden_sizes": hidden,
                        "parameters": parameters,
                        "reference": reference,
                        "y0": result.y0,
                        "relative_error": relative_error,
                        "final_loss": result.final_loss,
                    },
                    losses=result.losses,
                    duration_seconds=duration,
                ),
            )
            print(
                f"dim {dim:3d} seed {seed}  y0 {result.y0:8.4f}  "
                f"reference {reference:8.4f}  rel err {relative_error:7.4%}  "
                f"params {parameters:6d}  {duration:6.1f}s"
            )


if __name__ == "__main__":
    main()
