"""Throughput baselines for simulators, pricing, and the episode engine.

Run from the repository root. Generation, pricing, and training rates
share the paths-per-second unit but measure different work per path, so
they are comparable only within a row across devices or backends, never
across rows: a faster generator cannot lift the training rate beyond
generation's small share of the step. The pricing rows time the full
estimate including payoff and reduction on both the fold and the grid
route, so their ratio reflects the pricer a user calls rather than
generation in isolation.

    uv run python benchmarks/bench_paths.py [--device cuda] [--json out.json]
"""

import argparse
import json
import os
import time
from collections.abc import Callable

import torch

from deephedging import (
    CVaR,
    EuropeanCall,
    FeedForwardPolicy,
    GBMSimulator,
    HestonSimulator,
    NoiseSpec,
    ProportionalCost,
    hedge_pnl,
)


def _timed(fn: Callable[[], None], repeats: int, device: str) -> float:
    for _ in range(3):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    """Times path generation and a full training step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--paths", type=int, default=65_536)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)

    gbm = GBMSimulator(s0=100.0, sigma=0.2, maturity=0.25, n_steps=args.steps, device=args.device)
    heston = HestonSimulator(
        s0=100.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        maturity=0.25,
        n_steps=args.steps,
        device=args.device,
    )
    policy = FeedForwardPolicy(hidden_sizes=(64, 64)).to(args.device)
    payoff = EuropeanCall(strike=100.0)
    cost = ProportionalCost(rate=1e-3)
    risk = CVaR(alpha=0.95).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    noise = NoiseSpec(seed=1)

    def gen_gbm() -> None:
        gbm.simulate(args.paths, noise=noise)

    def gen_heston() -> None:
        heston.simulate(args.paths, noise=noise)

    def train_step() -> None:
        state = gbm.simulate(args.paths, noise=noise)
        loss = risk(-hedge_pnl(state, policy, payoff, cost, premium=4.0, amp=args.amp))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    results = {
        "device": args.device,
        "paths": args.paths,
        "steps": args.steps,
        "gbm_generate_mpaths_per_s": args.paths / _timed(gen_gbm, args.repeats, args.device) / 1e6,
        "heston_generate_mpaths_per_s": args.paths
        / _timed(gen_heston, args.repeats, args.device)
        / 1e6,
        "train_step_mpaths_per_s": args.paths / _timed(train_step, args.repeats, args.device) / 1e6,
    }
    if args.device == "cuda":
        from deephedging.market import CudaGBMSimulator, CudaHestonSimulator, kernels_available

        if kernels_available():
            fused_gbm = CudaGBMSimulator(s0=100.0, sigma=0.2, maturity=0.25, n_steps=args.steps)
            fused_heston = CudaHestonSimulator(
                s0=100.0,
                v0=0.04,
                kappa=1.5,
                theta=0.04,
                xi=0.5,
                rho=-0.7,
                maturity=0.25,
                n_steps=args.steps,
            )
            results["fused_gbm_mpaths_per_s"] = (
                args.paths
                / _timed(lambda: fused_gbm.simulate(args.paths, noise=noise), args.repeats, "cuda")
                / 1e6
            )
            results["fused_heston_mpaths_per_s"] = (
                args.paths
                / _timed(
                    lambda: fused_heston.simulate(args.paths, noise=noise), args.repeats, "cuda"
                )
                / 1e6
            )

            from deephedging import UpAndOutCall
            from deephedging.pricing import MonteCarloPricer

            barrier_steps = 250
            pricing_sim = CudaHestonSimulator(
                s0=100.0,
                v0=0.04,
                kappa=1.5,
                theta=0.04,
                xi=0.5,
                rho=-0.7,
                maturity=1.0,
                n_steps=barrier_steps,
            )
            barrier = UpAndOutCall(strike=100.0, barrier=130.0)
            pricer = MonteCarloPricer(n_paths=args.paths, seed=1)

            def fold_price() -> None:
                pricer.price(barrier, pricing_sim)

            def grid_price() -> None:
                state = pricing_sim.simulate(args.paths, noise=noise)
                barrier(state.spot).mean()

            results["fold_price_mpaths_per_s"] = (
                args.paths / _timed(fold_price, args.repeats, "cuda") / 1e6
            )
            results["grid_price_mpaths_per_s"] = (
                args.paths / _timed(grid_price, args.repeats, "cuda") / 1e6
            )
    for key, value in results.items():
        print(f"{key}: {value if isinstance(value, str | int) else f'{value:.3f}'}")
    if args.json:
        with open(args.json, "w", encoding="ascii") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
