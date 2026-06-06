"""Throughput baseline for simulators and the episode engine.

Run from the repository root. Prints paths per second for generation
alone and for a full training step, the numbers any future fused
generator must beat.

    uv run python benchmarks/bench_paths.py [--device cuda] [--json out.json]
"""

import argparse
import json
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
    for key, value in results.items():
        print(f"{key}: {value if isinstance(value, str | int) else f'{value:.3f}'}")
    if args.json:
        with open(args.json, "w", encoding="ascii") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
