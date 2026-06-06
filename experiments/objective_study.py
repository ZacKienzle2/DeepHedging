"""Risk objectives shaping the learned hedge.

The risk measure is the only thing telling the optimiser which losses
matter, so changing it changes the policy. Expected shortfall at rising
confidence pushes the hedge toward insuring ever deeper tails at the
price of mean performance, while the entropic measure weights losses
exponentially and interpolates smoothly between mean and worst case as
risk aversion grows. One market, one liability, one cost level, one
parameter budget; only the objective moves. Every arm scores on common
evaluation paths and reports the full summary, so the table shows how
each objective trades the mean against the tails it was asked to fear.
The entropic gradient concentrates on the worst paths as risk aversion
grows, so the shared batch size is chosen large enough to keep its
effective sample workable at the highest aversion in the grid.

    uv run python experiments/objective_study.py [--smoke]
"""

import argparse
import gc
import time

import torch

from deephedging import (
    CVaR,
    Entropic,
    EuropeanCall,
    FeedForwardPolicy,
    GBMSimulator,
    NoiseSpec,
    ProportionalCost,
    TrainConfig,
    bs_call_price,
    hedge_pnl,
    pnl_summary,
    train,
)
from deephedging.experiment import ExperimentRecord, append_record, load_records
from deephedging.risk.base import RiskMeasure

SIGMA = 0.2
MATURITY = 30 / 365
N_STEPS = 30
STRIKE = 100.0
COST_RATE = 1e-3
RESULTS = "experiments/objective_study.jsonl"
EVAL_SEED = 991


def make_objective(name: str) -> RiskMeasure:
    """Builds one risk objective by name.

    Args:
        name: ``cvar`` or ``entropic`` followed by its level, such as
            ``cvar0.95`` or ``entropic5.0``.

    Returns:
        The risk measure.

    Raises:
        ValueError: If the name parses to no known objective.
    """
    if name.startswith("cvar"):
        return CVaR(alpha=float(name[4:]))
    if name.startswith("entropic"):
        return Entropic(risk_aversion=float(name[8:]))
    raise ValueError(f"unknown objective {name}")


def main() -> None:
    """Runs the objective grid and prints the comparison table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    iterations = 30 if arguments.smoke else 3000
    batch_paths = 1024 if arguments.smoke else 65_536
    eval_paths = 4096 if arguments.smoke else 200_000
    seeds = (1,) if arguments.smoke else (1, 2, 3, 4, 5)
    objectives = (
        ("cvar0.95",)
        if arguments.smoke
        else ("cvar0.9", "cvar0.95", "cvar0.99", "entropic1.0", "entropic5.0", "entropic10.0")
    )

    results_path = RESULTS.replace(".jsonl", "_smoke.jsonl") if arguments.smoke else RESULTS
    completed = {record.name for record in load_records(results_path)}
    simulator = GBMSimulator(
        s0=100.0, sigma=SIGMA, maturity=MATURITY, n_steps=N_STEPS, device=arguments.device
    )
    payoff = EuropeanCall(strike=STRIKE)
    cost = ProportionalCost(rate=COST_RATE)
    premium = float(bs_call_price(100.0, STRIKE, SIGMA, MATURITY))
    eval_state = simulator.simulate(eval_paths, noise=NoiseSpec(seed=EVAL_SEED))

    for objective_name in objectives:
        for seed in seeds:
            run_name = f"objective/{objective_name}/seed{seed}"
            if run_name in completed:
                print(f"skip {run_name} (already recorded)")
                continue
            torch.manual_seed(seed)
            policy = FeedForwardPolicy(n_features=3, hidden_sizes=(64, 64)).to(arguments.device)
            config = TrainConfig(
                n_iterations=iterations,
                batch_paths=batch_paths,
                lr=1e-3,
                seed=seed,
                checkpoint_steps=(arguments.device == "cuda"),
                graph_episode=(arguments.device == "cuda"),
            )
            started = time.perf_counter()
            result = train(
                simulator,
                policy,
                payoff,
                cost,
                make_objective(objective_name),
                config,
                premium=premium,
            )
            duration = time.perf_counter() - started
            with torch.no_grad():
                pnl = hedge_pnl(eval_state, policy, payoff, cost, premium=premium)
            summary = pnl_summary(pnl)
            append_record(
                results_path,
                ExperimentRecord.from_run(
                    name=run_name,
                    config=config,
                    setup={
                        "objective": objective_name,
                        "cost_rate": COST_RATE,
                        "premium": premium,
                        "eval_seed": EVAL_SEED,
                        **summary,
                    },
                    result=result,
                    duration_seconds=duration,
                ),
            )
            print(
                f"{objective_name:13s} seed {seed}  mean {summary['mean']:7.4f}  "
                f"std {summary['std']:6.4f}  es95 {summary['es_95']:7.4f}  "
                f"es99 {summary['es_99']:7.4f}  {duration:6.1f}s"
            )
            del policy, result
            gc.collect()
            if arguments.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
