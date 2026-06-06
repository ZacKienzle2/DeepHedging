"""Hedging instrument span under stochastic volatility.

A call under Heston carries volatility risk the spot cannot span, so a
spot-only hedge, deep or analytic, leaves a tail driven by variance
moves. Adding a variance swap as a second tradable closes the span, and
the experiment measures how much tail risk the extra instrument
removes. Three arms share one premium and one set of evaluation noise
streams: the Black-Scholes delta hedge at the long-run volatility, a
deep policy trading the spot alone, and a deep policy trading spot and
swap jointly. The swap trades at its own price scale, so a proportional
cost rate charges it almost nothing relative to the spot; cost
sensitivity therefore reads on the spot leg. Every run appends a
full-provenance record so the table regenerates from the store alone.

    uv run python experiments/multi_instrument.py [--smoke]
"""

import argparse
import time

import torch

from deephedging import (
    CVaR,
    EuropeanCall,
    FeedForwardPolicy,
    HestonSimulator,
    HestonVarianceSwapSimulator,
    MonteCarloPricer,
    MultiAssetFeatures,
    NoCost,
    NoiseSpec,
    ProportionalCost,
    SingleAssetPayoff,
    TrainConfig,
    delta_hedge_positions,
    hedge_pnl,
    pnl_from_positions,
    pnl_summary,
    train,
)
from deephedging.experiment import ExperimentRecord, append_record, load_records
from deephedging.frictions.base import CostModel
from deephedging.training.trainer import TrainResult

MATURITY = 0.25
N_STEPS = 30
HESTON_PARAMETERS = (0.04, 1.5, 0.04, 0.5, -0.7)
SWAP_MATURITY = 0.5
STRIKE = 100.0
ALPHA = 0.95
RESULTS = "experiments/multi_instrument.jsonl"
EVAL_SEED = 991
PREMIUM_SEED = 7


def make_heston(device: str) -> HestonSimulator:
    """Builds the single-asset stochastic volatility market.

    Args:
        device: Device for path generation.

    Returns:
        The Heston simulator underlying every arm.
    """
    v0, kappa, theta, xi, rho = HESTON_PARAMETERS
    return HestonSimulator(
        s0=100.0,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        maturity=MATURITY,
        n_steps=N_STEPS,
        device=device,
    )


def cost_model(rate: float) -> CostModel:
    """Builds the cost model for a rate.

    Args:
        rate: Proportional cost rate; zero means frictionless.

    Returns:
        The cost model.
    """
    return NoCost() if rate == 0.0 else ProportionalCost(rate=rate)


def main() -> None:
    """Runs the instrument-span grid and prints the comparison table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    iterations = 30 if arguments.smoke else 2000
    batch_paths = 1024 if arguments.smoke else 65_536
    eval_paths = 4096 if arguments.smoke else 200_000
    premium_paths = 65_536 if arguments.smoke else 1_000_000
    seeds = (1,) if arguments.smoke else (1, 2, 3, 4, 5)
    cost_rates = (0.0,) if arguments.smoke else (0.0, 1e-3)

    results_path = RESULTS.replace(".jsonl", "_smoke.jsonl") if arguments.smoke else RESULTS
    completed = {record.name for record in load_records(results_path)}
    heston = make_heston(arguments.device)
    two_asset = HestonVarianceSwapSimulator(heston=heston, vs_maturity=SWAP_MATURITY)
    call = EuropeanCall(strike=STRIKE)
    wrapped_call = SingleAssetPayoff(inner=call)
    estimate = MonteCarloPricer(n_paths=premium_paths, seed=PREMIUM_SEED).price(call, heston)
    premium = estimate.value
    print(f"call premium {premium:.4f} (se {estimate.standard_error:.4f})")

    _, _, theta, _, _ = HESTON_PARAMETERS
    long_run_vol = theta**0.5
    eval_single = heston.simulate(eval_paths, noise=NoiseSpec(seed=EVAL_SEED))
    eval_double = two_asset.simulate(eval_paths, noise=NoiseSpec(seed=EVAL_SEED))
    deltas = delta_hedge_positions(eval_single.spot, STRIKE, long_run_vol, MATURITY)

    print("delta hedge on common evaluation paths:")
    for cost_rate in cost_rates:
        delta_pnl = pnl_from_positions(
            eval_single, deltas, call, cost_model(cost_rate), premium=premium
        )
        summary = pnl_summary(delta_pnl)
        print(f"  cost {cost_rate:6.4f}  {summary}")
        record_name = f"multi-instrument/delta/cost{cost_rate}"
        if record_name not in completed:
            append_record(
                results_path,
                ExperimentRecord.from_run(
                    name=record_name,
                    config=TrainConfig(n_iterations=0),
                    setup={
                        "arm": "delta",
                        "cost_rate": cost_rate,
                        "premium": premium,
                        "eval_seed": EVAL_SEED,
                        **summary,
                    },
                    result=TrainResult(),
                    duration_seconds=0.0,
                ),
            )

    for cost_rate in cost_rates:
        for arm in ("spot", "swap"):
            for seed in seeds:
                run_name = f"multi-instrument/{arm}/cost{cost_rate}/seed{seed}"
                if run_name in completed:
                    print(f"skip {run_name} (already recorded)")
                    continue
                torch.manual_seed(seed)
                if arm == "spot":
                    policy = FeedForwardPolicy(n_features=3, hidden_sizes=(64, 64))
                    simulator, payoff, feature_map = heston, call, None
                    eval_state = eval_single
                else:
                    policy = FeedForwardPolicy(n_features=5, hidden_sizes=(64, 64), n_outputs=2)
                    simulator, payoff = two_asset, wrapped_call
                    feature_map = MultiAssetFeatures(n_assets=2)
                    eval_state = eval_double
                policy = policy.to(arguments.device)
                config = TrainConfig(
                    n_iterations=iterations,
                    batch_paths=batch_paths,
                    lr=1e-3,
                    seed=seed,
                    graph_episode=(arguments.device == "cuda"),
                )
                started = time.perf_counter()
                result = train(
                    simulator,
                    policy,
                    payoff,
                    cost_model(cost_rate),
                    CVaR(alpha=ALPHA),
                    config,
                    premium=premium,
                    feature_map=feature_map,
                )
                duration = time.perf_counter() - started
                with torch.no_grad():
                    pnl = hedge_pnl(
                        eval_state,
                        policy,
                        payoff,
                        cost_model(cost_rate),
                        premium=premium,
                        feature_map=feature_map,
                    )
                summary = pnl_summary(pnl)
                append_record(
                    results_path,
                    ExperimentRecord.from_run(
                        name=run_name,
                        config=config,
                        setup={
                            "arm": arm,
                            "cost_rate": cost_rate,
                            "premium": premium,
                            "alpha": ALPHA,
                            "eval_seed": EVAL_SEED,
                            "heston": HESTON_PARAMETERS,
                            "swap_maturity": SWAP_MATURITY,
                            **summary,
                        },
                        result=result,
                        duration_seconds=duration,
                    ),
                )
                print(
                    f"{arm:5s} cost {cost_rate:6.4f} seed {seed}  "
                    f"es95 {summary['es_95']:7.4f}  es99 {summary['es_99']:7.4f}  "
                    f"mean {summary['mean']:7.4f}  {duration:6.1f}s"
                )


if __name__ == "__main__":
    main()
