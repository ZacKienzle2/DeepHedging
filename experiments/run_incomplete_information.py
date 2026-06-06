"""Incomplete-information ablation under Heston dynamics.

The instantaneous variance is hidden from every policy except the
oracle arm, so the spot-only feedforward policy is structurally blind,
the realised-volatility arm sees a hand-engineered estimate, and the
recurrent policy must learn to infer the hidden state from the return
history. The oracle bounds achievable performance from below in
expected shortfall and the blind arm from above; the two clean
contrasts are oracle versus realised, the value of observing variance
exactly, and realised versus blind, the value of any volatility signal.
Whether the learned recurrence beats the hand-crafted estimator is an
open hypothesis rather than a prediction, because both read the same
return history and realised quadratic variation is already its
sufficient statistic. The horizon spans one and a half mean-reversion
times so variance regimes move within an episode and memory has
something to track. Every arm trains per seed on its own noise streams
and evaluates on one common set of paths, so eval noise cancels in arm
comparisons, and every run appends a full-provenance record so the
table regenerates from the store alone.

    uv run python experiments/run_incomplete_information.py [--smoke]
"""

import argparse
import time

import torch

from deephedging import (
    CVaR,
    EuropeanCall,
    FeedForwardPolicy,
    HestonSimulator,
    NoiseSpec,
    ProportionalCost,
    RealizedVolFeatures,
    RecurrentPolicy,
    TrainConfig,
    VarianceFeatures,
    expected_shortfall,
    hedge_pnl,
    train,
)
from deephedging.experiment import ExperimentRecord, append_record, load_records
from deephedging.features import DefaultFeatures, FeatureMap
from deephedging.policies.base import HedgePolicy

MATURITY = 1.0
N_STEPS = 60
HESTON_PARAMETERS = (0.04, 1.5, 0.04, 0.5, -0.7)
STRIKE = 100.0
ALPHA = 0.95
RESULTS = "experiments/incomplete_information.jsonl"
EVAL_SEED = 991


def make_simulator(device: str) -> HestonSimulator:
    """Builds the partially observed market.

    Args:
        device: Device for path generation.

    Returns:
        The Heston simulator all arms share.
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


def make_arm(name: str, device: str) -> tuple[HedgePolicy, FeatureMap]:
    """Builds one policy-and-observation arm by name.

    Architectures are parameter-budget matched so capacity differences
    cannot masquerade as observability differences.

    Args:
        name: One of ``blind``, ``oracle``, ``realised``, ``recurrent``.
        device: Device for the policy.

    Returns:
        The policy and the feature map of the arm.

    Raises:
        ValueError: If the arm name is unknown.
    """
    step_size = MATURITY / N_STEPS
    if name == "blind":
        return FeedForwardPolicy(n_features=3, hidden_sizes=(64, 64)).to(device), DefaultFeatures()
    if name == "oracle":
        return (
            FeedForwardPolicy(n_features=4, hidden_sizes=(64, 64)).to(device),
            VarianceFeatures(),
        )
    if name == "realised":
        return (
            FeedForwardPolicy(n_features=4, hidden_sizes=(64, 64)).to(device),
            RealizedVolFeatures(window=20, step_size=step_size),
        )
    if name == "recurrent":
        return RecurrentPolicy(n_features=3, hidden_size=36).to(device), DefaultFeatures()
    raise ValueError(f"unknown arm {name}")


def main() -> None:
    """Runs the ablation grid and prints the result table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    base_iterations = 30 if arguments.smoke else 1500
    recurrent_iterations = 30 if arguments.smoke else 2500
    batch_paths = 1024 if arguments.smoke else 65_536
    eval_paths = 4096 if arguments.smoke else 200_000
    seeds = (1,) if arguments.smoke else (1, 2, 3, 4, 5, 6, 7, 8)
    cost_rates = (0.0,) if arguments.smoke else (0.0, 1e-3)

    results_path = RESULTS.replace(".jsonl", "_smoke.jsonl") if arguments.smoke else RESULTS
    completed = {record.name for record in load_records(results_path)}
    simulator = make_simulator(arguments.device)
    payoff = EuropeanCall(strike=STRIKE)
    rows = []
    for cost_rate in cost_rates:
        cost = ProportionalCost(rate=cost_rate)
        eval_state = simulator.simulate(eval_paths, noise=NoiseSpec(seed=EVAL_SEED))
        for arm in ("blind", "oracle", "realised", "recurrent"):
            for seed in seeds:
                run_name = f"incomplete-info/{arm}/cost{cost_rate}/seed{seed}"
                if run_name in completed:
                    print(f"skip {run_name} (already recorded)")
                    continue
                torch.manual_seed(seed)
                policy, feature_map = make_arm(arm, arguments.device)
                parameters = sum(p.numel() for p in policy.parameters())
                config = TrainConfig(
                    n_iterations=(recurrent_iterations if arm == "recurrent" else base_iterations),
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
                    cost,
                    CVaR(alpha=ALPHA),
                    config,
                    premium=0.0,
                    feature_map=feature_map,
                )
                duration = time.perf_counter() - started
                with torch.no_grad():
                    pnl = hedge_pnl(
                        eval_state,
                        policy,
                        payoff,
                        cost,
                        premium=0.0,
                        feature_map=feature_map,
                    )
                shortfall_95 = float(expected_shortfall(pnl, alpha=0.95))
                shortfall_99 = float(expected_shortfall(pnl, alpha=0.99))
                rows.append((arm, cost_rate, seed, parameters, shortfall_95, shortfall_99))
                append_record(
                    results_path,
                    ExperimentRecord.from_run(
                        name=f"incomplete-info/{arm}/cost{cost_rate}/seed{seed}",
                        config=config,
                        setup={
                            "arm": arm,
                            "cost_rate": cost_rate,
                            "parameters": parameters,
                            "alpha": ALPHA,
                            "es95": shortfall_95,
                            "es99": shortfall_99,
                            "eval_seed": 991,
                            "heston": HESTON_PARAMETERS,
                            "maturity": MATURITY,
                            "n_steps": N_STEPS,
                        },
                        result=result,
                        duration_seconds=duration,
                    ),
                )
                print(
                    f"{arm:9s} cost {cost_rate:6.4f} seed {seed}  "
                    f"params {parameters:5d}  es95 {shortfall_95:7.4f}  "
                    f"es99 {shortfall_99:7.4f}  {duration:6.1f}s"
                )

    print("\narm averages (es95 over seeds):")
    for cost_rate in cost_rates:
        for arm in ("blind", "oracle", "realised", "recurrent"):
            values = [r[4] for r in rows if r[0] == arm and r[1] == cost_rate]
            mean = sum(values) / len(values)
            print(f"  cost {cost_rate:6.4f} {arm:9s} {mean:7.4f}")


if __name__ == "__main__":
    main()
