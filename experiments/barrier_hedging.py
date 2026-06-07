"""Deep hedging of a barrier option where no analytic hedge exists.

An up-and-out call under Heston dynamics is the motivating case for
learned hedging. The contract is path dependent, the volatility is
stochastic, and no closed-form hedge ratio exists, so the practitioner
baseline is the Black-Scholes vanilla delta at the long-run volatility,
which neither sees the barrier nor stops trading after knockout. Two
deep arms isolate where the improvement comes from: one observes only
spot and time, the other adds the running maximum, the sufficient path
statistic for the knockout. All arms receive the same Monte Carlo
premium and score on common evaluation paths so comparisons are paired,
and every run appends a full-provenance record so the table regenerates
from the store alone.

    uv run python experiments/barrier_hedging.py [--smoke]
"""

import time

import torch
from _harness import open_store, parse_study_arguments, release_device

from deephedging import (
    CVaR,
    FeedForwardPolicy,
    HestonSimulator,
    MonteCarloPricer,
    NoCost,
    NoiseSpec,
    ProportionalCost,
    TrainConfig,
    UpAndOutCall,
    delta_hedge_positions,
    hedge_pnl,
    pnl_from_positions,
    pnl_summary,
    train,
)
from deephedging.experiment import ExperimentRecord, append_record
from deephedging.features import DefaultFeatures, FeatureMap, RunningMaxFeatures
from deephedging.frictions.base import CostModel
from deephedging.market import CudaHestonSimulator
from deephedging.training.trainer import TrainResult

MATURITY = 0.25
N_STEPS = 30
HESTON_PARAMETERS = (0.04, 1.5, 0.04, 0.5, -0.7)
STRIKE = 100.0
BARRIER = 120.0
ALPHA = 0.95
RESULTS = "experiments/barrier_hedging.jsonl"
EVAL_SEED = 991
PREMIUM_SEED = 7


def make_simulator(device: str, fused: bool) -> HestonSimulator | CudaHestonSimulator:
    """Builds the stochastic volatility market.

    Args:
        device: Device for path generation.
        fused: Whether to sample with the fused CUDA kernel, which
            generates two orders of magnitude faster but draws from its
            own Philox streams, so runs are reproducible against fused
            runs rather than against the eager stores.

    Returns:
        The Heston simulator all arms share.
    """
    v0, kappa, theta, xi, rho = HESTON_PARAMETERS
    if fused:
        return CudaHestonSimulator(
            s0=100.0,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            maturity=MATURITY,
            n_steps=N_STEPS,
        )
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


def make_arm(name: str, device: str) -> tuple[FeedForwardPolicy, FeatureMap]:
    """Builds one deep arm by name.

    Args:
        name: Either ``spot``, observing spot and time only, or
            ``barrier``, adding the running maximum.
        device: Device for the policy.

    Returns:
        The policy and the feature map of the arm.

    Raises:
        ValueError: If the arm name is unknown.
    """
    if name == "spot":
        return (
            FeedForwardPolicy(n_features=3, hidden_sizes=(64, 64)).to(device),
            DefaultFeatures(),
        )
    if name == "barrier":
        return (
            FeedForwardPolicy(n_features=4, hidden_sizes=(64, 64)).to(device),
            RunningMaxFeatures(),
        )
    raise ValueError(f"unknown arm {name}")


def cost_model(rate: float) -> CostModel:
    """Builds the cost model for a rate.

    Args:
        rate: Proportional cost rate; zero means frictionless.

    Returns:
        The cost model.
    """
    return NoCost() if rate == 0.0 else ProportionalCost(rate=rate)


def main() -> None:
    """Runs the barrier hedging grid and prints the comparison table."""
    arguments = parse_study_arguments(fused_option=True)

    iterations = 30 if arguments.smoke else 4000
    batch_paths = 1024 if arguments.smoke else 65_536
    eval_paths = 4096 if arguments.smoke else 200_000
    premium_paths = 65_536 if arguments.smoke else 1_000_000
    seeds = (1,) if arguments.smoke else (1, 2, 3, 4, 5)
    cost_rates = (0.0,) if arguments.smoke else (0.0, 1e-3)

    results_path, completed = open_store(RESULTS, arguments.smoke)
    simulator = make_simulator(arguments.device, arguments.fused)
    payoff = UpAndOutCall(strike=STRIKE, barrier=BARRIER)
    estimate = MonteCarloPricer(n_paths=premium_paths, seed=PREMIUM_SEED).price(payoff, simulator)
    premium = estimate.value
    print(f"barrier premium {premium:.4f} (se {estimate.standard_error:.4f})")

    _, _, theta, _, _ = HESTON_PARAMETERS
    long_run_vol = theta**0.5
    eval_state = simulator.simulate(eval_paths, noise=NoiseSpec(seed=EVAL_SEED))
    deltas = delta_hedge_positions(eval_state.spot, STRIKE, long_run_vol, MATURITY)

    print("vanilla delta hedge on common evaluation paths:")
    for cost_rate in cost_rates:
        delta_pnl = pnl_from_positions(
            eval_state, deltas, payoff, cost_model(cost_rate), premium=premium
        )
        summary = pnl_summary(delta_pnl)
        print(f"  cost {cost_rate:6.4f}  {summary}")
        record_name = f"barrier/delta/cost{cost_rate}"
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
                        "barrier": BARRIER,
                        "eval_seed": EVAL_SEED,
                        **summary,
                    },
                    result=TrainResult(),
                    duration_seconds=0.0,
                ),
            )

    for cost_rate in cost_rates:
        for arm in ("spot", "barrier"):
            for seed in seeds:
                run_name = f"barrier/{arm}/cost{cost_rate}/seed{seed}"
                if run_name in completed:
                    print(f"skip {run_name} (already recorded)")
                    continue
                torch.manual_seed(seed)
                policy, feature_map = make_arm(arm, arguments.device)
                config = TrainConfig(
                    n_iterations=iterations,
                    batch_paths=batch_paths,
                    lr=1e-3,
                    seed=seed,
                    checkpoint_steps=(arguments.device == "cuda"),
                    graph_episode=(arguments.device == "cuda"),
                    graph_generate=(arguments.fused and arguments.device == "cuda"),
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
                            "barrier": BARRIER,
                            "alpha": ALPHA,
                            "eval_seed": EVAL_SEED,
                            "heston": HESTON_PARAMETERS,
                            **summary,
                        },
                        result=result,
                        duration_seconds=duration,
                    ),
                )
                print(
                    f"{arm:7s} cost {cost_rate:6.4f} seed {seed}  "
                    f"es95 {summary['es_95']:7.4f}  es99 {summary['es_99']:7.4f}  "
                    f"mean {summary['mean']:7.4f}  {duration:6.1f}s"
                )
                del policy, result
                release_device(arguments.device)


if __name__ == "__main__":
    main()
