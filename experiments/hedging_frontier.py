"""Deep hedging against the analytic delta hedge across frictions.

The canonical deep hedging experiment. In the frictionless limit the
trained policy should recover the Black-Scholes delta, and under
transaction costs it should beat the delta hedge in expected shortfall
by learning to trade less, opening a no-trade band around the model
hedge. The grid sweeps cost levels and risk aversions for the deep
policy and scores the analytic hedge on identical evaluation paths, so
every comparison is paired. Each run appends a full-provenance record,
and the learned positions against the model delta on a spot grid are
stored alongside so the band figure regenerates from the store alone.

    uv run python experiments/hedging_frontier.py [--smoke]
"""

import time

import torch
from _harness import open_store, parse_study_arguments, release_device

from deephedging import (
    CVaR,
    EuropeanCall,
    FeedForwardPolicy,
    GBMSimulator,
    NoCost,
    NoiseSpec,
    ProportionalCost,
    TrainConfig,
    bs_call_delta,
    bs_call_price,
    delta_hedge_positions,
    hedge_pnl,
    pnl_from_positions,
    pnl_summary,
    train,
)
from deephedging.experiment import ExperimentRecord, append_record
from deephedging.frictions.base import CostModel
from deephedging.training.trainer import TrainResult

SIGMA = 0.2
MATURITY = 30 / 365
N_STEPS = 30
STRIKE = 100.0
RESULTS = "experiments/hedging_frontier.jsonl"
EVAL_SEED = 991
BAND_SPOTS = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]


def cost_model(rate: float) -> CostModel:
    """Builds the cost model for a rate.

    Args:
        rate: Proportional cost rate; zero means frictionless.

    Returns:
        The cost model.
    """
    return NoCost() if rate == 0.0 else ProportionalCost(rate=rate)


def learned_band(policy: FeedForwardPolicy, tau: float) -> dict[str, list[float]]:
    """Samples the learned position against the model delta on a spot grid.

    The position depends on the held inventory, so the band is sampled
    entering flat and entering at the model delta; the gap between the
    two responses is the learned no-trade behaviour.

    Args:
        policy: The trained policy.
        tau: Normalised time to maturity to probe at.

    Returns:
        Spot grid, model deltas, and the policy's response from a flat
        book and from a delta-held book.
    """
    device = next(policy.parameters()).device
    spots = torch.tensor(BAND_SPOTS, device=device)
    log_moneyness = torch.log(spots / 100.0)
    tau_column = torch.full_like(log_moneyness, tau)
    deltas = bs_call_delta(spots.double(), STRIKE, SIGMA, tau * MATURITY)
    with torch.no_grad():
        flat_features = torch.stack(
            (log_moneyness, tau_column, torch.zeros_like(log_moneyness)), dim=-1
        )
        from_flat, _ = policy(flat_features.float(), None)
        held_features = torch.stack((log_moneyness, tau_column, deltas.float()), dim=-1)
        from_delta, _ = policy(held_features.float(), None)
    return {
        "spots": BAND_SPOTS,
        "model_delta": deltas.cpu().tolist(),
        "policy_from_flat": from_flat.cpu().tolist(),
        "policy_from_delta": from_delta.cpu().tolist(),
    }


def band_probe(policy: FeedForwardPolicy, tau: float) -> dict[str, object]:
    """Sweeps held inventory at the money to expose the no-trade band.

    The no-trade band is the inventory interval the policy leaves in
    place rather than rebalancing toward the model delta. Sweeping the
    position input at fixed spot and time records the policy response
    on a grid; the interval where the response stays near the input is
    the band, and its width across cost levels tests the asymptotic
    one-third power law of Whalley and Wilmott.

    Args:
        policy: The trained policy.
        tau: Normalised time to maturity to probe at.

    Returns:
        The inventory grid, the policy response, and the model delta at
        the money.
    """
    device = next(policy.parameters()).device
    delta_atm = float(bs_call_delta(100.0, STRIKE, SIGMA, tau * MATURITY))
    inventory = torch.linspace(delta_atm - 0.6, delta_atm + 0.6, 49, device=device)
    log_moneyness = torch.zeros_like(inventory)
    tau_column = torch.full_like(inventory, tau)
    with torch.no_grad():
        features = torch.stack((log_moneyness, tau_column, inventory), dim=-1)
        response, _ = policy(features, None)
    return {
        "inventory": inventory.cpu().tolist(),
        "response": response.cpu().tolist(),
        "model_delta": delta_atm,
    }


def main() -> None:
    """Runs the frontier grid and prints the comparison table."""
    arguments = parse_study_arguments()

    iterations = 30 if arguments.smoke else 3000
    batch_paths = 1024 if arguments.smoke else 65_536
    eval_paths = 4096 if arguments.smoke else 200_000
    seeds = (1,) if arguments.smoke else (1, 2, 3, 4, 5)
    cost_rates = (0.0, 1e-3) if arguments.smoke else (0.0, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3)
    alphas = (0.95,) if arguments.smoke else (0.5, 0.9, 0.99)

    results_path, completed = open_store(RESULTS, arguments.smoke)
    premium = float(bs_call_price(100.0, STRIKE, SIGMA, MATURITY))
    simulator = GBMSimulator(
        s0=100.0, sigma=SIGMA, maturity=MATURITY, n_steps=N_STEPS, device=arguments.device
    )
    payoff = EuropeanCall(strike=STRIKE)
    eval_state = simulator.simulate(eval_paths, noise=NoiseSpec(seed=EVAL_SEED))
    deltas = delta_hedge_positions(eval_state.spot, STRIKE, SIGMA, MATURITY)

    print("analytic delta hedge on common evaluation paths:")
    for cost_rate in cost_rates:
        delta_pnl = pnl_from_positions(
            eval_state, deltas, payoff, cost_model(cost_rate), premium=premium
        )
        summary = pnl_summary(delta_pnl)
        print(f"  cost {cost_rate:6.4f}  {summary}")
        record_name = f"frontier/delta/cost{cost_rate}"
        if record_name not in completed:
            append_record(
                results_path,
                ExperimentRecord.from_run(
                    name=record_name,
                    config=TrainConfig(n_iterations=0),
                    setup={
                        "arm": "delta",
                        "cost_rate": cost_rate,
                        "alpha": None,
                        "eval_seed": EVAL_SEED,
                        **summary,
                    },
                    result=TrainResult(),
                    duration_seconds=0.0,
                ),
            )

    for cost_rate in cost_rates:
        for alpha in alphas:
            for seed in seeds:
                run_name = f"frontier/deep/cost{cost_rate}/alpha{alpha}/seed{seed}"
                if run_name in completed:
                    print(f"skip {run_name} (already recorded)")
                    continue
                torch.manual_seed(seed)
                policy = FeedForwardPolicy(hidden_sizes=(64, 64)).to(arguments.device)
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
                    cost_model(cost_rate),
                    CVaR(alpha=alpha),
                    config,
                    premium=premium,
                )
                duration = time.perf_counter() - started
                with torch.no_grad():
                    pnl = hedge_pnl(
                        eval_state, policy, payoff, cost_model(cost_rate), premium=premium
                    )
                summary = pnl_summary(pnl)
                band = learned_band(policy, tau=0.5)
                probe = band_probe(policy, tau=0.5)
                append_record(
                    results_path,
                    ExperimentRecord.from_run(
                        name=run_name,
                        config=config,
                        setup={
                            "arm": "deep",
                            "cost_rate": cost_rate,
                            "alpha": alpha,
                            "eval_seed": EVAL_SEED,
                            "band": band,
                            "band_probe": probe,
                            **summary,
                        },
                        result=result,
                        duration_seconds=duration,
                    ),
                )
                print(
                    f"deep cost {cost_rate:6.4f} alpha {alpha:4.2f} seed {seed}  "
                    f"es95 {summary['es_95']:7.4f}  mean {summary['mean']:7.4f}  "
                    f"{duration:6.1f}s"
                )
                del policy, result
                release_device(arguments.device)


if __name__ == "__main__":
    main()
