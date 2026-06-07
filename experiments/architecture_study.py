"""Hedging policy architectures at a matched parameter budget.

Three inductive biases compete on the same market, liability, and
budget. The feedforward policy learns the position surface freely, the
recurrent policy carries hidden state it does not need under Markovian
dynamics, and the no-transaction-band policy hard-codes the
band-around-delta structure that proportional costs make optimal and
learns only the widths. Parameter counts are matched within two
percent, so capacity differences cannot masquerade as inductive-bias
differences, and the recurrent arm trains longer because
backpropagation through the hidden chain converges more slowly. All
arms score on common evaluation paths against one premium, and every
run appends a full-provenance record so the table regenerates from the
store alone.

    uv run python experiments/architecture_study.py [--smoke]
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
    NoTransactionBandPolicy,
    ProportionalCost,
    RecurrentPolicy,
    TrainConfig,
    bs_call_price,
    hedge_pnl,
    pnl_summary,
    train,
)
from deephedging.experiment import ExperimentRecord, append_record
from deephedging.frictions.base import CostModel
from deephedging.policies.base import HedgePolicy

SIGMA = 0.2
MATURITY = 30 / 365
N_STEPS = 30
STRIKE = 100.0
ALPHA = 0.95
RESULTS = "experiments/architecture_study.jsonl"
EVAL_SEED = 991


def make_arm(name: str, device: str) -> HedgePolicy:
    """Builds one architecture arm by name.

    Args:
        name: One of ``ffn``, ``recurrent``, or ``band``.
        device: Device for the policy.

    Returns:
        The policy of the arm.

    Raises:
        ValueError: If the arm name is unknown.
    """
    if name == "ffn":
        return FeedForwardPolicy(n_features=3, hidden_sizes=(64, 64)).to(device)
    if name == "recurrent":
        return RecurrentPolicy(n_features=3, hidden_size=36).to(device)
    if name == "band":
        return NoTransactionBandPolicy(sigma=SIGMA, maturity=MATURITY, hidden_sizes=(64, 64)).to(
            device
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
    """Runs the architecture grid and prints the comparison table."""
    arguments = parse_study_arguments()

    base_iterations = 30 if arguments.smoke else 3000
    recurrent_iterations = 30 if arguments.smoke else 5000
    batch_paths = 1024 if arguments.smoke else 65_536
    eval_paths = 4096 if arguments.smoke else 200_000
    seeds = (1,) if arguments.smoke else (1, 2, 3, 4, 5)
    cost_rates = (0.0,) if arguments.smoke else (0.0, 1e-3)

    results_path, completed = open_store(RESULTS, arguments.smoke)
    simulator = GBMSimulator(
        s0=100.0, sigma=SIGMA, maturity=MATURITY, n_steps=N_STEPS, device=arguments.device
    )
    payoff = EuropeanCall(strike=STRIKE)
    premium = float(bs_call_price(100.0, STRIKE, SIGMA, MATURITY))
    eval_state = simulator.simulate(eval_paths, noise=NoiseSpec(seed=EVAL_SEED))

    for cost_rate in cost_rates:
        for arm in ("ffn", "recurrent", "band"):
            for seed in seeds:
                run_name = f"architecture/{arm}/cost{cost_rate}/seed{seed}"
                if run_name in completed:
                    print(f"skip {run_name} (already recorded)")
                    continue
                torch.manual_seed(seed)
                policy = make_arm(arm, arguments.device)
                parameters = sum(p.numel() for p in policy.parameters())
                config = TrainConfig(
                    n_iterations=(recurrent_iterations if arm == "recurrent" else base_iterations),
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
                    CVaR(alpha=ALPHA),
                    config,
                    premium=premium,
                )
                duration = time.perf_counter() - started
                with torch.no_grad():
                    pnl = hedge_pnl(
                        eval_state, policy, payoff, cost_model(cost_rate), premium=premium
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
                            "parameters": parameters,
                            "alpha": ALPHA,
                            "premium": premium,
                            "eval_seed": EVAL_SEED,
                            **summary,
                        },
                        result=result,
                        duration_seconds=duration,
                    ),
                )
                print(
                    f"{arm:9s} cost {cost_rate:6.4f} seed {seed}  "
                    f"params {parameters:5d}  es95 {summary['es_95']:7.4f}  "
                    f"es99 {summary['es_99']:7.4f}  {duration:6.1f}s"
                )
                del policy, result
                release_device(arguments.device)


if __name__ == "__main__":
    main()
