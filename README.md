# DeepHedging

Deep hedging research framework. Neural hedging policies are trained by
stochastic gradient descent on convex risk measures of terminal PnL over
simulated market paths under realistic frictions, with custom CUDA path
kernels, whole-episode graph capture, a noise-regenerative backward, and
a deep BSDE solver for high-dimensional semilinear pricing PDEs.

## Overview

Classical delta hedging assumes frictionless complete markets and derives
the hedge analytically. This framework drops both assumptions. Transaction
costs, discrete rebalancing, stochastic volatility, jumps, and barrier
liabilities enter the simulator, and the optimiser finds the policy the
simulated market rewards. Paths are generated on the fly each batch from
addressable noise streams, so data is unbounded, nothing overfits a
stored dataset, and any single batch replays exactly.

`Main.ipynb` is the executable walkthrough. The `experiments/` directory
holds the full studies; every run appends a JSON line carrying the
commit, library versions, device, seeds, and loss history, so each table
below regenerates from its committed store.

## Headline results

All numbers are repo artifacts produced on an RTX 5080 from the committed
experiment stores and the test suite.

Hedging an at-the-money call daily over thirty days against the
Black-Scholes delta baseline, expected shortfall at the ninety-fifth
percentile on two hundred thousand common evaluation paths, five seeds:

| Proportional cost | Delta hedge | Deep CVaR policy | Improvement |
|---|---|---|---|
| frictionless | 0.83 | 0.85 | parity, recovers the model hedge |
| 10 bp | 1.13 | 1.08 | 5% |
| 20 bp | 1.45 | 1.29 | 11% |
| 40 bp | 2.08 | 1.65 | 21% |

At forty basis points the deep policy also hedges at a 32% lower mean
cost. The stored inventory probes show the learned no-trade band widening
with cost at a fitted exponent of 0.28 to 0.33 at the median objective,
consistent with the Whalley-Wilmott cube-root law; at higher tail
aversion the fitted slope decreases, the expected direction once the
objective weights rare deviations rather than local variance.

Further studies in `experiments/` cover a barrier option with no analytic
hedge where the deep policy cuts the delta baseline's tail by 20-26%, a
tradable variance swap removing a further 11-12% of tail risk that the
spot cannot span, parameter-matched architecture and risk-objective
ablations, and deep BSDE pricing of a geometric basket call within 1.3%
of its closed form up to fifty dimensions.

Systems results are pinned by tests and benchmarks. Fused CUDA Philox
kernels reach several billion GBM paths per second and roughly two
hundred times the eager Heston rate with bitwise replay; whole-episode
graph capture collapses the dispatch-bound training iteration into one
launch; the noise-regenerative backward cuts peak training memory 12.7x
at a quarter-million paths and lifts the feasible batch from a quarter
million to beyond two million paths on a sixteen-gigabyte device.

## Layout

```
src/deephedging/
|-- market/       GBM, Heston, Merton jumps, correlated multi-asset,
|                 local vol, tilted, variance swap; fused CUDA samplers
|-- instruments/  European, barrier, basket payoffs and adapters
|-- frictions/    Proportional and per-asset transaction cost models
|-- risk/         Rockafellar-Uryasev CVaR, entropic risk
|-- policies/     Feedforward, recurrent, no-transaction-band networks
|-- features.py   Observation construction
|-- training/     Episode engine, training loop, graph capture,
|                 regenerative backward
|-- calibration/  Heston characteristic function, COS pricing, surface fit
|-- evaluation/   Closed forms, American exercise, baselines, metrics
|-- bsde/         Deep BSDE solver for semilinear pricing PDEs
|-- pricing.py    Closed-form and Monte Carlo pricers
`-- experiment.py Append-only provenance records
csrc/             Fused Philox path kernels (CUDA)
experiments/      Seven studies with committed result stores
benchmarks/       Throughput measurement and the regression gate
tests/            Unit suite, golden tests, design invariants
Main.ipynb        End-to-end walkthrough
```

## Getting started

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). A CUDA device
is used automatically when available; everything also runs on CPU. The
fused kernels compile on first use, which needs a CUDA toolchain and, on
Windows, an MSVC host compiler.

```bash
git clone https://github.com/ZacKienzle2/DeepHedging.git
cd DeepHedging
uv sync --extra dev
uv sync --extra notebook   # additionally, for Main.ipynb
```

Reproduce a study:

```bash
uv run python experiments/hedging_frontier.py --smoke   # minutes, CPU
uv run python experiments/hedging_frontier.py           # full grid, GPU
uv run python experiments/band_scaling.py               # fit from the store
```

## Quality gates

```bash
uv run ruff check .
uv run pyright
uv run pytest -q -m "not slow and not gpu"   # fast suite
uv run pytest -q -m "not gpu"                # plus training goldens
uv run pytest -q -m gpu                      # kernel and capture parity
```

Tests pin statistical relationships and closed-form references rather
than bitwise values, plus design invariants. Antithetic pairing measured
harmful at convergence, additive control variates measured gradient-inert,
and regenerated paths measured loss-identical to stored ones.

## Design notes

Key decisions and the failure modes they prevent are documented in the
module docstrings, among them log-space path evolution, the learned CVaR
threshold with quantile warm start, addressable noise streams mapped onto
Philox subsequences, the whole-episode capture unit, the seed requirement
on the regenerative backward, and the semilinear scope fence on the BSDE
solver. `Main.ipynb` closes with a summary.

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.

## License

Proprietary. All rights reserved. See [LICENSE](LICENSE).
