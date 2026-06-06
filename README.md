# DeepHedging

Deep hedging research framework. Neural hedging policies are trained by
stochastic gradient descent on convex risk measures of terminal PnL over
simulated market paths under realistic frictions, with a deep BSDE solver
for high-dimensional semilinear pricing PDEs alongside.

## Overview

Classical delta hedging assumes frictionless complete markets and derives
the hedge analytically. This framework drops both assumptions. Transaction
costs, discrete rebalancing, and stochastic volatility enter the
simulator, and the optimiser finds the policy the simulated market
rewards. Paths are generated on the fly each batch from addressable noise
streams, so data is unbounded, nothing overfits a stored dataset, and any
single batch replays exactly.

`Main.ipynb` is the executable walkthrough. It trains a CVaR hedger under
proportional costs, compares it against the no-hedge and Black-Scholes
delta baselines on common out-of-sample paths, and demonstrates
variance-aware Heston hedging, barrier liabilities, and deep BSDE pricing
against closed forms.

## Layout

```
src/deephedging/
|-- market/       GBM and Heston simulators, MarketState, NoiseSpec
|-- instruments/  European and barrier payoffs
|-- frictions/    Transaction cost models
|-- risk/         Rockafellar-Uryasev CVaR, entropic risk
|-- policies/     Feedforward and recurrent hedging networks
|-- features/     Observation construction (features.py)
|-- training/     Episode engine and training loop
|-- evaluation/   Black-Scholes closed forms, baselines, risk metrics
`-- bsde/         Deep BSDE solver for semilinear pricing PDEs
tests/            Unit suite plus golden tests against closed forms
Main.ipynb        End-to-end walkthrough
```

## Getting started

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). A CUDA device
is used automatically when available; everything also runs on CPU.

```bash
git clone https://github.com/ZacKienzle2/DeepHedging.git
cd DeepHedging
uv sync --extra dev
uv sync --extra notebook   # additionally, for Main.ipynb
```

## Quality gates

```bash
uv run ruff check .
uv run pyright
uv run pytest -q -m "not slow"   # fast suite
uv run pytest -q                 # full suite including training goldens
```

Tests pin statistical relationships and closed-form references rather
than bitwise values. Golden tests cover the Black-Scholes surface, the
Heston degenerate limits, and deep BSDE prices up to five dimensions.

## Design notes

Key decisions and the failure modes they prevent are documented in the
module docstrings, among them log-space path evolution, the learned CVaR
threshold with quantile warm start, addressable noise streams, and the
semilinear scope fence on the BSDE solver. `Main.ipynb` closes with a
summary.

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.

## License

Proprietary. All rights reserved. See [LICENSE](LICENSE).
