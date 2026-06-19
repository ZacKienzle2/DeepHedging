# Roadmap

This document records where the framework stands, the principles that
constrain how it grows, and the milestones that take it from a strong
research codebase to a state-of-the-art deep-hedging platform. It is a
living plan, and each entry carries a status tag and an acceptance
criterion so progress stays measurable rather than aspirational.

Each item carries a status tag. `[done]` means shipped and tested,
`[wip]` means in progress, `[next]` means scheduled for the current
horizon, `[planned]` means accepted but unscheduled, and `[research]`
means a feasibility study comes before any commitment.

## Vision

Hedge any liability under any market model and any cost structure by
letting stochastic gradient descent discover the policy a realistic
simulated market rewards, at a throughput and memory efficiency that
makes million-path batches and fifty-dimensional pricing routine. Every
result regenerates from a committed provenance record, and every design
decision is documented by the failure mode it prevents.

## Current state

The framework already covers the full deep-hedging loop end to end.

- The market layer simulates GBM, Heston, Merton jumps, correlated
  multi-asset GBM, Dupire local volatility, exponentially tilted
  importance sampling, and a tradable Heston variance swap. Fused Philox
  CUDA kernels back GBM and Heston with bitwise replay and
  register-resident path folds.
- The objectives are Rockafellar-Uryasev CVaR with a learned threshold
  and entropic risk, both importance-weight aware.
- The policies are feedforward, recurrent (GRU), and a
  no-transaction-band network anchored on the analytic delta.
- The instruments are a European call and put, arithmetic and geometric
  basket calls, an up-and-out barrier call, and a single-asset adapter.
- The frictions are proportional and per-asset proportional costs.
- The training systems provide whole-episode CUDA graph capture, in-graph
  batch generation, a noise-regenerative backward, gradient
  checkpointing, and bfloat16 autocast, all addressable from one config.
- The pricing and PDE layer holds closed-form Black-Scholes and Merton
  references, a Cox-Ross-Rubinstein tree and Longstaff-Schwartz Monte
  Carlo for the American put, a Monte Carlo pricer with fold fast paths,
  and a deep BSDE solver for semilinear parabolic PDEs to fifty
  dimensions.
- The calibration layer fits the Heston characteristic function through
  COS pricing, inverts implied volatility, calibrates the surface, and
  builds a Dupire local-vol surface.
- The evidence is seven committed studies with append-only provenance
  records, a unit and golden test suite that pins statistical
  relationships and design invariants, a throughput benchmark, and a
  regression gate.

## Guiding principles

These hold for every item below and gate every pull request.

1. Reproducibility is non-negotiable. New stochastic components address
   noise through `NoiseSpec` so any batch replays bitwise and survives
   the regenerative backward.
2. Provenance travels with results. Every study appends a record carrying
   the commit, library versions, device, and configuration.
3. Performance is measured, not asserted. Hot paths are profiled before
   they are optimised, and throughput changes pass the regression gate.
4. Scope fences stay explicit. A component documents the model class it
   serves and refuses inputs outside it rather than returning a silently
   wrong number.
5. Documentation explains the failure mode, not the mechanics. A
   docstring says what breaks without the decision it describes.
6. The type checker and linter are part of the build. Strict pyright and
   ruff stay green.

## Horizon 1. Depth and hardening (current)

Strengthen the existing pillars and close the highest-leverage gaps that
need no new subsystem.

### Risk objectives

- `[done]` Spectral (distortion) risk measure with a tunable spectrum
  that weights every quantile and recovers CVaR at the indicator
  spectrum. The empirical estimator matches CVaR within Monte Carlo error
  at that spectrum, and coherence is pinned by test.
- `[done]` Mean-variance and mean-semivariance objectives for the
  classical baseline the objective study lacked. They reduce to the
  sample mean as the aversion vanishes, and the downside variant ignores
  gains.
- `[planned]` Robust and worst-case objectives over a parameter ambiguity
  set for model-uncertainty hedging.

### Training systems

- `[done]` Optional learning-rate schedule, cosine or linear, wired
  through the config without disturbing graph capture. It applies eagerly
  outside the captured region, and a convergence-parity test guards it
  against the constant-rate baseline.
- `[done]` Optional global gradient-norm clip across policy and risk
  parameters, the stabiliser the high-variance tail objectives want. A
  clipped run completes with bounded gradient norm, and the default off
  preserves prior behaviour.
- `[planned]` Out-of-sample evaluation hook running on a held-out seed
  every few iterations to surface generalisation rather than only
  training loss.
- `[planned]` Distributed data-parallel wrapper. The design already
  synchronises risk parameters and addresses per-rank streams, so this
  adds the integration and a two-rank parity test.
- `[research]` Full-episode `torch.compile`, contingent on the
  feature-map graph breaks being resolvable.

### Evaluation

- `[done]` Greeks by automatic differentiation through the pricing map,
  returning delta, gamma, vega, theta, and rho elementwise over a
  broadcast surface and validated against the closed forms.
- `[done]` Bootstrap confidence intervals and a paired
  common-random-number comparison, so reported improvements carry error
  bars rather than a bare point estimate.
- `[planned]` Profit-and-loss attribution decomposing terminal PnL into
  delta, gamma, vega, and carry contributions.

### Project infrastructure

- `[done]` Changelog following Keep a Changelog, driven by the
  Conventional Commits history.
- `[done]` Pre-commit hooks running ruff and pyright so failures surface
  before CI.
- `[done]` Task runner exposing the quality gates and study entry points
  as named targets.
- `[done]` Coverage measurement in CI with a reported figure.
- `[done]` Issue and pull-request templates and a dependency-update
  policy.
- `[done]` Reproducible development container and editor configuration.
- `[planned]` Notebook execution check in CI so the walkthrough cannot
  rot.
- `[planned]` Citation metadata once the licence position allows it.

## Horizon 2. Breadth of models and instruments

Widen the market and product coverage to the families practitioners
actually face.

### Market models

- `[planned]` Quadratic-exponential Heston scheme (Andersen) to remove
  the full-truncation Euler bias near the Feller boundary. It must show
  lower discretisation bias than Euler at matched step count, pinned
  against the characteristic-function price.
- `[planned]` Bates model, which adds Merton jumps to Heston dynamics.
- `[planned]` SABR with an arbitrage-aware sampling scheme.
- `[research]` Rough volatility (rough Heston, rough Bergomi) through a
  hybrid scheme for the fractional kernel.
- `[research]` Local-stochastic volatility coupling a calibrated local
  surface to a stochastic variance driver.
- `[research]` Pure-jump Levy models (variance gamma, CGMY) and
  regime-switching dynamics.
- `[research]` Stochastic short rates (Hull-White, G2++) for
  rate-sensitive products.

### CUDA coverage

- `[planned]` Fused Merton kernel, the next throughput gap after GBM and
  Heston now that the eager jump sampler is the bottleneck.
- `[planned]` Fused correlated multi-asset kernel with a register-held
  Cholesky factor.
- `[research]` Quasi-Monte Carlo path kernels (scrambled Sobol) for the
  smooth-payoff regime, with the antithetic and control-variate
  invariants re-measured under the new sampler.

### Instruments

- `[planned]` Asian (arithmetic and geometric average) and lookback
  payoffs, reusing the streaming accumulators.
- `[planned]` Down-and-out, knock-in, and double-barrier options with a
  continuous-monitoring Brownian-bridge correction.
- `[planned]` Spread and rainbow multi-asset payoffs beyond the
  single-asset adapter.
- `[research]` Autocallable notes and cliquets with periodic resets and
  contingent coupons.

### Frictions

- `[planned]` Fixed-plus-proportional and per-trade fee models.
- `[planned]` Square-root market impact (Almgren-Chriss) and a
  bid-ask-spread cost with directional execution prices.
- `[research]` Inventory-dependent and funding or financing costs for
  multi-period carry.

## Horizon 3. Pricing, PDE, and calibration depth

Turn the pricing and calibration corners into first-class subsystems.

### Pricing and PDE

- `[planned]` Longstaff-Schwartz Monte Carlo generalised beyond the put
  to American and Bermudan calls, baskets, and configurable bases.
- `[planned]` Finite-difference PDE solver for low-dimensional American
  and barrier problems as a noise-free cross-check on Monte Carlo.
- `[planned]` Put pricing and general European payoffs in the COS pricer,
  extended off the zero-rate assumption.
- `[research]` Reflected BSDE for optimal stopping and American pricing.
- `[research]` Fully nonlinear and second-order BSDE for HJB problems,
  uncertain volatility, and gamma constraints, the scope the current
  semilinear solver fences out.
- `[research]` Jump-driven BSDE for Levy forward dynamics.

### Calibration

- `[planned]` Non-zero rates and dividend yields across the
  characteristic-function and inversion stack.
- `[planned]` SVI and SSVI surface parametrisation with
  no-butterfly-arbitrage constraints, replacing wing clamping with a
  principled fit.
- `[planned]` Global optimisers (differential evolution, CMA-ES) for the
  non-convex calibration surface, with the Adam local polish retained.
- `[planned]` SABR calibration through the Hagan expansion.
- `[research]` Joint and path-dependent calibration to American, barrier,
  and variance-swap quotes.
- `[research]` Market-data loaders and a snapshot format so calibration
  runs against real surfaces.

## Horizon 4. Policies and learning frontier

Push the policy class and the learning signal past the Markovian
feedforward baseline.

- `[planned]` Permutation-invariant deep-set policy for multi-asset books
  that scales without relearning asset orderings.
- `[research]` Attention or transformer policy over path history for
  strongly path-dependent liabilities.
- `[research]` Path-signature features capturing quadratic variation and
  higher increments for faster regime adaptation.
- `[research]` Distributional policies for position uncertainty under
  model risk.
- `[planned]` Curriculum schedules over volatility and moneyness to speed
  convergence on exotic payoffs.
- `[planned]` Hyperparameter search harness recording every trial to the
  provenance store.

## Horizon 5. Platform and MLOps

Make large-scale, long-running research dependable.

- `[planned]` Hosted API documentation generated from the docstrings,
  with architecture notes and decision records.
- `[planned]` Per-step training telemetry such as loss, gradient norm,
  and learning rate, with intermediate checkpoints in the experiment
  record.
- `[planned]` Query and aggregation layer over the provenance store with
  an optional database backend for large sweeps.
- `[planned]` Optional experiment-tracker integration behind a thin,
  dependency-light callback.
- `[planned]` Benchmark history and trend tracking beyond the single
  committed baseline.
- `[research]` Multi-node orchestration for distributed sweeps.

## Cross-cutting commitments

- Testing. Every behavioural change ships with a test, statistical
  relationships and design invariants are pinned rather than bitwise
  outputs, and golden references guard the closed forms.
- Performance. The regression gate guards throughput, and the memory
  levers stay composable and documented by the regime where each one pays
  off.
- Reproducibility. Noise stays addressable, provenance stays
  append-only, and the regenerative backward stays bitwise with the
  forward.
- Numerical integrity. Scope fences are enforced in code, and scheme bias
  is measured against an analytic or characteristic-function reference
  wherever one exists.

## How to propose a change

Open an issue describing the gap and the model class it serves, then a
pull request that follows `CONTRIBUTING.md` with one logical change per
commit, Conventional Commits, tests alongside behaviour, and the quality
gates green. Moving an item to `[done]` requires its acceptance criterion
to be demonstrable from the committed tests, benchmarks, or studies.
