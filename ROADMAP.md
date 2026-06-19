# Roadmap

This document records where the framework stands, the principles that
constrain how it grows, and the milestones that take it from a strong
research codebase to a state-of-the-art deep-hedging platform. It is a
living plan: each entry carries a status and an acceptance criterion so
progress is measurable rather than aspirational.

Status legend: `[done]` shipped and tested, `[wip]` in progress,
`[next]` scheduled for the current horizon, `[planned]` accepted but
unscheduled, `[research]` requires a feasibility study before commitment.

## Vision

Hedge any liability under any market model and any cost structure by
letting stochastic gradient descent discover the policy a realistic
simulated market rewards, at a throughput and memory efficiency that
makes million-path batches and fifty-dimensional pricing routine. Every
result regenerates from a committed provenance record, and every design
decision is documented by the failure mode it prevents.

## Current state

The framework already covers the full deep-hedging loop end to end.

- Market: GBM, Heston, Merton jumps, correlated multi-asset GBM, Dupire
  local volatility, exponentially tilted importance sampling, and a
  tradable Heston variance swap. Fused Philox CUDA kernels back GBM and
  Heston with bitwise replay and register-resident path folds.
- Objectives: Rockafellar-Uryasev CVaR with a learned threshold and
  entropic risk, both importance-weight aware.
- Policies: feedforward, recurrent (GRU), and a no-transaction-band
  network anchored on the analytic delta.
- Instruments: European call and put, arithmetic and geometric basket
  calls, an up-and-out barrier call, and a single-asset adapter.
- Frictions: proportional and per-asset proportional costs.
- Training systems: whole-episode CUDA graph capture, in-graph batch
  generation, a noise-regenerative backward, gradient checkpointing, and
  bfloat16 autocast, all addressable from one config.
- Pricing and PDE: closed-form Black-Scholes and Merton references, a
  Cox-Ross-Rubinstein tree and Longstaff-Schwartz Monte Carlo for the
  American put, a Monte Carlo pricer with fold fast paths, and a deep
  BSDE solver for semilinear parabolic PDEs to fifty dimensions.
- Calibration: Heston characteristic function with COS pricing, implied
  volatility inversion, surface calibration, and a Dupire local-vol
  surface.
- Evidence: seven committed studies with append-only provenance records,
  a unit and golden test suite pinning statistical relationships and
  design invariants, a throughput benchmark, and a regression gate.

## Guiding principles

These hold for every item below and gate every pull request.

1. Reproducibility is non-negotiable. New stochastic components address
   noise through `NoiseSpec` so any batch replays bitwise and survives
   the regenerative backward.
2. Provenance travels with results. Every study appends a record
   carrying commit, library versions, device, and configuration.
3. Performance is measured, not asserted. Hot paths are profiled before
   they are optimised, and throughput changes pass the regression gate.
4. Scope fences are explicit. A component documents the model class it
   serves and refuses inputs outside it rather than returning a silently
   wrong number.
5. Documentation explains the failure mode, not the mechanics. Docstrings
   say what breaks without the decision they describe.
6. The type checker and linter are part of the build. Strict pyright and
   ruff stay green.

## Horizon 1: depth and hardening (current)

Strengthen the existing pillars and close the highest-leverage gaps that
need no new subsystem.

### Risk objectives

- `[next]` Spectral (distortion) risk measure with a tunable risk
  spectrum that weights every quantile, recovering CVaR as the indicator
  spectrum. Acceptance: empirical estimator matches CVaR within Monte
  Carlo error at the indicator spectrum; coherence (monotonicity,
  positive homogeneity) pinned by test.
- `[next]` Mean-variance and mean-semivariance objectives for the
  classical baseline the objective study lacks. Acceptance: reduces to
  the sample mean as the aversion goes to zero; downside variant ignores
  gains.
- `[planned]` Robust and worst-case objectives over a parameter
  ambiguity set for model-uncertainty hedging.

### Training systems

- `[next]` Optional learning-rate schedule (cosine and linear warmup or
  decay) wired through the config without disturbing graph capture.
  Acceptance: schedule applies eagerly outside the captured region;
  convergence parity test against the constant-rate baseline.
- `[next]` Optional global gradient-norm clipping across policy and risk
  parameters, the stabiliser the high-variance tail objectives want.
  Acceptance: clipped run completes with bounded gradient norm; default
  off preserves current behaviour.
- `[planned]` Out-of-sample evaluation hook running on a held-out seed
  every N iterations to surface generalisation, not just training loss.
- `[planned]` Distributed data-parallel wrapper. The design already
  synchronises risk parameters and addresses per-rank streams; this adds
  the integration and a two-rank parity test.
- `[research]` `torch.compile` of the full episode, contingent on graph
  breaks from the feature-map indirection being resolvable.

### Evaluation

- `[planned]` Greeks by automatic differentiation through the pricer and
  the learned policy: delta, gamma, vega, theta, and rho.
- `[planned]` Profit-and-loss attribution decomposing terminal PnL into
  delta, gamma, vega, and carry contributions.
- `[planned]` Bootstrap confidence intervals and paired significance
  tests for metric differences across seeds, so reported improvements
  carry error bars.

### Project infrastructure

- `[next]` Changelog following Keep a Changelog, driven by the
  Conventional Commits history.
- `[next]` Pre-commit hooks running ruff and pyright so failures surface
  before CI.
- `[next]` Task runner exposing the quality gates and study entry points
  as named targets.
- `[next]` Coverage measurement in CI with a reported figure.
- `[next]` Issue and pull-request templates and a dependabot policy.
- `[next]` Citation metadata.
- `[planned]` Reproducible development container.
- `[planned]` Notebook execution check in CI so the walkthrough cannot
  rot.

## Horizon 2: breadth of models and instruments

Widen the market and product coverage to the families practitioners
actually face.

### Market models

- `[planned]` Quadratic-exponential Heston scheme (Andersen) to remove
  the full-truncation Euler bias near the Feller boundary. Acceptance:
  lower discretisation bias than Euler at matched step count, pinned
  against the characteristic-function price.
- `[planned]` Bates model: Heston dynamics with Merton jumps.
- `[planned]` SABR with an arbitrage-aware sampling scheme.
- `[research]` Rough volatility (rough Heston, rough Bergomi) via a
  hybrid scheme for the fractional kernel.
- `[research]` Local-stochastic volatility coupling a calibrated local
  surface to a stochastic variance driver.
- `[research]` Pure-jump Levy models (variance gamma, CGMY) and
  regime-switching dynamics.
- `[research]` Stochastic short rates (Hull-White, G2++) for rate-
  sensitive products.

### CUDA coverage

- `[planned]` Fused Merton kernel: the eager jump sampler is the next
  throughput gap after GBM and Heston.
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
- `[planned]` Spread and rainbow multi-asset payoffs beyond the single-
  asset adapter.
- `[research]` Autocallable notes and cliquets with periodic resets and
  contingent coupons.

### Frictions

- `[planned]` Fixed-plus-proportional and per-trade fee models.
- `[planned]` Square-root market impact (Almgren-Chriss) and a
  bid-ask-spread cost with directional execution prices.
- `[research]` Inventory-dependent and funding or financing costs for
  multi-period carry.

## Horizon 3: pricing, PDE, and calibration depth

Turn the pricing and calibration corners into first-class subsystems.

### Pricing and PDE

- `[planned]` Longstaff-Schwartz Monte Carlo generalised beyond the put
  to American and Bermudan calls, baskets, and configurable bases.
- `[planned]` Finite-difference PDE solver for low-dimensional American
  and barrier problems as a noise-free cross-check on Monte Carlo.
- `[planned]` Put pricing and general European payoffs in the COS
  pricer, extended off the zero-rate assumption.
- `[research]` Reflected BSDE for optimal stopping and American pricing.
- `[research]` Fully nonlinear and second-order BSDE for HJB problems,
  uncertain volatility, and gamma constraints, the scope the current
  semilinear solver explicitly fences out.
- `[research]` Jump-driven BSDE for Levy forward dynamics.

### Calibration

- `[planned]` Non-zero rates and dividend yields across the
  characteristic-function and inversion stack.
- `[planned]` SVI and SSVI surface parametrisation with no-butterfly-
  arbitrage constraints, replacing wing clamping with a principled fit.
- `[planned]` Global optimisers (differential evolution, CMA-ES) for the
  non-convex calibration surface, with the Adam local polish retained.
- `[planned]` SABR calibration via the Hagan expansion.
- `[research]` Joint and path-dependent calibration to American, barrier,
  and variance-swap quotes.
- `[research]` Market-data loaders and a snapshot format so calibration
  runs against real surfaces.

## Horizon 4: policies and learning frontier

Push the policy class and the learning signal past the Markovian
feedforward baseline.

- `[planned]` Permutation-invariant deep-set policy for multi-asset
  books that scales without relearning asset orderings.
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

## Horizon 5: platform and MLOps

Make large-scale, long-running research dependable.

- `[planned]` Hosted API documentation generated from the docstrings,
  with architecture notes and decision records.
- `[planned]` Per-step training telemetry (loss, gradient norm, learning
  rate) and intermediate checkpoints in the experiment record.
- `[planned]` Query and aggregation layer over the provenance store with
  optional database backend for large sweeps.
- `[planned]` Optional experiment-tracker integration behind a thin,
  dependency-light callback.
- `[planned]` Benchmark history and trend tracking beyond the single
  committed baseline.
- `[research]` Multi-node orchestration for distributed sweeps.

## Cross-cutting commitments

- Testing: every behavioural change ships with a test; statistical
  relationships and design invariants are pinned rather than bitwise
  outputs; golden references guard the closed forms.
- Performance: the regression gate guards throughput; memory levers stay
  composable and documented by the regime where each pays off.
- Reproducibility: noise stays addressable, provenance stays append-only,
  and the regenerative backward stays bitwise with the forward.
- Numerical integrity: scope fences are enforced in code, and scheme bias
  is measured against an analytic or characteristic-function reference
  wherever one exists.

## How to propose a change

Open an issue describing the gap and the model class it serves, then a
pull request that follows `CONTRIBUTING.md`: one logical change per
commit, Conventional Commits, tests alongside behaviour, and the quality
gates green. Moving an item to `[done]` requires its acceptance criterion
to be demonstrable from the committed tests, benchmarks, or studies.
