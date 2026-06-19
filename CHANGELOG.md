# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ROADMAP.md` recording the current state, guiding principles, and the
  phased milestones toward a state-of-the-art deep-hedging platform.
- Spectral (distortion) risk measure with an exponential spectrum and a
  CVaR spectrum, weighting every quantile rather than only the tail.
- Mean-variance and mean-semivariance objectives as the classical
  baseline for the risk-objective studies.
- Optional learning-rate schedule (cosine, linear) and global gradient-
  norm clipping in the training loop, both compatible with episode capture.
- Project infrastructure: changelog, pre-commit hooks, contributor task
  runner, issue and pull-request templates, dependency-update policy,
  editor configuration, and a reproducible development container.
- Coverage measurement in continuous integration.

## [0.1.0]

Initial research framework.

### Added

- Market simulators: GBM, Heston, Merton jumps, correlated multi-asset
  GBM, Dupire local volatility, exponentially tilted importance sampling,
  and a tradable Heston variance swap, with fused Philox CUDA kernels for
  GBM and Heston offering bitwise replay and register-resident path folds.
- Risk objectives: Rockafellar-Uryasev CVaR with a learned threshold and
  entropic risk, both importance-weight aware.
- Policies: feedforward, recurrent, and no-transaction-band networks.
- Instruments: European call and put, arithmetic and geometric basket
  calls, an up-and-out barrier call, and a single-asset adapter.
- Frictions: proportional and per-asset proportional transaction costs.
- Training systems: whole-episode CUDA graph capture, in-graph batch
  generation, a noise-regenerative backward, gradient checkpointing, and
  bfloat16 autocast.
- Pricing and PDE: closed-form Black-Scholes and Merton references, a
  binomial tree and Longstaff-Schwartz Monte Carlo for the American put,
  a Monte Carlo pricer with fold fast paths, and a deep BSDE solver for
  semilinear parabolic PDEs to fifty dimensions.
- Calibration: Heston characteristic function with COS pricing, implied
  volatility inversion, surface calibration, and a Dupire local-vol
  surface.
- Evidence: seven committed studies with append-only provenance records,
  a unit and golden test suite, throughput benchmarks, and a regression
  gate.

[Unreleased]: https://github.com/ZacKienzle2/DeepHedging/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ZacKienzle2/DeepHedging/releases/tag/v0.1.0
