# Performance review

This review records what the profilers measured in the hedging engine, which
published work explains each measurement, what was changed as a result and what
the change measured afterwards. Every number was taken on one machine: an NVIDIA
GeForce RTX 5080 with 16 GiB, Windows 11, Python 3.12, PyTorch 2.12.0 built
against CUDA 13.0. The literature was retrieved and its metadata resolved from
the registrars with the `bibliography` package of the thesis repository, so
every reference below carries the DOI its publisher registered.

## Method

Three instruments answered three different questions.

- `torch.profiler` with CPU and CUDA activities attributed device time to
  kernels and counted host launches. It reports where the time goes, so it chose
  the targets.
- pytest-benchmark timed the throughput suite in `tests/benchmarks`, reporting a
  median and an interquartile range over calibrated rounds and comparing a run
  against a saved one. It replaced a timing loop and a JSON baseline gate
  written for this repository.
- `pyperf timeit` compared an implementation against the one it replaced,
  reporting a mean and a standard deviation over worker processes, for functions
  outside the benchmark suite.

A change counted as an improvement only when the interval of the new measurement
cleared the interval of the old one. Georges, Buytaert and Eeckhout (2007) and
Kalibera and Jones (2013) show that single timings mislead, and Mytkowicz et al.
(2009) show that incidental setup such as environment size moves a measurement
by more than many claimed speedups, which is why the spreads are reported with
every number.

The workload for the training numbers is the benchmark configuration: 65536
paths, 30 rebalancing dates, a feedforward policy of two 64-wide hidden layers,
a CVaR objective and proportional costs.

## Diagnosis and remedies

### The eager training step was bound by kernel launches

The profiler counted 1258 kernel launches per eager training iteration. Each
date of the episode issued the policy network and then, separately, the gains,
the trade, the cost and three accumulations into the running PnL, each a kernel
of one or two microseconds on a 65536-element vector, each with its own backward
kernels. Paszke et al. (2019) describe the eager design that makes every
operator a separate dispatch, and Ansel et al. (2024) give the per-operator
overhead of eager execution among the reasons PyTorch 2 adds graph capture and
compilation.

Only the policy call depends on the previous date, since the held position is an
input to the next observation. Gains and costs depend on the whole position path
but on no later decision. Allen and Kennedy (1987) give the rule for this case:
distribute the loop so that statements outside a dependence cycle leave it and
become whole-array operations. The episode now runs the network alone per date,
stacks the positions and settles gains and costs once over the grid through
`pnl_from_positions`, the function the analytic baselines already used. The two
settlement paths are now one.

Launches fell from 1258 to 801 per iteration. The eager CUDA step fell from a
21.0 ms median (interquartile range 3.3 ms) to 14.2 ms (0.65 ms).

### Graph capture left the optimiser outside the graph

Capture used `torch.cuda.make_graphed_callables`, which records the forward and
backward of a callable. The optimiser step, the gradient reset and clipping
stayed eager, and PyTorch 2.12 warned during the backward that the
AccumulateGrad nodes created during capture ran on a different stream from the
incoming gradients, which forces a synchronisation. The warning was being
filtered rather than addressed.

The CUDA graphs notes in the PyTorch documentation give whole-network capture
for this case: warm up on a side stream, capture forward, backward and the
optimiser step together, and use an optimiser whose state lives on the device.
Adam (Kingma and Ba, 2014) now runs as PyTorch's fused implementation with
`capturable` set, the learning rates are device tensors the scheduler fills in
place, clipping is captured with the rest, and the generated variant advances
its Philox offset inside the graph. An iteration is one replay and one loss
copy. The warm-up steps are undone before training so the first replay matches
the first eager iteration, which the parity tests check to 1e-5.

The stream warning disappeared with its filter. The 200-iteration generated
benchmark fell from 3.04 s to 2.67 s.

### Matrix multiplies ran on the SIMT pipeline

Under float32 the profiler showed the network's matrix multiplies as
`cutlass_80_simt_sgemm` and `magma_sgemmEx` kernels, which use the ordinary
floating-point units, and 61 per cent of device time in them. The tensor cores
Jia et al. (2018) characterise stayed idle. Micikevicius et al. (2017) establish
reduced-precision training with full-precision state, and Kalamkar et al. (2019)
show that bfloat16 needs no loss scaling because it keeps the float32 exponent
range.

The engine already offered bfloat16 autocast for the network alone, with
positions, PnL and the risk reduction kept in float32, and its documentation
said the casts cost more than they saved at 64-wide layers. That held for the
eager loop, where each cast is one more launch. Under capture a launch costs
nothing, so autocast was made compatible with capture by disabling its weight
cache, which each per-date autocast region discarded anyway. The profile then
shows `cutlass_80_tensorop_bf16` and `wmma_tensorop` kernels.

The generated benchmark measured 2.67 s in float32, 2.41 s with TF32 matrix
multiplies and 1.80 s under bfloat16 autocast. TF32 is a global PyTorch setting,
`torch.backends.cuda.matmul.fp32_precision = "tf32"`, and bfloat16 is the `amp`
flag. Both stay off by default because they change numerics.

### The bootstrap issued one resample per Python iteration

`bootstrap_metric` and `paired_bootstrap` drew and scored each resample in its
own loop iteration: one index draw, one gather and one metric call, a thousand
times. Efron (1979) defines the method, and `scipy.stats.bootstrap` in the SciPy
library (Virtanen et al., 2020) evaluates a vectorised statistic over a whole
batch of resamples along one axis. `torch.vmap` could not stand in, because
`torch.quantile` has no batching rule and falls back to a loop with a warning.

The resamples are now drawn as one index matrix per chunk and the metric reduces
the last dimension, the same contract as SciPy's vectorised statistic.
`expected_shortfall` reduces that dimension with `torch.quantile(dim=-1)`. The
weighted quantile, which existed twice and read an index back to the host in
both copies, is now one device-side function.

For 1000 resamples of expected shortfall on 20000 paths, pyperf measured 2.09 s
to 451 ms (standard deviation 27 ms) on the CPU and 445 ms to 15.6 ms (0.2 ms)
on CUDA.

### The deep BSDE solver called its network once per date

The solver of Han, Jentzen and E (2018) evaluated the `Z` network inside the
loop over dates. `Z` reads only time and the forward state, and the forward
state never reads `Y`, so neither belongs to the recursion. The same loop
distribution applies: the forward path is one cumulative sum, `Z` is one network
call over every date, and the loop keeps only the elementwise `Y` update.

200 training iterations at 512 paths and 10 dates measured 716 ms (58 ms) before
and 301 ms (19 ms) after.

### Heston calibration used a first-order optimiser on a smooth problem

Calibration minimised a vega-weighted squared price error over five parameters
with Adam for 800 steps, pricing through the COS expansion of Fang and Oosterlee
(2008). The objective is smooth, deterministic and low-dimensional, the setting
limited-memory BFGS (Liu and Nocedal, 1989) was designed for.
`torch.optim.LBFGS` with a strong Wolfe line search reached a loss of 7.8e-18 in
28 evaluations and 0.11 s on the golden test quotes, where Adam took 3.07 s and
stopped at 4.2e-8. The parameter constraints now come from the softplus and tanh
bijectors of `torch.distributions`, and the implied-volatility inversion and the
calibration weights call the shared Black-Scholes price and vega, which now
broadcast over tensor strikes and volatilities, instead of their own copies.

### The Merton series priced each term separately

The Merton closed form summed 40 Poisson-weighted Black-Scholes prices in a
Python loop, and each call checked its maturity on the host. The counts are now
a tensor, the weights come from `torch.distributions.Poisson.log_prob`, and one
broadcast call prices every term. pyperf measured 1.82 ms to 108 us.

### The band network waited on the held position

The no-transaction-band policy read the held position as a network input, which
put its network inside the per-date dependence cycle. Whalley and Wilmott (1997,
eq. 3.10) derive the band edges of the Davis, Panas and Zariphopoulou (1993)
problem, in the small-cost limit, as functions of spot and time alone. Imaki et
al. (2021, eq. 12) build their network the same way and state that it does not
need the current hedge ratio as an input. With the held position removed, one
network call over 30 by 65536 rows returns the band edges of every date, and
only the elementwise clamp stays in the loop. This is the batching of the
non-recurrent work that Appleyard, Kocisky and Blunsom (2016) prescribe. pyperf
times an eager CUDA step of the band policy at 42.8 ms before and 29.6 ms after.
The float32 step is now led by one 7.2 ms weight-gradient GEMM that reduces over
all 1.97 million rows, and the bfloat16 step takes 9.8 ms of device time.

### Pseudo-random paths wasted variance

Glasserman (2003, section 5.5) reports Sobol points ahead of Monte Carlo on
single-asset path problems with 30 dates. His Table 5.6 shows the Brownian
bridge and principal components constructions reducing the error further.
`GBMSimulator` now takes `sampler="sobol"`. It draws a scrambled Sobol set per
call with the linear digit scrambling of his section 5.4 (Owen, 1998) and builds
each path by the Brownian bridge, generalised from his Figure 3.2 to any grid. A
scrambled draw of 65536 by 30 points costs 1.46 ms on the CPU, against 24.5 ms
for `torch.randn` in float64. Over 128 randomisations of 16384 paths and 30
dates, the path count times the replicate variance measured as follows.

| Estimator                      | Pseudo-random | Sobol, bridge |
| ------------------------------ | ------------- | ------------- |
| Mean PnL of the delta hedge    | 0.125         | 0.0036        |
| VaR95 of the delta hedge       | 1.43          | 0.76          |
| Mean PnL of a band policy      | 8.85          | 0.0079        |
| ES95 of a band policy          | 122           | 0.081         |
| Entropic risk of a band policy | 1.45e4        | 1.36e4        |
| Gradient of that entropic risk | 6.32e4        | 5.77e4        |

The entropic objective of an untrained policy is the exponential of an unbounded
loss. That integrand lies outside the bounded-variation class that QMC error
bounds cover, and its variance stayed within 10 per cent. The principal
components construction did worse than the bridge on the clamped hedge at 32
dates, as Glasserman's table finds for barrier options.

### Wing prices and implied volatilities were wrong

The implied volatility solver bracketed with 25 bisection steps, polished with
12 Halley steps, and accepted any result whose price residual was below 1e-6 of
spot. Over maturities from one day to two years, volatilities from 0.1 to 400
per cent and strikes from half to twice spot, it returned NaN for many
out-of-the-money quotes. Where a tiny price met the absolute residual it
returned finite values off by up to 7 per cent. Two defects combined. The call
price `S N(d1) - K N(d2)` cancels for out-of-the-money options. On this build
`torch.special.ndtr` keeps absolute but not relative accuracy for negative
arguments, off by a relative 1e-10 at -5 and 2e-2 at -8 against 40-digit
references. Jaeckel (2015) shows that the Black function, not the solver, sets
the attainable precision of an implied volatility.

Prices now go through his normalised Black function in the four regimes of his
Section 6, with tail probabilities from `erfc` as in his eq. 6.15. The series
coefficients that the paper leaves to its reference code are derived here. The
ratio `Y = Phi / phi` satisfies `Y' = 1 + z Y`, whose recursion gives the Taylor
series of the small-volatility regime to any order. The asymptotic series of his
eq. 6.13 is differenced term by term, with summands of one sign. The solver
follows his Sections 4 and 5: a four-branch initial guess from the rational
cubic of Delbourgo and Gregory (1985), then two third-order Householder steps on
his three-branch objective. The relative error is at most 3.0e-15 on his Figure
7 and 8 domains and on log-moneyness down to -32 with total volatility up to 6.
A third step leaves it unchanged. 4096 quotes take 4.60 ms, against 6.97 ms
before. The accurate price costs 628 us on the same grid, against 117 us for the
cancelling formula, and sits off the training path.

### Checks that confirmed the existing code

Lord and Kahl (2010, Theorem 3.7) prove the Heston characteristic function
continuous under the principal branch for every parameter set when `exp(-D tau)`
sits under the logarithm. `heston_cf.py` already uses that formulation, and its
docstring, which claimed more than the theorem proves, was corrected. Capriotti
(2011) shows that the adjoint mode returns every first-order sensitivity for a
bounded multiple of one pricing call, and `european_greeks` already takes one
reverse sweep for all four. Higham (1993) bounds recursive summation by
`(n - 1) u` and pairwise summation by `u log2 n`. Every large reduction here is
a single torch reduction of the pairwise class.

## Extensions from the same literature

### Rough Bergomi, sampled exactly on the grid

Bayer, Friz and Gatheral (2016, Section 4) simulate the rough Bergomi model
through the Cholesky factor of the joint covariance of the Volterra process and
the price driver, and call it slow because of the dense factor.
`RoughBergomiSimulator` builds that covariance from their eq. 4.1, with the
hypergeometric function from mpmath. On a rebalancing grid the factor is small.
For 30 dates it takes 40 ms once per configuration, and pyperf times 65536 paths
at 1.72 ms on CUDA. The variance is exact on the grid, and only the price
integral is discretised.

### Several risk aversions in one training run

Murray et al. (2022, Section 2.3) learn the policies of many risk aversions at
once by adding the level to the agent state. `LevelFeatures` shows each path its
level, and `MultiLevelRisk` averages each level's own risk over its paths. For a
band policy over five entropic risk aversions, one run took 7.8 s against 31.1 s
for five separate runs of the same batch and length. The entropic risk on 131072
held-out paths measured as follows.

| Risk aversion | Five separate runs | One joint run |
| ------------- | ------------------ | ------------- |
| 0.5           | 0.3094             | 0.3146        |
| 1             | 0.3833             | 0.3871        |
| 2             | 0.5117             | 0.5248        |
| 4             | 0.8603             | 0.8692        |
| 8             | 6.5999             | 1.6991        |

Each level of the joint run sees a fifth of the batch. At risk aversion 8 the
separate run failed where the joint one did not. The entropic gradient weights
paths by their softmax, which concentrates on the worst few at high aversion.

### Deep backward dynamic programming

`solve_backward` follows the DBDP1 scheme of Hure, Pham and Warin (2020, eq.
3.6), and its reflected variant (eq. 3.10) prices optimal stopping. On the
one-dimensional call with a 5 per cent rate it came within 1 per cent of
Black-Scholes over three seeds in about 1.7 s. The global deep BSDE solver was
closer and faster on that problem. On the American put the reflected scheme came
within 3 per cent of the binomial tree, a problem the global solver cannot pose.
The upward bias is the expected effect of reflecting a noisy fit on the payoff.

## Tooling and tests

The hand-written timing loop and JSON baseline gate were replaced by
pytest-benchmark for the reasons in the Method section, and the nightly GPU
workflow compares each run against the previous night's result on the same
runner with a 15 per cent median threshold, rather than against a file measured
on another device.

The repository now renders the shared template rather than keeping its own copy
of the configuration, whose duplication Sharma, Fragkoulis and Spinellis (2016)
identify as a configuration smell. The test run is strict: warnings are errors,
test order is random and each test has a timeout. Order dependence is one of the
main causes of flaky tests (Luo et al., 2014; Zhang et al., 2014). The warning
filter found two tests here that converted a gradient-carrying tensor to a
float, and each failed only when it was the first in the process to trigger
PyTorch's once-per-process warning, which is the order dependence the random
ordering exposes. The template's `nox` sessions add property-based test
generation with Hypothesis (Claessen and Hughes, 2000; MacIver et al., 2019) and
mutation testing with cosmic-ray (Jia and Harman, 2011).

cosmic-ray applied 178 mutations to `training/engine.py`, and the unit tests
killed 154. The generated equivalence test against the per-date settlement loop
killed 11 of the 24 survivors, among them every mutation of the time-to-maturity
grid and of the cost and liquidation indices, for a score of 165 of 178. Eight
of the 13 that remain cannot change a result: three alter a path sum whose
argument is only ever one- or two-dimensional, three alter the gradient
checkpoint switch, which recomputes the same values, one enables the autocast
weight cache, which matters only under graph capture and so outside this test
command, and one lengthens the maturity grid past the last entry the loop reads.
The other five change the default premium and the default `amp` flag, which
every generated call passes explicitly.

## Results

| Measurement                                  | Before         | After           |
| -------------------------------------------- | -------------- | --------------- |
| Kernel launches per eager iteration          | 1258           | 801             |
| Eager CUDA training step, median             | 21.0 ms        | 14.2 ms         |
| Generated training, 200 iterations, float32  | 3.04 s         | 2.67 s          |
| Same, TF32 matrix multiplies                 |                | 2.41 s          |
| Same, bfloat16 autocast                      |                | 1.80 s          |
| Bootstrap of expected shortfall, CPU         | 2.09 s         | 451 ms          |
| Bootstrap of expected shortfall, CUDA        | 445 ms         | 15.6 ms         |
| Deep BSDE, 200 iterations                    | 716 ms         | 301 ms          |
| Heston calibration, golden quotes            | 3.07 s, 4.2e-8 | 0.11 s, 7.8e-18 |
| Merton closed form                           | 1.82 ms        | 108 us          |
| Eager CUDA step, band policy                 | 42.8 ms        | 29.6 ms         |
| Implied volatility, 4096 quotes, max error   | 6.97 ms, 7e-2  | 4.60 ms, 3e-15  |
| ES95 of a band policy, paths times variance  | 122            | 0.081           |
| Band step, four entropic levels, CUDA        | 36.5 ms        | 33.4 ms         |
| Rough Bergomi factor, new maturity, 30 dates | 26.6 ms        | 7.62 us         |

The eager CPU step, the eager and fused generators and the barrier pricers did
not change beyond their spreads, and nothing in this review touched them.

## What was measured and left

The fused generators already run the path recursion in registers with cuRAND's
Philox generator, whose counter-based design (Salmon et al., 2011) is what makes
the noise streams addressable and the in-graph offset possible. They take 18 us
for GBM and 28 us for Heston at 65536 paths, a fraction of a percent of a
training iteration, so they were not changed. The eager Heston sampler at 4.2 ms
keeps the full-truncation Euler scheme Lord, Koekkoek and van Dijk (2009)
recommend as the reference oracle for the fused kernel.

`NoiseSpec` seeds its generators through SplitMix64, the published mixer of
Steele, Lea and Flood (2014), rather than an ad hoc hash. Replacing it would
reseed every committed experiment store.

## Second profile, after the extensions

The same instruments were run again once the extensions were in, on each
workload the branch had added or changed: the feedforward, recurrent and band
training steps, the band step over four entropic levels, the pricer and the
implied-volatility inversion over 4096 quotes on the device, the backward scheme
over ten dates of 20 steps at 1024 paths, and rough Bergomi sampling at 65536
paths. Each workload ran twice to warm up and once under `torch.profiler`, which
counted the kernel launches and the blocking host synchronisations from the CUDA
runtime events and summed the kernel time on the device. pyperf timed the same
callable, ending in a device synchronisation.

| Workload                        | Kernels | Host span | Launches | Host syncs | pyperf          |
| ------------------------------- | ------- | --------- | -------- | ---------- | --------------- |
| Feedforward training step       | 11.2 ms | 13.3 ms   | 891      | 0          | 15.3 +- 0.7 ms  |
| Recurrent training step         | 44.4 ms | 23.2 ms   | 889      | 0          | 48.5 +- 0.9 ms  |
| Band step, four entropic levels | 26.0 ms | 17.2 ms   | 853      | 30         | 36.5 +- 1.9 ms  |
| Pricer, 4096 quotes, CUDA       | 0.50 ms | 2.3 ms    | 363      | 9          | 4.20 +- 0.29 ms |
| Implied volatility, 4096, CUDA  | 2.2 ms  | 8.5 ms    | 1295     | 30         | 14.7 +- 0.7 ms  |
| Backward scheme, 200 steps      | 20.5 ms | 104 ms    | 13426    | 217        | 307 +- 9 ms     |
| Rough Bergomi, 65536 paths      | 1.4 ms  | 0.9 ms    | 26       | 1          | 1.68 +- 0.05 ms |

The table separates two regimes. Where the kernel time exceeds the host span, as
for the recurrent step, the device is the bottleneck and the remedy is in the
kernels. Where the host span exceeds it, as for the pricer, the inversion and
the backward scheme, the device waits on Python and the driver, and faster
kernels would change nothing.

### Host synchronisations the extensions introduced

A blocking copy from pageable host memory stalls the host until the device has
drained its queue, so a synchronisation inside a loop serialises host and device
once per iteration. The profile found two, both in code this branch added.
`LevelFeatures` built its code tensor from a Python tuple at every date, one
blocking copy per date, and now builds the column once per path count, dtype and
device. The band step over four levels went from 30 synchronisations to none,
and from 36.5 +- 1.9 ms to 33.4 +- 1.0 ms. The backward scheme built the time of
its date as a device tensor at every optimisation step and now fills it once per
date, which took it from 217 synchronisations to 17. Its run time moved from 307
+- 9 ms to 291 +- 19 ms, inside the spread, because the remaining cost is
dispatch: issuing the 67 launches of a step takes the host about 0.5 ms, while
the device runs them in about 0.1 ms.

The remedy for that dispatch cost is the one the training loop already uses:
capturing the whole optimisation step, the backward and the fused Adam update in
one CUDA graph and replaying it. Kwon et al. (2020) measure the same regime, in
which the framework's scheduling of small kernels dominates the kernels
themselves, and remove it by recording the GPU tasks once and replaying them.
Capture needs a step with no synchronisation in it, which the change above
provides, and fixed shapes, which the scheme has within each date.

### The pricer and the inversion wait on the host

On the device the implied-volatility inversion is three times slower than on the
processor, 14.7 +- 0.7 ms against 4.52 +- 0.18 ms for the same 4096 quotes, and
the pricer is slower by more, 4.20 +- 0.29 ms against 892 +- 18 us. The
inversion's kernels take 2.2 ms of an 8.5 ms host span. The normalised Black
function picks one of four regimes per element and evaluates a regime only where
it is selected, through `bool(mask.any())` and boolean indexing, and each of
those is a synchronisation. The inversion calls it five times, for 30
synchronisations and 1295 launches.

Evaluating every regime on every element and selecting with `torch.where` is the
predicated form of the same computation. On a SIMT device the lanes of a warp
that take different branches already execute both paths one after the other
(Fung et al., 2009), and Anantpur and Govindarajan (2014) show that converting
divergent control flow into predicated straight-line code can run faster than
the branching form. The cost is the arithmetic of the unselected regimes,
including the two 17-term series, on elements that do not need them. The device
here is idle for most of the span, so that arithmetic is expected to cost less
than the synchronisations, and with none left the whole inversion becomes
capturable as one graph. That trade is to be measured before it is made, and the
processor path keeps the selective form.

### The band network over all dates

Running the band network once over all dates turned 30 small products into one.
The weight gradient of the first layer is now a single `sgemm_largek` call that
reduces over 65536 times 30 rows, 29 per cent of the kernel time, and the SiLU
forward and backward over the same activations take another 29 per cent. The
first is a matrix product whose reduction dimension dwarfs the other two, the
shape Stream-K addresses by splitting the reduction across every multiprocessor
and fixing up the partial sums (Osama et al., 2023), and Tang et al. (2021)
measure the same tall-and-skinny shape in mixed precision. cuBLAS already
selects a large-K kernel here, and its split-K reduction kernels appear in the
backward scheme's profile, so what remains open is the precision rather than the
decomposition: the product runs on the FP32 SIMT pipeline, and the TF32 and
bfloat16 results earlier in this review apply to it.

The SiLU pair is bound by bandwidth, since each element is read and written once
for a handful of operations. Muller et al. (2021) show that a network this
narrow, 64 wide, fits in on-chip memory, and their fully fused multilayer
perceptron evaluates every layer of it inside one kernel, so the activations
never reach global memory. That would remove the SiLU traffic and the
bias-gradient reductions together. It needs either a CUDA kernel beside the
Philox kernels in `csrc` or the `torch.compile` path that the Windows build
cannot take.

### The recurrent policy is bound by the device

The recurrent step is the one workload whose kernels outlast the host, 44.4 ms
of kernels against a 23.2 ms host span. The GRU cell forward and backward take
40 per cent of the kernel time and the bias-gradient reductions another 13, all
issued once per date, because each date's input includes the position the
previous date chose. That dependency rules out cuDNN's sequence kernels, which
need the whole input sequence up front. Appleyard, Kocisky and Blunsom (2016)
cut exactly these per-step costs by fusing each step's pointwise operations and
by combining the matrix products that do not depend on the recurrence across
time steps into one. The second applies to the backward here: once every date's
gates are known, the weight and bias gradients of all 30 dates are one product
and one reduction rather than 30 of each.

### The rough Bergomi factor is built on the host

Sampling 65536 rough Bergomi paths is one tensor-core double-precision product
and takes 1.68 +- 0.05 ms. Building the factor is the cost: one mpmath
hypergeometric evaluation per pair of dates, 32.0 +- 4.9 ms for 30 dates, 138 +-
18 ms for 64 and 495 +- 9 ms for 128, quadratic in the grid as the pair count
predicts.

On a uniform grid the covariance is self-similar in the maturity. The Volterra
block scales as `maturity^(2H)`, the cross block as `maturity^(H + 1/2)` and the
driver block as `maturity`, so the factor for any maturity is the unit-maturity
factor with its rows rescaled, which agrees with a direct build to 6e-15
relative. The factor now caches the unit build per Hurst exponent, correlation
and grid, and a 30-date factor for a maturity not seen before costs 7.62 us
against 26.6 ms.

The first build remains, and two remedies apply to it in order. Evaluating `G`
in double precision rather than through mpmath is the first; Pearson, Olver and
Porter (2016) compare the series, transformations and quadrature that do so
across the parameter plane, and Johansson (2019) gives the rigorous
error-bounded evaluation that would check a faster one. On grids fine enough
that the cubic Cholesky factorisation itself dominates, the hybrid scheme of
Bennedsen, Lunde and Pakkanen (2017) simulates the Volterra process in
`O(n log n)` per path through a discrete convolution, and McCrickerd and
Pakkanen (2018) pair it with conditional Monte Carlo and control variates that
cut the path count needed for rough Bergomi prices.

### Mutation scores of the ghostwritten tests

cosmic-ray ran against each new module with its ghostwritten test alone. An
equivalence test carries an oracle, the replaced implementation or the defining
formula in mpmath, and fails when a value moves. A fuzz test carries none: it
calls the function over its valid domain and fails only when the call raises. On
`networks.py` its test killed none of 7 mutants, on `risk/multi_level.py` 77 of
the 91 that completed and on `features.py` 28 of 158, every kill by an exception
or a timeout. The last module also holds the feature maps that predate this
branch, which the ghostwritten file for `LevelFeatures` never calls. The fuzz
tests therefore guard the domain of each function and say nothing about its
values, which is why a rewrite keeps its predecessor in `deephedging.baselines`
for the ghostwriter to compare against, as the settlement engine, the Merton
series, the bootstrap, the pricer, the inversion and the rough Bergomi factor
now do.

The template's `mutants` session formats its test path with the platform
separator, which cosmic-ray's POSIX split of the test command turns into a path
that does not exist, so on Windows the session counts every mutant as killed.
The scores above come from cosmic-ray run directly with a POSIX path.

## Remaining bottlenecks

The policy recursion is sequential by construction, since each position feeds
the next observation (Buehler et al., 2019). Two costs remain inside it.

The first is the per-date weight-gradient GEMM of a policy whose network reads
the held position, which reduces over 65536 rows at every date. The band policy
now runs its network once over all dates. For the feedforward and recurrent
policies the batching of Appleyard, Kocisky and Blunsom (2016) would take a
custom autograd function over the whole recursion that accumulates the weight
gradients of all dates in one multiply.

The second is memory traffic. With bfloat16 the largest remaining kernels are
the bias-gradient reductions and the SiLU backward, both elementwise over 65536
by 64 activations and bound by bandwidth in the sense of the roofline model
(Williams, Waterman and Patterson, 2009). Ivanov et al. (2020) show that
operator fusion is the remedy for this class of cost in training, and Wang, Lin
and Yi (2010) measure it for GPUs. In PyTorch that is `torch.compile`, whose
Inductor backend generates fused Triton kernels (Ansel et al., 2024; Tillet,
Kung and Cox, 2019; Li et al., 2021). Triton publishes no Windows build, and
`torch.compile` on a CUDA module here stops with `TritonMissing`, so the
`compile_policy` option is exercised on the CPU backend alone and excludes graph
capture. A Linux GPU runner is where that remedy can be measured.

For pricing rather than training, multilevel Monte Carlo (Giles, 2008) and the
quadratic-exponential Heston scheme (Andersen, 2008) reduce the path count and
the step count needed for a given error, and neither is implemented yet.

## References

- Allen, R., & Kennedy, K. (1987). Automatic translation of FORTRAN programs to
  vector form. ACM Transactions on Programming Languages and Systems, 9(4),
  491-542. <https://doi.org/10.1145/29873.29875>
- Anantpur, J., & Govindarajan, R. (2014). Taming Control Divergence in GPUs
  through Control Flow Linearization. Compiler Construction, 133-153.
  <https://doi.org/10.1007/978-3-642-54807-9_8>
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic
  volatility model. The Journal of Computational Finance, 11(3), 1-42.
  <https://doi.org/10.21314/jcf.2008.189>
- Ansel, J., et al. (2024). PyTorch 2: Faster Machine Learning Through Dynamic
  Python Bytecode Transformation and Graph Compilation. Proceedings of the 29th
  ACM International Conference on Architectural Support for Programming
  Languages and Operating Systems, Volume 2, 929-947.
  <https://doi.org/10.1145/3620665.3640366>
- Appleyard, J., Kocisky, T., & Blunsom, P. (2016). Optimizing Performance of
  Recurrent Neural Networks on GPUs. arXiv.
  <https://doi.org/10.48550/ARXIV.1604.01946>
- Bayer, C., Friz, P., & Gatheral, J. (2015). Pricing under rough volatility.
  Quantitative Finance, 16(6), 887-904.
  <https://doi.org/10.1080/14697688.2015.1099717>
- Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2017). Hybrid scheme for Brownian
  semistationary processes. Finance and Stochastics, 21(4), 931-965.
  <https://doi.org/10.1007/s00780-017-0335-5>
- Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging.
  Quantitative Finance, 19(8), 1271-1291.
  <https://doi.org/10.1080/14697688.2019.1571683>
- Capriotti, L. (2011). Fast Greeks by algorithmic differentiation. The Journal
  of Computational Finance, 14(3), 3-35. <https://doi.org/10.21314/jcf.2011.234>
- Claessen, K., & Hughes, J. (2000). QuickCheck. Proceedings of the Fifth ACM
  SIGPLAN International Conference on Functional Programming, 268-279.
  <https://doi.org/10.1145/351240.351266>
- Davis, M. H. A., Panas, V. G., & Zariphopoulou, T. (1993). European Option
  Pricing with Transaction Costs. SIAM Journal on Control and Optimization,
  31(2), 470-493. <https://doi.org/10.1137/0331022>
- Delbourgo, R., & Gregory, J. A. (1985). Shape Preserving Piecewise Rational
  Interpolation. SIAM Journal on Scientific and Statistical Computing, 6(4),
  967-976. <https://doi.org/10.1137/0906065>
- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. The Annals
  of Statistics, 7(1). <https://doi.org/10.1214/aos/1176344552>
- Fang, F., & Oosterlee, C. W. (2008). A Novel Pricing Method for European
  Options Based on Fourier-Cosine Series Expansions. SIAM Journal on Scientific
  Computing, 31(2), 826-848. <https://doi.org/10.1137/080718061>
- Fung, W. W. L., Sham, I., Yuan, G., & Aamodt, T. M. (2009). Dynamic warp
  formation. ACM Transactions on Architecture and Code Optimization, 6(2), 1-37.
  <https://doi.org/10.1145/1543753.1543756>
- Georges, A., Buytaert, D., & Eeckhout, L. (2007). Statistically rigorous java
  performance evaluation. Proceedings of the 22nd Annual ACM SIGPLAN Conference
  on Object-Oriented Programming Systems, Languages and Applications, 57-76.
  <https://doi.org/10.1145/1297027.1297033>
- Giles, M. B. (2008). Multilevel Monte Carlo Path Simulation. Operations
  Research, 56(3), 607-617. <https://doi.org/10.1287/opre.1070.0496>
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer
  New York. <https://doi.org/10.1007/978-0-387-21617-1>
- Han, J., Jentzen, A., & E, W. (2018). Solving high-dimensional partial
  differential equations using deep learning. Proceedings of the National
  Academy of Sciences, 115(34), 8505-8510.
  <https://doi.org/10.1073/pnas.1718942115>
- Higham, N. J. (1993). The Accuracy of Floating Point Summation. SIAM Journal
  on Scientific Computing, 14(4), 783-799. <https://doi.org/10.1137/0914050>
- Hure, C., Pham, H., & Warin, X. (2020). Deep backward schemes for
  high-dimensional nonlinear PDEs. Mathematics of Computation, 89(324),
  1547-1579. <https://doi.org/10.1090/mcom/3514>
- Imaki, S., Imajo, K., Ito, K., Minami, K., & Nakagawa, K. (2021).
  No-Transaction Band Network: A Neural Network Architecture for Efficient Deep
  Hedging. arXiv. <https://doi.org/10.48550/ARXIV.2103.01775>
- Ivanov, A., Dryden, N., Ben-Nun, T., Li, S., & Hoefler, T. (2020). Data
  Movement Is All You Need: A Case Study on Optimizing Transformers. arXiv.
  <https://doi.org/10.48550/ARXIV.2007.00072>
- Jackel, P. (2015). Let's Be Rational. Wilmott, 2015(75), 40-53.
  <https://doi.org/10.1002/wilm.10395>
- Jia, Y., & Harman, M. (2011). An Analysis and Survey of the Development of
  Mutation Testing. IEEE Transactions on Software Engineering, 37(5), 649-678.
  <https://doi.org/10.1109/tse.2010.62>
- Jia, Z., Maggioni, M., Staiger, B., & Scarpazza, D. P. (2018). Dissecting the
  NVIDIA Volta GPU Architecture via Microbenchmarking. arXiv.
  <https://doi.org/10.48550/ARXIV.1804.06826>
- Johansson, F. (2019). Computing Hypergeometric Functions Rigorously. ACM
  Transactions on Mathematical Software, 45(3), 1-26.
  <https://doi.org/10.1145/3328732>
- Kalamkar, D., et al. (2019). A Study of BFLOAT16 for Deep Learning Training.
  arXiv. <https://doi.org/10.48550/ARXIV.1905.12322>
- Kalibera, T., & Jones, R. (2013). Rigorous benchmarking in reasonable time.
  Proceedings of the 2013 International Symposium on Memory Management, 63-74.
  <https://doi.org/10.1145/2464157.2464160>
- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
  arXiv. <https://doi.org/10.48550/ARXIV.1412.6980>
- Kwon, W., Yu, G.-I., Jeong, E., & Chun, B.-G. (2020). Nimble: Lightweight and
  Parallel GPU Task Scheduling for Deep Learning. arXiv.
  <https://doi.org/10.48550/ARXIV.2012.02732>
- Li, M., et al. (2021). The Deep Learning Compiler: A Comprehensive Survey.
  IEEE Transactions on Parallel and Distributed Systems, 32(3), 708-727.
  <https://doi.org/10.1109/tpds.2020.3030548>
- Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large
  scale optimization. Mathematical Programming, 45(1-3), 503-528.
  <https://doi.org/10.1007/bf01589116>
- Lord, R., & Kahl, C. (2010). Complex logarithms in Heston-like models.
  Mathematical Finance, 20(4), 671-694.
  <https://doi.org/10.1111/j.1467-9965.2010.00416.x>
- Lord, R., Koekkoek, R., & Dijk, D. V. (2009). A comparison of biased
  simulation schemes for stochastic volatility models. Quantitative Finance,
  10(2), 177-194. <https://doi.org/10.1080/14697680802392496>
- Luo, Q., Hariri, F., Eloussi, L., & Marinov, D. (2014). An empirical analysis
  of flaky tests. Proceedings of the 22nd ACM SIGSOFT International Symposium on
  Foundations of Software Engineering, 643-653.
  <https://doi.org/10.1145/2635868.2635920>
- MacIver, D., Hatfield-Dodds, Z., & Contributors, M. (2019). Hypothesis: A new
  approach to property-based testing. Journal of Open Source Software,
  4(43), 1891. <https://doi.org/10.21105/joss.01891>
- McCrickerd, R., & Pakkanen, M. S. (2018). Turbocharging Monte Carlo pricing
  for the rough Bergomi model. Quantitative Finance, 18(11), 1877-1886.
  <https://doi.org/10.1080/14697688.2018.1459812>
- Micikevicius, P., et al. (2017). Mixed Precision Training. arXiv.
  <https://doi.org/10.48550/ARXIV.1710.03740>
- Muller, T., Rousselle, F., Novak, J., & Keller, A. (2021). Real-time neural
  radiance caching for path tracing. ACM Transactions on Graphics, 40(4), 1-16.
  <https://doi.org/10.1145/3450626.3459812>
- Murray, P., Wood, B., Buehler, H., Wiese, M., & Pakkanen, M. (2022). Deep
  Hedging: Continuous Reinforcement Learning for Hedging of General Portfolios
  across Multiple Risk Aversions. Proceedings of the Third ACM International
  Conference on AI in Finance, 361-368.
  <https://doi.org/10.1145/3533271.3561731>
- Mytkowicz, T., Diwan, A., Hauswirth, M., & Sweeney, P. F. (2009). Producing
  wrong data without doing anything obviously wrong! Proceedings of the 14th
  International Conference on Architectural Support for Programming Languages
  and Operating Systems, 265-276. <https://doi.org/10.1145/1508244.1508275>
- Osama, M., Merrill, D., Cecka, C., Garland, M., & Owens, J. D. (2023).
  Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix
  Multiplication on the GPU. Proceedings of the 28th ACM SIGPLAN Annual
  Symposium on Principles and Practice of Parallel Programming, 429-431.
  <https://doi.org/10.1145/3572848.3577479>
- Owen, A. B. (1998). Scrambling Sobol' and Niederreiter-Xing Points. Journal of
  Complexity, 14(4), 466-489. <https://doi.org/10.1006/jcom.1998.0487>
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep
  Learning Library. arXiv. <https://doi.org/10.48550/ARXIV.1912.01703>
- Pearson, J. W., Olver, S., & Porter, M. A. (2016). Numerical methods for the
  computation of the confluent and Gauss hypergeometric functions. Numerical
  Algorithms, 74(3), 821-866. <https://doi.org/10.1007/s11075-016-0173-0>
- Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. (2011). Parallel
  random numbers. Proceedings of 2011 International Conference for High
  Performance Computing, Networking, Storage and Analysis, 1-12.
  <https://doi.org/10.1145/2063384.2063405>
- Sharma, T., Fragkoulis, M., & Spinellis, D. (2016). Does your configuration
  code smell? Proceedings of the 13th International Conference on Mining
  Software Repositories, 189-200. <https://doi.org/10.1145/2901739.2901761>
- Steele, G. L., Lea, D., & Flood, C. H. (2014). Fast splittable pseudorandom
  number generators. Proceedings of the 2014 ACM International Conference on
  Object Oriented Programming Systems Languages & Applications, 453-472.
  <https://doi.org/10.1145/2660193.2660195>
- Tang, H., Komatsu, K., Sato, M., & Kobayashi, H. (2021). Efficient
  Mixed-Precision Tall-and-Skinny Matrix-Matrix Multiplication for GPUs.
  International Journal of Networking and Computing, 11(2), 267-282.
  <https://doi.org/10.15803/ijnc.11.2_267>
- Tillet, P., Kung, H. T., & Cox, D. (2019). Triton: an intermediate language
  and compiler for tiled neural network computations. Proceedings of the 3rd ACM
  SIGPLAN International Workshop on Machine Learning and Programming Languages,
  10-19. <https://doi.org/10.1145/3315508.3329973>
- Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific
  computing in Python. Nature Methods, 17(3), 261-272.
  <https://doi.org/10.1038/s41592-019-0686-2>
- Wang, G., Lin, Y., & Yi, W. (2010). Kernel Fusion: An Effective Method for
  Better Power Efficiency on Multithreaded GPU. 2010 IEEE/ACM International
  Conference on Green Computing and Communications & International Conference on
  Cyber, Physical and Social Computing, 344-350.
  <https://doi.org/10.1109/greencom-cpscom.2010.102>
- Whalley, A. E., & Wilmott, P. (1997). An Asymptotic Analysis of an Optimal
  Hedging Model for Option Pricing with Transaction Costs. Mathematical Finance,
  7(3), 307-324. <https://doi.org/10.1111/1467-9965.00034>
- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline. Communications
  of the ACM, 52(4), 65-76. <https://doi.org/10.1145/1498765.1498785>
- Zhang, S., Jalali, D., Wuttke, J., Muslu, K., Lam, W., Ernst, M. D., & Notkin,
  D. (2014). Empirically revisiting the test independence assumption.
  Proceedings of the 2014 International Symposium on Software Testing and
  Analysis, 385-396. <https://doi.org/10.1145/2610384.2610404>
