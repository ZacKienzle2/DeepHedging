# Experiments

Each study is a standalone runner that trains its grid, scores every arm
on common evaluation paths, and appends one JSON line per run to its
store with the commit, library versions, device, seeds, configuration,
and loss history. Tables and figures regenerate from the stores alone.
Runners resume from completed record names, so an interrupted grid picks
up where it stopped, and `--smoke` redirects to a separate store with
reduced sizes so trial runs never mask full results.

| Study | Claim it tests | Store |
|---|---|---|
| `hedging_frontier.py` | Deep CVaR policy matches the Black-Scholes delta hedge frictionless and beats it increasingly with transaction costs, across risk aversions; stores the learned position against the model delta and an at-the-money inventory probe per run | `hedging_frontier.jsonl` |
| `band_scaling.py` | The no-trade band widens with cost at the Whalley-Wilmott cube-root exponent at the median objective; reads the frontier store, trains nothing | `hedging_frontier.jsonl` |
| `barrier_hedging.py` | Where no analytic hedge exists, an up-and-out call under Heston, the deep policy beats the vanilla delta fallback; a running-maximum observation arm isolates the feature's value | `barrier_hedging.jsonl` |
| `multi_instrument.py` | A tradable variance swap spans the volatility risk the spot cannot, cutting tail risk beyond the best spot-only hedge | `multi_instrument.jsonl` |
| `architecture_study.py` | Free-form, recurrent, and band-structured policies at parameter budgets matched within two percent, so capacity cannot masquerade as inductive bias | `architecture_study.jsonl` |
| `objective_study.py` | Expected shortfall at rising confidence against entropic risk aversion, everything else fixed; how the objective trades the mean against the tail | `objective_study.jsonl` |
| `bsde_pricing.py` | Deep BSDE prices the geometric basket call against its lognormal closed form to fifty dimensions | `bsde_pricing.jsonl` |

## Running

```bash
uv run python experiments/<study>.py --smoke    # minutes on CPU, separate store
uv run python experiments/<study>.py            # full grid, CUDA when available
uv run python experiments/band_scaling.py       # analysis only, after the frontier
```

The Heston studies accept `--fused` to sample with the fused CUDA kernel,
which generates two orders of magnitude faster but draws from its own
Philox streams; the committed stores were produced with the eager default
and reproduce against it.

## Reading a store

```python
from deephedging.experiment import load_records

for record in load_records("experiments/hedging_frontier.jsonl"):
    print(record.name, record.setup["es_95"])
```

Every record's `provenance` carries the commit hash that produced it, so
a stored number is checkable against the exact code that generated it.
