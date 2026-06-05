# DeepHedging

Deep hedging research and implementation: learning optimal hedging strategies
for derivative portfolios with neural networks, trained directly on simulated
market dynamics under realistic frictions (transaction costs, market impact,
risk limits).

## Overview

Classical delta hedging assumes frictionless, complete markets. Deep hedging
replaces the analytic hedge with a parameterised policy (neural network) that
maps market state to hedge positions, optimised against a convex risk measure
(e.g. CVaR, entropic risk) over simulated paths.

## Project Structure

```
DeepHedging/
|-- src/          # Library code
|-- notebooks/    # Research notebooks
|-- tests/        # Unit and integration tests
|-- data/         # Local datasets (gitignored)
`-- docs/         # Documentation
```

## Getting Started

### Prerequisites

- Python 3.11+

### Installation

```bash
git clone <repo-url>
cd DeepHedging
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -e ".[dev]"
```

### Usage

```bash
# TODO: add entry points / examples
```

## Testing

```bash
pytest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.

## License

Proprietary. All rights reserved. See [LICENSE](LICENSE).
