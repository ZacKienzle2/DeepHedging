# Contributor task runner. Every target wraps the same uv-pinned toolchain
# the CI lanes use, so local and remote results agree. Run `make help` for
# the list.

.DEFAULT_GOAL := help
.PHONY: help sync lint format format-check typecheck test test-fast test-gpu \
        coverage check bench hooks clean

help: ## List the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

sync: ## Install the project with the dev extra
	uv sync --extra dev

lint: ## Run the linter
	uv run ruff check .

format: ## Format the codebase in place
	uv run ruff format .

format-check: ## Check formatting without writing
	uv run ruff format --check .

typecheck: ## Run the strict type checker
	uv run pyright

test-fast: ## Run the fast test suite
	uv run pytest -q -m "not slow and not gpu"

test: ## Run the full CPU suite including training goldens
	uv run pytest -q -m "not gpu"

test-gpu: ## Run the CUDA kernel and capture suite
	uv run pytest -q -m gpu

coverage: ## Measure coverage over the CPU suite
	uv run --with pytest-cov pytest -q -m "not gpu" \
		--cov=deephedging --cov-report=term-missing --cov-report=xml

check: lint format-check typecheck test-fast ## Run the full local gate

bench: ## Run the throughput benchmark on CUDA
	uv run python benchmarks/bench_paths.py --device cuda

hooks: ## Install the pre-commit hooks
	uv run pre-commit install

clean: ## Remove caches and build artifacts
	uv run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	rm -rf .pytest_cache .ruff_cache build dist coverage.xml .coverage
