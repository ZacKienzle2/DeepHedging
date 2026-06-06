"""Benchmark regression gate.

Compares a fresh benchmark result against a committed baseline and
exits non-zero when any throughput falls more than the threshold below
it. The baseline is a deliberate committed artefact: improvements
ratchet the bar by updating it in review, and a baseline measured on
one device is invalid on another, so the device strings must match.

    uv run python benchmarks/compare_baseline.py baseline.json current.json
"""

import argparse
import json
import sys

_THRESHOLD = 0.85


def compare(baseline: dict[str, object], current: dict[str, object]) -> list[str]:
    """Lists every throughput regression beyond the threshold.

    Args:
        baseline: The committed baseline benchmark result.
        current: The freshly measured benchmark result.

    Returns:
        Human-readable failure lines; empty when the gate passes.

    Raises:
        ValueError: If the results were measured on different devices.
    """
    if baseline.get("device") != current.get("device"):
        raise ValueError(
            f"device mismatch: baseline {baseline.get('device')} vs current {current.get('device')}"
        )
    failures = []
    for key, recorded in baseline.items():
        if not key.endswith("_mpaths_per_s") or not isinstance(recorded, (int, float)):
            continue
        measured = current.get(key)
        if not isinstance(measured, (int, float)):
            failures.append(f"{key}: missing from current result")
            continue
        if measured < _THRESHOLD * recorded:
            failures.append(
                f"{key}: {measured:.2f} fell below {_THRESHOLD:.0%} of baseline {recorded:.2f}"
            )
    return failures


def main() -> int:
    """Runs the gate over two result files.

    Returns:
        Zero when no metric regresses beyond the threshold.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("current")
    arguments = parser.parse_args()
    with open(arguments.baseline, encoding="ascii") as handle:
        baseline = json.load(handle)
    with open(arguments.current, encoding="ascii") as handle:
        current = json.load(handle)
    failures = compare(baseline, current)
    for line in failures:
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
