"""Launches the experiment studies across the available devices.

The grid cells inside each study are independent training runs, and the
studies themselves are independent programs, so the cheapest parallelism
is one study per device with no gradient communication at all. Each
worker process sees a single device through its visibility mask, runs
its queue of studies sequentially, and the per-study resume logic makes
the whole launch restartable. On a single-device machine the launcher
degenerates to running the studies in order.

    uv run python experiments/launch.py [--smoke]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from threading import Thread

import torch

STUDIES = (
    "bsde_pricing.py",
    "barrier_hedging.py",
    "multi_instrument.py",
    "architecture_study.py",
    "objective_study.py",
    "hedging_frontier.py",
)


def run_queue(device_index: int, studies: list[str], smoke: bool) -> list[int]:
    """Runs a queue of studies on one device.

    Args:
        device_index: CUDA device the worker owns.
        studies: Study file names to run in order.
        smoke: Whether to pass the smoke flag through.

    Returns:
        Exit codes in queue order.
    """
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device_index))
    directory = Path(__file__).parent
    codes = []
    for study in studies:
        command = [sys.executable, str(directory / study)]
        if smoke:
            command.append("--smoke")
        print(f"device {device_index}: {study}")
        codes.append(subprocess.run(command, env=environment, check=False).returncode)
    return codes


def main() -> None:
    """Partitions the studies across devices and waits for completion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()

    device_count = max(torch.cuda.device_count(), 1)
    queues: list[list[str]] = [[] for _ in range(device_count)]
    for index, study in enumerate(STUDIES):
        queues[index % device_count].append(study)

    results: dict[int, list[int]] = {}

    def worker(device_index: int) -> None:
        results[device_index] = run_queue(device_index, queues[device_index], arguments.smoke)

    threads = [
        Thread(target=worker, args=(index,)) for index in range(device_count) if queues[index]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures = sum(code != 0 for codes in results.values() for code in codes)
    print(f"{len(STUDIES)} studies across {device_count} device(s), {failures} failure(s)")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
