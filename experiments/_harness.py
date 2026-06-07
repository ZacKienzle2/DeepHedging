"""Shared mechanics for the experiment runners.

Every study parses the same flags, resumes from the same record-name
scheme, redirects smoke runs to the same separate store, and releases
the device the same way between runs. Centralising the mechanism stops
it drifting across runners; the grids, premiums, and scoring stay in
each study where they carry the scientific content.
"""

import argparse
import gc

import torch

from deephedging.experiment import load_records


def parse_study_arguments(fused_option: bool = False) -> argparse.Namespace:
    """Parses the flags shared by every study runner.

    Args:
        fused_option: Whether the study offers the fused CUDA sampler.

    Returns:
        Parsed arguments with ``smoke``, ``device``, and, when offered,
        ``fused``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    if fused_option:
        parser.add_argument("--fused", action="store_true")
    arguments = parser.parse_args()
    if fused_option and arguments.fused:
        from deephedging.market import kernels_available

        if not kernels_available():
            parser.error("--fused requires the CUDA kernel toolchain")
    return arguments


def open_store(results: str, smoke: bool) -> tuple[str, set[str]]:
    """Resolves the result store and the completed run names.

    Smoke runs write to a separate store so reduced-size records never
    mask full runs on resume.

    Args:
        results: Path of the full-run store.
        smoke: Whether this is a smoke run.

    Returns:
        The store path and the names already recorded in it.
    """
    path = results.replace(".jsonl", "_smoke.jsonl") if smoke else results
    return path, {record.name for record in load_records(path)}


def release_device(device: str) -> None:
    """Frees cached device memory between training runs.

    Collection runs first so any captured graph destructs before its
    pool is released; freeing pool memory under a live graph poisons
    the allocator, and the synchronisation orders the release after all
    queued work.

    Args:
        device: Device the previous run trained on.
    """
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
