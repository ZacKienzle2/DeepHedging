"""Append-only experiment records with seed and build provenance.

A result without the commit, library versions, device, and seeds that
produced it cannot be reproduced or compared, so every record carries
that provenance alongside the configuration and the outcome. Storage is
one JSON object per line in an append-only file: diffable, greppable,
and free of any tracking framework, matching the repository's rule that
frozen dataclasses plus plain files are the extension mechanism.
"""

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from deephedging.bsde.solver import BSDEConfig, BSDEResult
from deephedging.training.trainer import TrainConfig, TrainResult


def capture_provenance() -> dict[str, str]:
    """Captures the build and device identity of the current run.

    Returns:
        Mapping with the commit hash, library and CUDA versions, device
        name, platform string, and a UTC timestamp.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = "unknown"
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "commit": commit,
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "device": device,
        "platform": platform.platform(),
        "created_utc": datetime.now(UTC).isoformat(),
    }


@dataclass(frozen=True)
class ExperimentRecord:
    """One training run with everything needed to reproduce it.

    Attributes:
        name: Human-chosen experiment identifier.
        provenance: Build and device identity at run time.
        train_config: The training hyperparameters as plain values.
        setup: Caller-described market, payoff, cost, and risk settings.
        losses: Recorded objective trajectory.
        duration_seconds: Wall-clock training time.
    """

    name: str
    provenance: dict[str, str]
    train_config: dict[str, object]
    setup: dict[str, object]
    losses: list[float]
    duration_seconds: float

    @classmethod
    def from_run(
        cls,
        name: str,
        config: TrainConfig | BSDEConfig,
        setup: dict[str, object],
        result: TrainResult | BSDEResult,
        duration_seconds: float,
    ) -> "ExperimentRecord":
        """Builds a record from a completed training run.

        Args:
            name: Human-chosen experiment identifier.
            config: The hedging or BSDE training configuration used.
            setup: Caller-described market, payoff, cost, and risk
                settings as plain serialisable values.
            result: The training outcome.
            duration_seconds: Wall-clock training time.

        Returns:
            The populated record with fresh provenance.
        """
        return cls(
            name=name,
            provenance=capture_provenance(),
            train_config=asdict(config),
            setup=setup,
            losses=result.losses,
            duration_seconds=duration_seconds,
        )


def append_record(path: str | Path, record: ExperimentRecord) -> None:
    """Appends one record to the experiment file.

    Args:
        path: Target file; created with parents when absent.
        record: The record to persist.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="ascii") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def load_records(path: str | Path) -> list[ExperimentRecord]:
    """Loads every record from an experiment file.

    Args:
        path: The experiment file to read.

    Returns:
        Records in append order; empty when the file does not exist.
    """
    target = Path(path)
    if not target.exists():
        return []
    records = []
    with target.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.strip():
                records.append(ExperimentRecord(**json.loads(line)))
    return records
