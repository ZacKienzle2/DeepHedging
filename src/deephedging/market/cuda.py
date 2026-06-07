"""CUDA path generators implementing the simulator contracts.

The fused kernels carry the path recursion in registers, so generation
costs one launch per batch instead of one launch per step per tensor
operation. The (seed, stream) pair maps onto the Philox (seed,
subsequence) pair directly, so replay is exact per backend while
cross-backend bit equality with the eager simulators is not promised;
parity is distributional and pinned by the test suite. The extension is
compiled on first use; the eager simulators remain the reference oracle
and the fallback wherever no toolchain or device exists.
"""

import glob
import os
import shutil
import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch

from deephedging.market.noise import NoiseSpec
from deephedging.market.state import MarketState, PathFolds

_MAX_STREAM = 1 << 32


def _ensure_msvc_on_path() -> None:
    if os.name != "nt" or shutil.which("cl") is not None:
        return
    program_files = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    pattern = os.path.join(
        program_files,
        "Microsoft Visual Studio",
        "*",
        "*",
        "VC",
        "Tools",
        "MSVC",
        "*",
        "bin",
        "Hostx64",
        "x64",
    )
    candidates = sorted(glob.glob(pattern))
    if candidates:
        os.environ["PATH"] = candidates[-1] + os.pathsep + os.environ["PATH"]


@cache
def _kernels() -> Any:
    from torch.utils.cpp_extension import load

    _ensure_msvc_on_path()
    source = Path(__file__).resolve().parents[3] / "csrc" / "path_kernels.cu"
    cuda_flags = ["-O3"]
    if os.name == "nt":
        cuda_flags.append("-Xcompiler")
        cuda_flags.append("/Zc:preprocessor")
    return load(
        name="deephedging_paths",
        sources=[str(source)],
        extra_cuda_cflags=cuda_flags,
        verbose=False,
    )


def kernels_available() -> bool:
    """Reports whether the fused generators can run here.

    A broken toolchain warns with the underlying error so it stays
    distinguishable from the no-device case; silent fallback would make
    a misconfigured compiler look like a CPU-only machine.

    Returns:
        True when a CUDA device exists and the extension compiles.
    """
    if not torch.cuda.is_available():
        return False
    try:
        _kernels()
    except (RuntimeError, OSError, ImportError) as error:
        warnings.warn(
            f"CUDA device present but the kernel extension failed to build: {error}",
            stacklevel=2,
        )
        return False
    return True


def _resolve_noise(noise: NoiseSpec | None) -> NoiseSpec:
    if noise is None:
        return NoiseSpec(seed=int(torch.randint(0, 2**62, ()).item()))
    if not 0 <= noise.stream < _MAX_STREAM:
        raise ValueError(f"stream must lie in [0, 2^32), got {noise.stream}")
    return noise


@dataclass(frozen=True)
class CudaGBMSimulator:
    """Fused-kernel GBM sampler, exact in distribution.

    Attributes:
        s0: Initial spot price.
        sigma: Volatility.
        maturity: Horizon in years.
        n_steps: Number of rebalancing intervals.
        mu: Drift under the simulation measure.
        device: Always a CUDA device.
    """

    s0: float
    sigma: float
    maturity: float
    n_steps: int
    mu: float = 0.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.s0 <= 0.0:
            raise ValueError(f"s0 must be positive, got {self.s0}")
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")
        if not str(self.device).startswith("cuda"):
            raise ValueError(f"device must be a CUDA device, got {self.device}")

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates GBM market paths on the device.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state whose spot grid satisfies ``spot[0] == s0``.
        """
        spec = _resolve_noise(noise)
        spot = _kernels().gbm_paths(
            n_paths,
            self.n_steps,
            self.s0,
            self.mu,
            self.sigma,
            self.maturity,
            spec.seed,
            spec.stream,
        )
        return MarketState(spot=spot)

    def simulate_with_offset(self, n_paths: int, seed: int, offset: torch.Tensor) -> MarketState:
        """Simulates GBM paths reading the stream offset from the device.

        The kernel reads the shifted Philox subsequence from the offset
        tensor at launch, so the call is capturable in a CUDA graph and
        every replay draws the batch the offset names at replay time.
        Writing ``stream << 32`` into the tensor reproduces
        :meth:`simulate` with that stream bitwise.

        Args:
            n_paths: Number of independent paths.
            seed: Philox seed shared with the offset's stream family.
            offset: One-element int64 CUDA tensor holding the shifted
                subsequence base.

        Returns:
            Market state whose spot grid satisfies ``spot[0] == s0``.
        """
        spot = _kernels().gbm_paths_offset(
            n_paths,
            self.n_steps,
            self.s0,
            self.mu,
            self.sigma,
            self.maturity,
            seed,
            offset,
        )
        return MarketState(spot=spot)

    def simulate_folds(self, n_paths: int, noise: NoiseSpec | None = None) -> PathFolds:
        """Computes per-path statistics in registers, never the grid.

        Consumes exactly the same Philox stream as :meth:`simulate`, so
        the folds match the grid-derived statistics bitwise, which the
        parity tests pin.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Terminal, running maximum, and running minimum per path.
        """
        spec = _resolve_noise(noise)
        terminal, running_max, running_min = _kernels().gbm_folds(
            n_paths,
            self.n_steps,
            self.s0,
            self.mu,
            self.sigma,
            self.maturity,
            spec.seed,
            spec.stream,
        )
        return PathFolds(terminal=terminal, running_max=running_max, running_min=running_min)


@dataclass(frozen=True)
class CudaHestonSimulator:
    """Fused-kernel Heston sampler with the full-truncation scheme.

    Attributes:
        s0: Initial spot price.
        v0: Initial variance.
        kappa: Mean-reversion speed of the variance.
        theta: Long-run variance level.
        xi: Volatility of variance.
        rho: Correlation between price and variance Brownian drivers.
        maturity: Horizon in years.
        n_steps: Number of rebalancing intervals.
        mu: Drift under the simulation measure.
        device: Always a CUDA device.
    """

    s0: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    maturity: float
    n_steps: int
    mu: float = 0.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.s0 <= 0.0:
            raise ValueError(f"s0 must be positive, got {self.s0}")
        if self.v0 < 0.0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta < 0.0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if self.xi < 0.0:
            raise ValueError(f"xi must be non-negative, got {self.xi}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")
        if not str(self.device).startswith("cuda"):
            raise ValueError(f"device must be a CUDA device, got {self.device}")

    def simulate(self, n_paths: int, noise: NoiseSpec | None = None) -> MarketState:
        """Simulates Heston market paths on the device.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Market state with the clamped ``variance`` channel attached.
        """
        spec = _resolve_noise(noise)
        spot, variance = _kernels().heston_paths(
            n_paths,
            self.n_steps,
            self.s0,
            self.v0,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.mu,
            self.maturity,
            spec.seed,
            spec.stream,
        )
        return MarketState(spot=spot, aux={"variance": variance})

    def simulate_with_offset(self, n_paths: int, seed: int, offset: torch.Tensor) -> MarketState:
        """Simulates Heston paths reading the stream offset from the device.

        The kernel reads the shifted Philox subsequence from the offset
        tensor at launch, so the call is capturable in a CUDA graph and
        every replay draws the batch the offset names at replay time.
        Writing ``stream << 32`` into the tensor reproduces
        :meth:`simulate` with that stream bitwise.

        Args:
            n_paths: Number of independent paths.
            seed: Philox seed shared with the offset's stream family.
            offset: One-element int64 CUDA tensor holding the shifted
                subsequence base.

        Returns:
            Market state with the clamped ``variance`` channel attached.
        """
        spot, variance = _kernels().heston_paths_offset(
            n_paths,
            self.n_steps,
            self.s0,
            self.v0,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.mu,
            self.maturity,
            seed,
            offset,
        )
        return MarketState(spot=spot, aux={"variance": variance})

    def simulate_folds(self, n_paths: int, noise: NoiseSpec | None = None) -> PathFolds:
        """Computes per-path statistics in registers, never the grid.

        Consumes exactly the same Philox stream as :meth:`simulate`, so
        the folds match the grid-derived statistics bitwise, which the
        parity tests pin.

        Args:
            n_paths: Number of independent paths.
            noise: Optional addressable noise stream for exact replay.

        Returns:
            Terminal, running maximum, and running minimum per path.
        """
        spec = _resolve_noise(noise)
        terminal, running_max, running_min = _kernels().heston_folds(
            n_paths,
            self.n_steps,
            self.s0,
            self.v0,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.mu,
            self.maturity,
            spec.seed,
            spec.stream,
        )
        return PathFolds(terminal=terminal, running_max=running_max, running_min=running_min)
