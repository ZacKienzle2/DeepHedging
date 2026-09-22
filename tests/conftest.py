"""Shared pytest configuration.

Hypothesis runs without a per-example deadline. The first call of a torch
operator pays for kernel selection and allocator warm-up, so the first example
of a generated test takes hundreds of times longer than the rest, and the
deadline reported that as a flaky test.
"""

import pytest
import torch
from hypothesis import settings

settings.register_profile("deephedging", deadline=None)
settings.load_profile("deephedging")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skips GPU-marked tests when no CUDA device is available.

    Args:
        items: Collected test items.
    """
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA device not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
