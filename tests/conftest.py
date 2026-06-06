"""Shared pytest configuration."""

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skips GPU-marked tests when no CUDA device is available.

    Args:
        config: The pytest configuration object.
        items: Collected test items.
    """
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA device not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
