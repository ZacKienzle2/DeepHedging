from __future__ import annotations

import importlib.metadata

import deephedging as m


def test_version() -> None:
    assert importlib.metadata.version("deephedging") == m.__version__
