"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

# make src importable without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from fimicyber.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()
