"""Test tutorial 1: Getting Started."""

import pytest

from .conftest import run_tutorial


@pytest.mark.tutorial
def test_getting_started() -> None:
    """Test that the getting started tutorial runs without errors."""
    run_tutorial("1_getting_started.py")
