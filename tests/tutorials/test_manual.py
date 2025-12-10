"""Test manual: Complete Examples Manual."""

import pytest

from .conftest import run_tutorial


@pytest.mark.tutorial
def test_manual() -> None:
    """Test that the complete examples manual runs without errors."""
    run_tutorial("manual.py")
