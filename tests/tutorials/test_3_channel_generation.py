"""Test tutorial 3: Detailed Channel Generation."""

import pytest

from .conftest import run_tutorial


@pytest.mark.tutorial
def test_channel_generation() -> None:
    """Test that the channel generation tutorial runs without errors."""
    run_tutorial("3_channel_generation.py")
