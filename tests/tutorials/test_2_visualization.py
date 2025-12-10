"""Test tutorial 2: Visualization and Scene."""

import pytest

from .conftest import run_tutorial


@pytest.mark.tutorial
def test_visualization() -> None:
    """Test that the visualization tutorial runs without errors."""
    run_tutorial("2_visualization.py")
