"""Test tutorial 2: Visualization and Scene."""

from .conftest import run_tutorial


def test_visualization():
    """Test that the visualization tutorial runs without errors."""
    run_tutorial("2_visualization.py")

