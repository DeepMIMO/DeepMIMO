"""Test tutorial 5: Doppler and Mobility."""

from .conftest import run_tutorial


def test_doppler_mobility():
    """Test that the doppler and mobility tutorial runs without errors."""
    run_tutorial("5_doppler_mobility.py")

