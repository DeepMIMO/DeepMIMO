"""Test tutorial 6: Beamforming."""

import pytest

from .conftest import run_tutorial


@pytest.mark.tutorial
def test_beamforming():
    """Test that the beamforming tutorial runs without errors."""
    run_tutorial("6_beamforming.py")
