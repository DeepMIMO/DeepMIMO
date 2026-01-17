"""Test application: Channel Prediction."""

import pytest

from .conftest import run_application


@pytest.mark.application
@pytest.mark.slow
def test_channel_prediction() -> None:
    """Test that the channel prediction application runs without errors.

    This test verifies the complete workflow:
    - Simple two-user interpolation
    - Extracting all linear sequences
    - Creating uniform-length sequences
    - Computing baseline channels (no interpolation)
    - Computing interpolated channels
    - Adding Doppler effects with multiple configurations
    - Generating all visualizations and comparisons

    Note: This is marked as slow since it performs channel generation
    and creates multiple visualizations.
    """
    run_application("1_channel_prediction.py")
