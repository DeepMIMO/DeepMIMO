"""Test tutorial 7: Convert & Upload Ray-tracing Dataset."""

import pytest

from .conftest import run_tutorial


@pytest.mark.tutorial
def test_converters():
    """Test that the converters tutorial runs without errors."""
    run_tutorial("7_converters.py")

