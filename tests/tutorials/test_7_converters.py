"""Test tutorial 7: Convert & Upload Ray-tracing Dataset."""

from .conftest import run_tutorial


def test_converters():
    """Test that the converters tutorial runs without errors."""
    run_tutorial("7_converters.py")

