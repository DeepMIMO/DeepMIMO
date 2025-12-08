"""Test that all tutorial Python files execute without errors."""

import pathlib
import runpy

import pytest

# Set matplotlib to non-interactive backend before any tutorials import it
import matplotlib
matplotlib.use('Agg')

TUTORIALS_DIR = pathlib.Path(__file__).parent.parent / "docs" / "tutorials"


@pytest.mark.parametrize("path", sorted(TUTORIALS_DIR.glob("*.py")))
def test_tutorial_runs_without_error(path):
    """Execute each tutorial .py file and ensure no exceptions are raised.
    
    Uses Agg backend for matplotlib to avoid display issues during testing.
    """
    runpy.run_path(str(path), run_name="__main__")

