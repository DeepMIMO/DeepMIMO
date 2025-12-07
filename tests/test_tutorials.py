"""Test that all tutorial Python files execute without errors."""

import pathlib
import runpy

import pytest

TUTORIALS_DIR = pathlib.Path(__file__).parent.parent / "docs" / "tutorials"


@pytest.mark.parametrize("path", sorted(TUTORIALS_DIR.glob("*.py")))
def test_tutorial_runs_without_error(path):
    """Execute each tutorial .py file and ensure no exceptions are raised."""
    runpy.run_path(str(path), run_name="__main__")

