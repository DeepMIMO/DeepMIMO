"""Shared fixtures and utilities for tutorial tests."""

import pathlib
import runpy
import warnings

import matplotlib as mpl
import pytest

# Set matplotlib to non-interactive backend before any tutorials import it
mpl.use("Agg")

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


# Close all figures after each test to avoid memory warnings
@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    mpl.pyplot.close("all")


def run_tutorial(tutorial_name: str) -> None:
    """Execute a tutorial Python file.

    Args:
        tutorial_name: Name of the tutorial file (e.g., "1_getting_started.py")

    """
    tutorials_dir = pathlib.Path(__file__).parent.parent.parent / "docs" / "tutorials"
    tutorial_path = tutorials_dir / tutorial_name

    if not tutorial_path.exists():
        msg = f"Tutorial not found: {tutorial_path}"
        raise FileNotFoundError(msg)

    runpy.run_path(str(tutorial_path), run_name="__main__")
