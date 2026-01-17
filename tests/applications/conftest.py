"""Shared fixtures and utilities for application tests."""

import pathlib
import runpy
import warnings

import matplotlib as mpl
import pytest

# Set matplotlib to non-interactive backend before any applications import it
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


def run_application(application_name: str) -> None:
    """Execute an application Python file.

    Args:
        application_name: Name of the application file (e.g., "1_channel_prediction.py")

    """
    applications_dir = pathlib.Path(__file__).parent.parent.parent / "docs" / "applications"
    application_path = applications_dir / application_name

    if not application_path.exists():
        msg = f"Application not found: {application_path}"
        raise FileNotFoundError(msg)

    runpy.run_path(str(application_path), run_name="__main__")
