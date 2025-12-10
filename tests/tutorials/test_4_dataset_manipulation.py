"""Test tutorial 4: User Selection and Dataset Manipulation."""

import pytest

from .conftest import run_tutorial


@pytest.mark.tutorial
def test_dataset_manipulation() -> None:
    """Test that the dataset manipulation tutorial runs without errors."""
    run_tutorial("4_dataset_manipulation.py")
