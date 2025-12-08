"""Test tutorial 8: Migration Guide."""

import pytest

from .conftest import run_tutorial


@pytest.mark.tutorial
def test_migration_guide():
    """Test that the migration guide tutorial runs without errors."""
    run_tutorial("8_migration_guide.py")

