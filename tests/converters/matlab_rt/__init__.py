# ruff: noqa: TC003
"""Tests for the MATLAB RT converter MVP."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def expect_raises(expected: type[BaseException]) -> Iterator[None]:
    """Assert that a block raises the expected exception type."""
    try:
        yield
    except expected:
        return

    message = "Expected exception was not raised."
    raise AssertionError(message)
