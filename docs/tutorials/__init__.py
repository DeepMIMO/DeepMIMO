"""Tutorials package for DeepMIMO documentation."""

from __future__ import annotations

import os
import warnings


def on_startup(*, command: str, dirty: bool) -> None:  # noqa: ARG001
    """MkDocs hook: quieter notebook execution for static HTML export."""
    # MkDocs + nbconvert: tqdm refreshes become one line per update in static HTML.
    os.environ["TQDM_DISABLE"] = "1"
    # New Python kernels inherit this; suppresses warnings in executed notebook output.
    os.environ["PYTHONWARNINGS"] = "ignore"
    # Same-process execution (if any) still respects the filters registry.
    warnings.filterwarnings("ignore")
