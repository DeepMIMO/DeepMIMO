"""DeepMIMO exporters module.

This module provides functionality for exporting data to different formats.
Each exporter has its own dependencies which can be installed separately:

- AODT exporter: pip install 'deepmimo[aodt]'
- Sionna exporter: pip install 'deepmimo[sionna]'
"""


# Import the modules but don't execute the imports until needed
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "aodt_exporter":
        import importlib

        _module = importlib.import_module(".aodt_exporter", package=__name__)
        globals()[name] = _module  # Cache the module in the namespace
        return _module
    if name == "sionna_exporter":
        import importlib

        _module = importlib.import_module(".sionna_exporter", package=__name__)
        globals()[name] = _module  # Cache the module in the namespace
        return _module
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = ["aodt_exporter", "sionna_exporter"]
