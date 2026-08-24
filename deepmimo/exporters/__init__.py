"""DeepMIMO exporters module.

This module provides functionality for exporting data to different formats.
Each exporter has its own dependencies which can be installed separately:

- AODT exporter: pip install 'deepmimo[aodt]'
- Sionna exporter: pip install 'deepmimo[sionna]'
- Mitsuba exporter: no extra dependencies; turns a scenario back into a scene
"""

# Import the modules but don't execute the imports until needed
import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "aodt_exporter":
        _module = importlib.import_module(".aodt_exporter", package=__name__)
        globals()[name] = _module  # Cache the module in the namespace
        return _module
    if name == "mitsuba_exporter":
        _module = importlib.import_module(".mitsuba_exporter", package=__name__)
        globals()[name] = _module  # Cache the module in the namespace
        return _module
    if name == "sionna_exporter":
        _module = importlib.import_module(".sionna_exporter", package=__name__)
        globals()[name] = _module  # Cache the module in the namespace
        return _module
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = ["aodt_exporter", "mitsuba_exporter", "sionna_exporter"]
