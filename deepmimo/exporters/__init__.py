"""
DeepMIMO exporters module.

This module provides functionality for exporting data to different formats.
Each exporter has its own dependencies which can be installed separately:

- AODT exporter: pip install 'deepmimo[aodt]'
- Sionna exporter: pip install 'deepmimo[sionna]'
"""

# Import the functions but don't execute the imports until needed
def __getattr__(name):
    if name == 'aodt_exporter':
        from .aodt_exporter import aodt_exporter as _func
        globals()[name] = _func  # Cache the function in the module's namespace
        return _func
    elif name == 'sionna_exporter':
        from .sionna_exporter import sionna_exporter as _func
        globals()[name] = _func  # Cache the function in the module's namespace
        return _func
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['aodt_exporter', 'sionna_exporter']
