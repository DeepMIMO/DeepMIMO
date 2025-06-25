"""
Sionna Ray Tracing converter module for DeepMIMO.

This module provides functionality for converting Sionna ray tracing data
into the DeepMIMO format.

Note: This functionality requires additional dependencies.
Install them using: pip install 'deepmimo[sionna1]' or 'deepmimo[sionna019]'
"""

from .sionna_exporter import sionna_exporter

__all__ = ['sionna_exporter'] 