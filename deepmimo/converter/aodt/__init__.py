"""
AODT (Aerial Omniverse Digital Twin) module for DeepMIMO.

This module provides tools for exporting data to parquet format.
Note: This functionality requires additional dependencies.
Install them using: pip install 'deepmimo[aodt]'
"""

from .aodt_exporter import aodt_exporter

__all__ = ['aodt_exporter'] 