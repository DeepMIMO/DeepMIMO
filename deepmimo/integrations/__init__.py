"""DeepMIMO integrations module.

This module provides integrations with external tools and platforms:
- Sionna adapter for compatibility with NVIDIA Sionna
- Web export for DeepMIMO web visualizer
"""

from .web import export_dataset_to_binary

# SionnaAdapter is available but not imported by default to avoid circular imports
# Import explicitly: from deepmimo.integrations.sionna_adapter import SionnaAdapter

__all__ = [
    "export_dataset_to_binary",
]
