"""DeepMIMO Datasets Module.

This module provides dataset classes and operations:
- Dataset, MacroDataset, DynamicDataset classes
- Dataset loading and generation functions
- Dataset visualization and plotting
- Dataset sampling utilities
- Dataset summarization
"""

from .array_wrapper import DeepMIMOArray
from .dataset import Dataset, DynamicDataset, MacroDataset
from .generate import generate
from .load import load, validate_txrx_sets
from .sampling import (
    dbw2watt,
    get_grid_idxs,
    get_idxs_with_limits,
    get_linear_idxs,
    get_uniform_idxs,
)
from .summary import plot_summary, summary
from .visualization import plot_coverage, plot_power_discarding, plot_rays

__all__ = [
    # Core dataset classes
    "Dataset",
    # Array wrapper
    "DeepMIMOArray",
    "DynamicDataset",
    "MacroDataset",
    "dbw2watt",
    # Dataset operations
    "generate",
    "get_grid_idxs",
    "get_idxs_with_limits",
    # Sampling utilities
    "get_linear_idxs",
    "get_uniform_idxs",
    "load",
    # Visualization
    "plot_coverage",
    "plot_power_discarding",
    "plot_rays",
    "plot_summary",
    # Summary
    "summary",
    # Validation
    "validate_txrx_sets",
]

