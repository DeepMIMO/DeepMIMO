"""
Dataset module for DeepMIMO.

This module provides classes for managing DeepMIMO datasets:

Dataset: For managing individual DeepMIMO datasets, including:
- Channel matrices 
- Path information (angles, powers, delays)
- Position information
- TX/RX configuration information
- Metadata

MacroDataset: For managing collections of related DeepMIMO datasets that *may* share:
- Scene configuration
- Material properties
- Loading parameters 
- Ray-tracing parameters

DynamicDataset: For dynamic datasets that consist of multiple (macro)datasets across time snapshots:
- All txrx sets are the same for all time snapshots

Load functions: For loading and generating datasets from ray-tracing data
"""

from .dataset import Dataset
from .macro_dataset import MacroDataset
from .dynamic_dataset import DynamicDataset
from .load import load, _load_dataset, _load_raytracing_scene, _load_tx_rx_raydata
from .array_wrapper import DeepMIMOArray
from .sampling import get_linear_idxs, get_uniform_idxs, get_grid_idxs, get_idxs_with_limits

__all__ = [
    'Dataset',
    'MacroDataset', 
    'DynamicDataset',
    'DeepMIMOArray',
    'load',
    '_load_dataset',
    '_load_raytracing_scene',
    '_load_tx_rx_raydata',
    'get_linear_idxs',
    'get_uniform_idxs',
    'get_grid_idxs',
    'get_idxs_with_limits'
] 