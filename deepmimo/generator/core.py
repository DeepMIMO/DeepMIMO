"""
DeepMIMO Core Generation Module.

This module provides the core functionality for generating and managing DeepMIMO datasets.
It handles:
- Dataset generation and scenario management
- Ray-tracing data loading and processing
- Channel computation and parameter validation
- Multi-user MIMO channel generation

The module serves as the main entry point for creating DeepMIMO datasets from ray-tracing data.
"""

# Standard library imports
import os
from typing import Dict, List, Any

# Third-party imports
import numpy as np

# Local imports
from .. import consts as c
from ..general_utils import (get_mat_filename, load_dict_from_json, 
                             get_scenario_folder, get_params_path, load_mat, DotDict)
from ..scene import Scene
from ..dataset import Dataset, MacroDataset, DynamicDataset
from ..materials import MaterialList

# Channel generation
from .channel import ChannelParameters

# Scenario management
from ..api import download

# Import load function from dataset module
from ..dataset.load import load

__all__ = [
    'load',
    'generate'
]

def generate(scen_name: str, load_params: Dict[str, Any] = {},
            ch_gen_params: Dict[str, Any] = {}) -> Dataset:
    """Generate a DeepMIMO dataset for a given scenario.
    
    This function wraps loading scenario data, computing channels, and organizing results.

    Args:
        scen_name (str): Name of the scenario to generate data for
        load_params (dict): Parameters for loading the scenario. Defaults to {}.
        ch_gen_params (dict): Parameters for channel generation. Defaults to {}.

    Returns:
        Dataset: Generated DeepMIMO dataset containing channel matrices and metadata
        
    Raises:
        ValueError: If scenario name is invalid or required files are missing
    """
    dataset = load(scen_name, **load_params)
    
    # Create channel generation parameters
    ch_params = ch_gen_params if ch_gen_params else ChannelParameters()
    
    # Compute channels - will be propagated to all child datasets if MacroDataset
    _ = dataset.compute_channels(ch_params)

    return dataset

