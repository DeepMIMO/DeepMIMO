"""
DeepMIMO Python Package.
"""

__version__ = "4.0.0b11"

# Core functionality
from .generator.core import (
    generate,
    load,
)
from .dataset import Dataset, MacroDataset, DynamicDataset

# TX/RX handling
from .txrx import (
    TxRxSet,
    TxRxPair,
    get_txrx_sets,
    get_txrx_pairs,
    print_available_txrx_pair_ids,
)

# Visualization
from .visualization import (
    plot_coverage,
    plot_rays,
    plot_power_discarding,
)

from .generator.geometry import (
    steering_vec,
)

# Channel parameters
from .generator.channel import ChannelParameters

from .converters.converter import convert
from .info import info
from .general_utils import (
    get_available_scenarios,
    get_params_path,
    get_scenario_folder,
    load_dict_from_json,
    zip,
    unzip,
)

from .summary import summary, plot_summary

from .api import (
    upload,
    upload_rt_source,
    download,
    search,
    upload_images,
)

# Physical world representation
from .scene import (
    Face,
    PhysicalElement,
    PhysicalElementGroup,
    Scene
)

# Materials
from .materials import (
    Material,
    MaterialList,
)

# Import immediate modules
from . import consts
from . import general_utils

# Import the config instance
from .config import config

__all__ = [
    # Core functionality
    'generate',
    'convert', 
    'info',
    'load',
    'Dataset',
    'MacroDataset',
    'DynamicDataset',
    'ChannelParameters',
    
    # TX/RX handling
    'TxRxSet',
    'TxRxPair',
    'get_txrx_sets',
    'get_txrx_pairs',
    'print_available_txrx_pair_ids',
    
    # Visualization
    'plot_coverage',
    'plot_rays',
    'plot_power_discarding',

    # Physical world representation
    'Face',
    'PhysicalElement',
    'PhysicalElementGroup',
    'Scene',
    
    # Materials
    'Material',
    'MaterialList',
    
    # General utilities
    'summary',
    'plot_summary',

    # Database API
    'upload',
    'upload_rt_source',
    'upload_images',
    'download',
    'search',

    # Scenario management utils
    'get_available_scenarios',
    'get_params_path',
    'get_scenario_folder',
    'load_dict_from_json',
    
    # Constants and configuration
    'consts',
    'general_utils',
    'config',

    # Zip/unzip
    'zip',
    'unzip',

    # Beamforming utils
    'steering_vec',
]
