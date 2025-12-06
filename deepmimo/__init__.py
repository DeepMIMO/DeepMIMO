"""DeepMIMO Python Package."""

__version__ = "4.0.0b11"

# Core functionality
# Import immediate modules
from . import consts, general_utils
from .api import (
    download,
    search,
    upload,
    upload_images,
    upload_rt_source,
)

# Import the config instance
from .config import config
from .converters.converter import convert
from .general_utils import (
    get_available_scenarios,
    get_params_path,
    get_rt_source_folder,
    get_rt_sources_dir,
    get_scenario_folder,
    load_dict_from_json,
    unzip,
    zip,
)

# Channel parameters
from .generator.channel import ChannelParameters
from .generator.core import (
    generate,
    load,
)
from .generator.dataset import Dataset, DynamicDataset, MacroDataset
from .generator.geometry import (
    steering_vec,
)

# Visualization
from .generator.visualization import (
    plot_coverage,
    plot_power_discarding,
    plot_rays,
)
from .info import info

# Materials
from .materials import (
    Material,
    MaterialList,
)

# Physical world representation
from .scene import Face, PhysicalElement, PhysicalElementGroup, Scene
from .summary import plot_summary, summary

# TX/RX handling
from .txrx import (
    TxRxPair,
    TxRxSet,
    get_txrx_pairs,
    get_txrx_sets,
    print_available_txrx_pair_ids,
)

__all__ = [
    # Core functionality
    "generate",
    "convert",
    "info",
    "load",
    "Dataset",
    "MacroDataset",
    "DynamicDataset",
    "ChannelParameters",
    # TX/RX handling
    "TxRxSet",
    "TxRxPair",
    "get_txrx_sets",
    "get_txrx_pairs",
    "print_available_txrx_pair_ids",
    # Visualization
    "plot_coverage",
    "plot_rays",
    "plot_power_discarding",
    # Physical world representation
    "Face",
    "PhysicalElement",
    "PhysicalElementGroup",
    "Scene",
    # Materials
    "Material",
    "MaterialList",
    # General utilities
    "summary",
    "plot_summary",
    # Database API
    "upload",
    "upload_rt_source",
    "upload_images",
    "download",
    "search",
    # Scenario management utils
    "get_available_scenarios",
    "get_params_path",
    "get_scenario_folder",
    "get_rt_sources_dir",
    "get_rt_source_folder",
    "load_dict_from_json",
    # Constants and configuration
    "consts",
    "general_utils",
    "config",
    # Zip/unzip
    "zip",
    "unzip",
    # Beamforming utils
    "steering_vec",
]
