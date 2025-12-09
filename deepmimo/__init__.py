"""DeepMIMO Python Package."""

__version__ = "4.0.0b11"

# Core functionality
# Import immediate modules
from . import consts, general_utils

# API imports - using root api.py until api/ folder is fully migrated
from .api import (
    download,
    search,
    upload,
    upload_images,
    upload_rt_source,
)

# Import the config instance
from .config import config

# Converters
from .converters.converter import convert

# Core models (moved from root to core/)
from .core.materials import Material, MaterialList
from .core.rt_params import RayTracingParameters
from .core.scene import BoundingBox, Face, PhysicalElement, PhysicalElementGroup, Scene
from .core.txrx import (
    TxRxPair,
    TxRxSet,
    get_txrx_pairs,
    get_txrx_sets,
    print_available_txrx_pair_ids,
)

# Datasets (moved from generator/ to datasets/)
from .datasets.dataset import Dataset, DynamicDataset, MacroDataset
from .datasets.generate import generate
from .datasets.load import load

# Summary (moved to datasets/)
from .datasets.summary import plot_summary, summary
from .datasets.visualization import (
    plot_coverage,
    plot_power_discarding,
    plot_rays,
)

# General utilities
from .general_utils import (
    get_available_scenarios,
    get_params_path,
    get_rt_source_folder,
    get_rt_sources_dir,
    get_scenario_folder,
    load_dict_from_json,
    unzip,
    zip,  # noqa: A004
)

# Channel parameters (still in generator/)
from .generator.channel import ChannelParameters
from .generator.geometry import steering_vec

# Info
from .info import info

# Backward compatibility - re-export web_export
from .integrations import web as web_export

# Integrations
from .integrations.web import export_dataset_to_binary

__all__ = [
    # Bounding Box
    "BoundingBox",
    # Channel Parameters
    "ChannelParameters",
    # Datasets
    "Dataset",
    "DynamicDataset",
    # Physical world representation
    "Face",
    "MacroDataset",
    # Materials
    "Material",
    "MaterialList",
    "PhysicalElement",
    "PhysicalElementGroup",
    # Ray Tracing Parameters
    "RayTracingParameters",
    "Scene",
    # TX/RX handling
    "TxRxPair",
    "TxRxSet",
    "config",
    # Constants and configuration
    "consts",
    # Converters
    "convert",
    # Database API
    "download",
    # Integrations
    "export_dataset_to_binary",
    "general_utils",
    # Core functionality
    "generate",
    # Scenario management utils
    "get_available_scenarios",
    "get_params_path",
    "get_rt_source_folder",
    "get_rt_sources_dir",
    "get_scenario_folder",
    "get_txrx_pairs",
    "get_txrx_sets",
    "info",
    "load",
    "load_dict_from_json",
    # Visualization
    "plot_coverage",
    "plot_power_discarding",
    "plot_rays",
    "plot_summary",
    "print_available_txrx_pair_ids",
    "search",
    # Beamforming utils
    "steering_vec",
    # General utilities
    "summary",
    "unzip",
    # Database API
    "upload",
    "upload_images",
    "upload_rt_source",
    # Backward compatibility
    "web_export",
    # Zip/unzip
    "zip",
]
