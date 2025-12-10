"""DeepMIMO Python Package."""

__version__ = "4.0.0b11"

# Core functionality
# Import immediate modules
from . import consts

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

# Core models
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

# Datasets
from .datasets.dataset import Dataset, DynamicDataset, MacroDataset
from .datasets.generate import generate
from .datasets.load import load

# Sampling functions (moved to datasets/)
from .datasets.sampling import (
    dbw2watt,
    get_grid_idxs,
    get_idxs_with_limits,
    get_linear_idxs,
    get_uniform_idxs,
)

# Summary (moved to datasets/)
from .datasets.summary import plot_summary, summary
from .datasets.visualization import (
    plot_coverage,
    plot_power_discarding,
    plot_rays,
)

# Channel generator
from .generator.channel import ChannelParameters
from .generator.geometry import steering_vec

# Re-export web module as web_export
from .integrations import web as web_export

# Integrations
from .integrations.web import export_dataset_to_binary

# Utilities
from .utils import (
    DelegatingList,
    DotDict,
    get_available_scenarios,
    get_mat_filename,
    get_params_path,
    get_rt_source_folder,
    get_rt_sources_dir,
    get_scenario_folder,
    get_scenarios_dir,
    info,
    load_dict_from_json,
    unzip,
    zip,  # noqa: A004
)

__all__ = [
    # Bounding Box
    "BoundingBox",
    # Channel Parameters
    "ChannelParameters",
    # Datasets
    "Dataset",
    # Data structures
    "DelegatingList",
    "DotDict",
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
    # Core functionality
    "generate",
    # Sampling functions
    "dbw2watt",
    "get_grid_idxs",
    "get_idxs_with_limits",
    "get_linear_idxs",
    "get_uniform_idxs",
    # Scenario management utils
    "get_available_scenarios",
    "get_mat_filename",
    "get_params_path",
    "get_rt_source_folder",
    "get_rt_sources_dir",
    "get_scenario_folder",
    "get_scenarios_dir",
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
    "web_export",
    # Zip/unzip
    "zip",
]
