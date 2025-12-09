"""Smoke tests for importing core DeepMIMO modules."""

import importlib

import pytest

# Excluding anything under deepmimo.pipelines.*
MODULES = [
    # API
    "deepmimo.api",
    # Core models
    "deepmimo.core.materials",
    "deepmimo.core.rt_params",
    "deepmimo.core.scene",
    "deepmimo.core.txrx",
    # Datasets
    "deepmimo.datasets.dataset",
    "deepmimo.datasets.array_wrapper",
    "deepmimo.datasets.visualization",
    "deepmimo.datasets.sampling",
    "deepmimo.datasets.summary",
    "deepmimo.datasets.generate",
    "deepmimo.datasets.load",
    # Generator (streamlined)
    "deepmimo.generator.channel",
    "deepmimo.generator.geometry",
    "deepmimo.generator.ant_patterns",
    # Integrations
    "deepmimo.integrations.web",
    "deepmimo.integrations.sionna_adapter",
    # Utilities
    "deepmimo.general_utils",
    "deepmimo.info",
    # Config
    "deepmimo.config",
    "deepmimo.consts",
    # Exporters
    "deepmimo.exporters.sionna_exporter",
    "deepmimo.exporters.aodt_exporter",
    # Converters
    "deepmimo.converters.wireless_insite.xml_parser",
    "deepmimo.converters.wireless_insite.setup_parser",
    "deepmimo.converters.wireless_insite.p2m_parser",
    "deepmimo.converters.wireless_insite.insite_txrx",
    "deepmimo.converters.wireless_insite.insite_scene",
    "deepmimo.converters.wireless_insite.insite_rt_params",
    "deepmimo.converters.wireless_insite.insite_paths",
    "deepmimo.converters.wireless_insite.insite_materials",
    "deepmimo.converters.wireless_insite.insite_converter",
    "deepmimo.converters.sionna_rt.sionna_txrx",
    "deepmimo.converters.sionna_rt.sionna_scene",
    "deepmimo.converters.sionna_rt.sionna_rt_params",
    "deepmimo.converters.sionna_rt.sionna_paths",
    "deepmimo.converters.sionna_rt.sionna_converter",
    "deepmimo.converters.sionna_rt.sionna_materials",
    "deepmimo.converters.converter_utils",
    "deepmimo.converters.converter",
    "deepmimo.converters.aodt.safe_import",
    "deepmimo.converters.aodt.aodt_utils",
    "deepmimo.converters.aodt.aodt_txrx",
    "deepmimo.converters.aodt.aodt_rt_params",
    "deepmimo.converters.aodt.aodt_scene",
    "deepmimo.converters.aodt.aodt_paths",
    "deepmimo.converters.aodt.aodt_materials",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import_module(module_name: str) -> None:
    """Import a module and skip if optional dependencies are missing."""
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        pytest.skip(f"Optional dependency missing while importing {module_name}: {e}")
