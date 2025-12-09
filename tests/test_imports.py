"""Smoke tests for importing core DeepMIMO modules."""

import importlib

import pytest

# Excluding anything under deepmimo.pipelines.*
MODULES = [
    "deepmimo.api",
    "deepmimo.web_export",
    "deepmimo.txrx",
    "deepmimo.summary",
    "deepmimo.scene",
    "deepmimo.rt_params",
    "deepmimo.materials",
    "deepmimo.integrations.sionna_adapter",
    "deepmimo.info",
    "deepmimo.generator.visualization",
    "deepmimo.generator.geometry",
    "deepmimo.generator.generator_utils",
    "deepmimo.generator.dataset",
    "deepmimo.generator.core",
    "deepmimo.generator.channel",
    "deepmimo.generator.array_wrapper",
    "deepmimo.generator.ant_patterns",
    "deepmimo.general_utils",
    "deepmimo.exporters.sionna_exporter",
    "deepmimo.exporters.aodt_exporter",
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
    "deepmimo.consts",
    "deepmimo.config",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import_module(module_name: str) -> None:
    """Import a module and skip if optional dependencies are missing."""
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        pytest.skip(f"Optional dependency missing while importing {module_name}: {e}")
