"""Tests for DeepMIMO constants."""


from deepmimo import consts


def test_consts_version() -> None:
    """Test version constant exists."""
    assert hasattr(consts, "VERSION")
    assert isinstance(consts.VERSION, str)


def test_consts_files() -> None:
    """Test file path constants."""
    assert consts.SCENARIOS_FOLDER == "deepmimo_scenarios"
    assert consts.RT_SOURCES_FOLDER == "deepmimo_rt_sources"
    assert consts.PARAMS_FILENAME == "params"


def test_interaction_codes() -> None:
    """Test interaction codes."""
    assert consts.INTERACTION_LOS == 0
    assert consts.INTERACTION_REFLECTION == 1
    assert consts.INTERACTION_DIFFRACTION == 2
    assert consts.INTERACTION_SCATTERING == 3
    assert consts.INTERACTION_TRANSMISSION == 4


def test_raytracer_names() -> None:
    """Test raytracer name constants."""
    assert consts.RAYTRACER_NAME_WIRELESS_INSITE == "Remcom Wireless Insite"
    assert consts.RAYTRACER_NAME_SIONNA == "Sionna Ray Tracing"
    assert consts.RAYTRACER_NAME_AODT == "Aerial Omniverse Digital Twin"


def test_dataset_aliases() -> None:
    """Test dataset aliases dictionary."""
    assert isinstance(consts.DATASET_ALIASES, dict)
    assert "ch" in consts.DATASET_ALIASES
    assert consts.DATASET_ALIASES["ch"] == consts.CHANNEL_PARAM_NAME
    assert "pwr" in consts.DATASET_ALIASES
    assert consts.DATASET_ALIASES["pwr"] == consts.POWER_PARAM_NAME
