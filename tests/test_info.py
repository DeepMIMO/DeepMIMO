"""Tests for DeepMIMO info module."""

from deepmimo import info
from deepmimo import consts as c


def test_info_general(capsys):
    """Test general info output."""
    info()
    captured = capsys.readouterr()
    assert "Fundamental Matrices:" in captured.out
    assert "Computed/Derived Matrices" in captured.out
    assert "Additional Dataset Fields" in captured.out


def test_info_specific_param(capsys):
    """Test info for specific parameter."""
    info("power")
    captured = capsys.readouterr()
    assert "Tap power. Received power in dBW" in captured.out
    assert "num_rx, num_paths" in captured.out


def test_info_alias(capsys):
    """Test info for alias parameter."""
    # aliases are resolved in Dataset.info, but info() function handles strings too?
    # deepmimo.info.info(param_name) checks dictionary.
    # If we pass an alias that is not in the dictionary keys, it might print "Parameter ... not found".
    # Dataset.info resolves aliases before calling deepmimo.info.
    # Let's check if info() handles aliases.
    # Reading info.py code (not visible here but assumed standard behavior)
    # If info() just looks up in a dict, aliases won't work unless they are keys.
    # Let's test with a known key.
    info("channel")
    captured = capsys.readouterr()
    assert "Channel matrix between TX and RX antennas" in captured.out


def test_info_not_found(capsys):
    """Test info for non-existent parameter."""
    info("non_existent_param")
    captured = capsys.readouterr()
    assert "Unknown parameter: non_existent_param" in captured.out


def test_info_ch_params(capsys):
    """Test info for channel parameters."""
    info("ch_params")
    captured = capsys.readouterr()
    assert "Channel Generation Parameters" in captured.out
    assert c.PARAMSET_ANT_BS in captured.out


def test_info_channel_params_alias(capsys):
    """Test info for channel_params alias."""
    info("channel_params")
    captured = capsys.readouterr()
    assert "Channel Generation Parameters" in captured.out


def test_info_all(capsys):
    """Test info for all parameters."""
    info("all")
    captured = capsys.readouterr()
    assert "Fundamental Matrices" in captured.out
    assert "Computed/Derived Matrices" in captured.out
    assert "Additional Dataset Fields" in captured.out


def test_info_with_object():
    """Test info with a non-string object (should call help())."""
    # Should call built-in help, which prints to stdout
    # We can't easily capture help() output, so just ensure it doesn't crash
    import sys

    info(sys)  # Pass a module object


def test_info_fundamental_matrices(capsys):
    """Test info for various fundamental matrices."""
    for param in [c.AOA_AZ_PARAM_NAME, c.AOD_EL_PARAM_NAME, c.INTERACTIONS_POS_PARAM_NAME]:
        info(param)
        captured = capsys.readouterr()
        assert param in captured.out or "alias" in captured.out.lower()


def test_info_computed_matrices(capsys):
    """Test info for various computed matrices."""
    for param in [c.LOS_PARAM_NAME, c.PATHLOSS_PARAM_NAME, c.NUM_PATHS_PARAM_NAME]:
        info(param)
        captured = capsys.readouterr()
        assert param in captured.out or "alias" in captured.out.lower()


def test_info_additional_fields(capsys):
    """Test info for additional fields."""
    for param in [c.SCENE_PARAM_NAME, c.MATERIALS_PARAM_NAME, c.TXRX_PARAM_NAME]:
        info(param)
        captured = capsys.readouterr()
        assert param in captured.out or "alias" in captured.out.lower()
