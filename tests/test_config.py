"""Tests for DeepMIMO configuration module."""

import pytest
from deepmimo import config

@pytest.fixture
def reset_config():
    """Reset configuration before and after each test."""
    config.reset()
    yield
    config.reset()

def test_singleton(reset_config):
    """Test that config is a singleton."""
    from deepmimo.config import DeepMIMOConfig
    c1 = DeepMIMOConfig()
    c2 = DeepMIMOConfig()
    assert c1 is c2

def test_initial_values(reset_config):
    """Test default configuration values."""
    assert config.get("use_gpu") is False
    assert config.get("gpu_device_id") == 0
    assert config.get("scenarios_folder") == "deepmimo_scenarios"
    assert config.get("rt_sources_folder") == "deepmimo_rt_sources"

def test_set_get(reset_config):
    """Test setting and getting configuration values."""
    config.set("test_key", "test_value")
    assert config.get("test_key") == "test_value"
    
    config.set("use_gpu", True)
    assert config.get("use_gpu") is True

def test_get_default(reset_config):
    """Test getting non-existent key with default."""
    assert config.get("non_existent", "default") == "default"
    assert config.get("non_existent") is None

def test_call_interface(reset_config):
    """Test function-like interface."""
    # Set using kwargs
    config(test_key="test_value", use_gpu=True)
    assert config.get("test_key") == "test_value"
    assert config.get("use_gpu") is True
    
    # Set using positional args
    config("another_key", 123)
    assert config.get("another_key") == 123
    
    # Get using positional arg
    assert config("another_key") == 123
    
    # Error mixing args and kwargs
    with pytest.raises(ValueError):
        config("key", "value", other="kwarg")

def test_reset(reset_config):
    """Test resetting configuration."""
    config.set("use_gpu", True)
    config.set("new_param", 123)
    
    config.reset()
    
    assert config.get("use_gpu") is False
    assert config.get("new_param") is None

def test_get_all(reset_config):
    """Test getting all configuration values."""
    all_config = config.get_all()
    assert isinstance(all_config, dict)
    assert "use_gpu" in all_config
    assert "scenarios_folder" in all_config
    
    # Ensure it's a copy
    all_config["use_gpu"] = True
    assert config.get("use_gpu") is False

def test_repr(reset_config):
    """Test string representation."""
    repr_str = repr(config)
    assert "DeepMIMO Configuration:" in repr_str
    assert "use_gpu: False" in repr_str

def test_print_config(reset_config, capsys):
    """Test print_config method."""
    config.print_config()
    captured = capsys.readouterr()
    assert "DeepMIMO Configuration:" in captured.out
    assert "use_gpu" in captured.out

def test_call_no_args(reset_config, capsys):
    """Test calling config with no arguments prints configuration."""
    config()  # Should print config
    captured = capsys.readouterr()
    assert "DeepMIMO Configuration:" in captured.out
    assert "use_gpu" in captured.out

