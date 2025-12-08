"""Tests for DeepMIMO Array Wrapper."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from deepmimo.generator.array_wrapper import DeepMIMOArray


def test_array_creation():
    """Test creation of DeepMIMOArray."""
    mock_dataset = MagicMock()
    data = np.array([1, 2, 3])

    dm_array = DeepMIMOArray(data, mock_dataset, "test_array")

    assert isinstance(dm_array, DeepMIMOArray)
    assert isinstance(dm_array, np.ndarray)
    assert np.array_equal(dm_array, data)
    assert dm_array.dataset is mock_dataset
    assert dm_array.name == "test_array"


def test_array_slicing():
    """Test that slicing preserves metadata."""
    mock_dataset = MagicMock()
    data = np.array([1, 2, 3, 4])
    dm_array = DeepMIMOArray(data, mock_dataset, "test_array")

    sliced = dm_array[0:2]

    assert isinstance(sliced, DeepMIMOArray)
    assert np.array_equal(sliced, np.array([1, 2]))
    assert sliced.dataset is mock_dataset
    assert sliced.name == "test_array"


def test_plot_1d():
    """Test plotting 1D array."""
    mock_dataset = MagicMock()
    data = np.array([1, 2, 3])
    dm_array = DeepMIMOArray(data, mock_dataset, "power")  # 'power' has a known title

    dm_array.plot()

    mock_dataset.plot_coverage.assert_called_once()
    args, kwargs = mock_dataset.plot_coverage.call_args
    assert args[0] is dm_array  # Should pass self
    assert kwargs["cbar_title"] == "Power (dBW)"


def test_plot_2d():
    """Test plotting 2D array."""
    mock_dataset = MagicMock()
    data = np.array([[1, 2], [3, 4]])  # [2, 2]
    dm_array = DeepMIMOArray(data, mock_dataset, "phase")

    dm_array.plot(path_idx=1)

    mock_dataset.plot_coverage.assert_called_once()
    args, kwargs = mock_dataset.plot_coverage.call_args
    # args[0] should be slice
    np.testing.assert_array_equal(args[0], data[:, 1])
    assert kwargs["cbar_title"] == "Phase (deg)"


def test_plot_3d():
    """Test plotting 3D array."""
    mock_dataset = MagicMock()
    data = np.zeros((2, 2, 2))
    data[0, 1, 1] = 99
    dm_array = DeepMIMOArray(data, mock_dataset, "custom")

    dm_array.plot(path_idx=1, interaction_idx=1)

    mock_dataset.plot_coverage.assert_called_once()
    args, kwargs = mock_dataset.plot_coverage.call_args
    np.testing.assert_array_equal(args[0], data[:, 1, 1])
    assert "cbar_title" not in kwargs  # Custom name has no default title


def test_plot_invalid_dim():
    """Test plotting with invalid dimensions."""
    mock_dataset = MagicMock()
    data = np.zeros((2, 2, 2, 2))
    dm_array = DeepMIMOArray(data, mock_dataset, "test")

    with pytest.raises(ValueError):
        dm_array.plot()
