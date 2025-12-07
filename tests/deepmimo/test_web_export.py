"""Tests for DeepMIMO Web Export module."""

import pytest
import numpy as np
import os
from pathlib import Path
from deepmimo.web_export import export_dataset_to_binary, _save_binary_array
from deepmimo.generator.dataset import Dataset, MacroDataset

class MockDataset:
    """Mock dataset for testing export."""
    def __init__(self):
        self.rx_pos = np.array([[0,0,0], [10,0,0], [0,10,0]], dtype=np.float32)
        self.tx_pos = np.array([5,5,5], dtype=np.float32)
        self.inter = np.ones((3, 2), dtype=np.float32)
        self.inter_pos = np.zeros((3, 2, 3, 3), dtype=np.float32)
        self.power = np.zeros((3, 2), dtype=np.float32)
        # self.datasets = None # Not a macro dataset (removed to avoid confusion in web_export)

def test_save_binary_array(tmp_path):
    """Test saving array to binary."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    file_path = tmp_path / "test.bin"
    _save_binary_array(arr, file_path)
    
    assert file_path.exists()
    assert file_path.stat().st_size > 0

def test_export_single_dataset(tmp_path):
    """Test exporting a single dataset."""
    ds = MockDataset()
    # Mock txrx attribute which is usually a dictionary in export logic
    # Wait, _process_single_dataset_to_binary takes tx_set_id, rx_set_id arguments directly.
    # export_dataset_to_binary handles calling it.
    # For single dataset, it calls with 1, 1.
    
    dataset_name = "test_scen"
    export_dataset_to_binary(ds, dataset_name, str(tmp_path))
    
    out_dir = tmp_path / dataset_name
    assert out_dir.exists()
    assert (out_dir / "metadata.bin").exists()
    assert (out_dir / "rx_pos_tx_1_rx_1.bin").exists()

def test_export_macro_dataset(tmp_path):
    """Test exporting a macro dataset."""
    ds1 = MockDataset()
    # For macro dataset, export expects dataset.datasets list
    # and each dataset to have dataset['txrx']['tx_set_id'] etc.
    
    # We need to mock dictionary access for MockDataset or use DotDict
    class MockDsWithDict(MockDataset):
        def __getitem__(self, key):
            if key == 'txrx':
                return {'tx_set_id': 0, 'rx_set_id': 1}
            return getattr(self, key)
            
    ds1 = MockDsWithDict()
    
    macro_ds = MacroDataset()
    macro_ds.datasets = [ds1]
    
    dataset_name = "macro_scen"
    export_dataset_to_binary(macro_ds, dataset_name, str(tmp_path))
    
    out_dir = tmp_path / dataset_name
    assert out_dir.exists()
    assert (out_dir / "metadata.bin").exists()
    assert (out_dir / "rx_pos_tx_0_rx_1.bin").exists()

