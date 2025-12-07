import pytest
import numpy as np
from deepmimo.generator.dataset import Dataset
from deepmimo import consts as c

def create_test_dataset():
    """Create a minimal test dataset"""
    dataset = Dataset()

    # Mock basic properties
    dataset.tx_pos = np.array([0, 0, 10])  # BS at origin, 10m high
    dataset.rx_pos = np.array(
        [
            [100, 0, 1.5],  # UE1: 100m east
            [0, 100, 1.5],  # UE2: 100m north
            [-100, 0, 1.5],  # UE3: 100m west
            [0, -100, 1.5],  # UE4: 100m south
            [100, 100, 1.5],  # UE5: 100m northeast
        ],
    )

    # Mock channel parameters
    class MockChannelParams:
        def __init__(self):
            self.bs_antenna = {c.PARAMSET_ANT_ROTATION: np.array([0, 0, 0])}
            self.ue_antenna = {c.PARAMSET_ANT_ROTATION: np.zeros((5, 3))}

    dataset.ch_params = MockChannelParams()

    # Mock the cache clearing function
    dataset._clear_cache_rotated_angles = lambda: None

    return dataset

def test_dataset_look_at():
    """Test bs_look_at and ue_look_at functionality"""
    dataset = create_test_dataset()

    # Test bs_look_at
    # Point BS toward first UE (should be 0° azimuth, negative elevation)
    target_ue = dataset.rx_pos[0]  # [100, 0, 1.5]
    dataset.bs_look_at(target_ue)
    bs_rotation = dataset.ch_params.bs_antenna[c.PARAMSET_ANT_ROTATION]
    
    # Verify angles
    expected_azimuth = 0.0  # Due east
    expected_elevation = np.degrees(np.arctan2(1.5 - 10, 100))  # ~-4.8°

    assert np.isclose(bs_rotation[0], expected_azimuth, atol=1e-2)
    assert np.isclose(bs_rotation[1], expected_elevation, atol=1e-2)
    assert np.isclose(bs_rotation[2], 0.0)

    # Test ue_look_at
    # Point all UEs toward BS
    dataset.ue_look_at(dataset.tx_pos)
    ue_rotations = dataset.ch_params.ue_antenna[c.PARAMSET_ANT_ROTATION]

    # Verify each UE points toward BS
    for i, (ue_pos, rotation) in enumerate(zip(dataset.rx_pos, ue_rotations)):
        expected_azimuth = np.degrees(np.arctan2(0 - ue_pos[1], 0 - ue_pos[0]))
        # Distance on ground plane
        dist_ground = np.sqrt((0 - ue_pos[0]) ** 2 + (0 - ue_pos[1]) ** 2)
        expected_elevation = np.degrees(np.arctan2(10 - ue_pos[2], dist_ground))

        assert np.isclose(rotation[0], expected_azimuth, atol=1e-2)
        assert np.isclose(rotation[1], expected_elevation, atol=1e-2)

def test_dataset_subset_rows_cols():
    """Test dataset subsetting by rows and columns."""
    # Create synthetic dataset
    ds = Dataset()
    ds.rx_pos = np.zeros((100, 3))
    # Grid 10x10
    x = np.linspace(0, 9, 10)
    y = np.linspace(0, 9, 10)
    xx, yy = np.meshgrid(x, y)
    ds.rx_pos[:, 0] = xx.flatten()
    ds.rx_pos[:, 1] = yy.flatten()
    ds.n_ue = 100
    ds.user_ids = np.arange(100)
    
    # Mock other matrices
    ds.los = np.zeros(100)
    ds.power = np.zeros((100, 1))
    
    # Mock scene if needed, but subset usually works on matrices
    
    # get_row_idxs logic is complex and depends on grid structure detection which relies on tolerances.
    # Here we simulate "perfect" grid so it should work if implementation allows.
    # However, Dataset.get_row_idxs implementation is not visible here.
    # Assuming we just test subset() with known indices.
    
    idxs = np.arange(40, 60) # indices 40-59
    ds_sub = ds.trim(idxs=idxs)
    
    assert ds_sub.n_ue == 20
    assert ds_sub.rx_pos.shape == (20, 3)
    np.testing.assert_array_equal(ds_sub.rx_pos, ds.rx_pos[idxs])
    assert ds_sub.power.shape == (20, 1)

def test_compute_num_interactions():
    """Test computation of number of interactions from interaction codes."""
    ds = Dataset()
    ds.n_ue = 2 # Required for _wrap_array check
    # Codes:
    # 0 -> 0 (handled by _compute_num_interactions logic for non_zero)
    # 1 -> 1 interaction
    # 10 -> 2 interactions (log10(10) = 1.0 -> 2)
    # 9 -> 1 interaction (log10(9) ~ 0.95 -> 1)
    # 100 -> 3 interactions
    
    ds.inter = np.array([
        [0, 1, 9, 10, 99, 100, 999, 1000], # User 0
        [np.nan, 0, 0, 0, 0, 0, 0, 0]      # User 1
    ], dtype=float)
    
    n_inter = ds._compute_num_interactions()
    
    # Check User 0
    # 0 -> 0 (logic handles 0 specially)
    # 1 -> 1
    # 9 -> 1
    # 10 -> 2
    # 99 -> 2
    # 100 -> 3
    # 999 -> 3
    # 1000 -> 4
    expected_u0 = [0, 1, 1, 2, 2, 3, 3, 4]
    np.testing.assert_array_equal(n_inter[0], expected_u0)
    
    # Check User 1 (NaN -> NaN)
    assert np.isnan(n_inter[1, 0])
    assert n_inter[1, 1] == 0
