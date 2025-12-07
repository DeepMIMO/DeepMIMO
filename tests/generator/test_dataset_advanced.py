"""Advanced tests for DeepMIMO Dataset."""

import numpy as np
import pytest
from deepmimo import Dataset, consts as c

@pytest.fixture
def dataset_full():
    """Create a fully populated dataset."""
    ds = Dataset()
    ds.n_ue = 4
    ds.rx_pos = np.array([
        [0, 0, 1.5],
        [100, 0, 1.5],
        [0, 100, 1.5],
        [100, 100, 1.5]
    ])
    ds.tx_pos = np.array([50, 50, 10])
    
    # 2 paths per user
    ds.num_paths = np.array([2, 2, 2, 2])
    ds.inter = np.array([
        [c.INTERACTION_LOS, c.INTERACTION_REFLECTION],
        [c.INTERACTION_DIFFRACTION, 121], # Diff, Ref-Diff-Ref
        [np.nan, np.nan],
        [c.INTERACTION_LOS, c.INTERACTION_SCATTERING]
    ])
    
    # Mock angles
    ds.aoa_az = np.zeros((4, 2))
    ds.aoa_el = np.zeros((4, 2))
    ds.aod_az = np.zeros((4, 2))
    ds.aod_el = np.zeros((4, 2))
    
    # Mock rotations (required for trim_by_fov)
    ds.ch_params = type('P', (), {
        'bs_antenna': {c.PARAMSET_ANT_ROTATION: np.zeros(3)},
        'ue_antenna': {c.PARAMSET_ANT_ROTATION: np.zeros((4, 3))}
    })
    ds._clear_cache_rotated_angles = lambda: None
    
    # Initialize other required arrays
    ds.power = np.zeros((4, 2))
    ds.phase = np.zeros((4, 2))
    ds.delay = np.zeros((4, 2))
    ds.inter_pos = np.zeros((4, 2, 5, 3)) # 5 max interactions
    
    return ds

def test_trim_by_path_type(dataset_full):
    """Test trimming by path type."""
    ds = dataset_full
    
    # Keep only LoS
    ds_los = ds._trim_by_path_type(['LoS'])
    assert ds_los.inter.shape == (4, 1)
    assert not np.isnan(ds_los.inter[0, 0])
    assert not np.isnan(ds_los.inter[3, 0])
    assert np.isnan(ds_los.inter[1, 0]) # User 1 has no LoS
    
    # Keep Reflection
    ds_ref = ds._trim_by_path_type(['R'])
    assert ds_ref.inter.shape == (4, 1) # Reduced to 1 column
    assert not np.isnan(ds_ref.inter[0, 0]) # The Ref path moved to index 0

def test_trim_by_path_depth(dataset_full):
    """Test trimming by path depth (number of interactions)."""
    ds = dataset_full
    
    # Depth 0 (LoS)
    ds_d0 = ds._trim_by_path_depth(0)
    assert ds_d0.inter.shape == (4, 1)
    assert not np.isnan(ds_d0.inter[0, 0]) # LoS (0 interactions)
    assert np.isnan(ds_d0.inter[1, 0])
    
    # Depth 1 (LoS, Ref, Diff, Scat)
    ds_d1 = ds._trim_by_path_depth(1)
    # User 0: Path 0 (LoS, d=0) kept, Path 1 (Ref, d=1) kept. -> 2 paths.
    # User 1: Path 0 (Diff, d=1) kept, Path 1 (121, d=3) dropped. -> 1 path.
    assert ds_d1.inter.shape == (4, 2)
    assert not np.isnan(ds_d1.inter[0, 0]) 
    assert not np.isnan(ds_d1.inter[0, 1])
    assert not np.isnan(ds_d1.inter[1, 0])
    assert np.isnan(ds_d1.inter[1, 1])

def test_trim_by_fov(dataset_full):
    """Test trimming by Field of View."""
    ds = dataset_full
    
    # Mock angles to test FoV
    # Rotated angles are computed from raw angles + rotation.
    # Since rotations are 0, rotated = raw.
    
    # User 0: Path 0 @ az=0, Path 1 @ az=180
    ds.aoa_az[0, 0] = 0
    ds.aoa_az[0, 1] = 180
    # Set elevation to 90 (equator) to ensure azimuth is preserved
    # (If theta=0 (pole), azimuth is undefined/lost in rotation)
    ds.aoa_el[:] = 90 
    
    # FoV: +/- 90 deg (180 deg total) around 0 (implied by x-axis? No, FoV is usually centered on boresight)
    # If UE boresight is 0, 0 is in, 180 is out (back).
    
    # trim_by_fov(ue_fov=[120, 180]) -> Horizontal 120 means +/- 60.
    ds_fov = ds._trim_by_fov(ue_fov=[120, 180])
    
    # Path 0 (0 deg) is within +/- 60.
    # Path 1 (180 deg) is outside.
    # User 1 has paths at 0 deg (default initialization). So User 1 has 2 paths kept.
    # So max paths is 2.
    assert ds_fov.aoa_az.shape == (4, 2)
    assert not np.isnan(ds_fov.aoa_az[0, 0]) # User 0 Path 0 kept
    assert np.isnan(ds_fov.aoa_az[0, 1]) # User 0 Path 1 dropped (NaN)

def test_grid_functions(dataset_full):
    """Test grid detection and indexing."""
    ds = dataset_full
    # Grid is 2x2: (0,0), (100,0), (0,100), (100,100)
    
    grid_info = ds._compute_grid_info()
    assert np.array_equal(grid_info["grid_size"], [2, 2])
    assert np.allclose(grid_info["grid_spacing"], [100, 100])
    
    assert ds.has_valid_grid()
    
    # Get row 0 (y=0) -> (0,0), (100,0) -> indices 0, 1
    row0 = ds._get_row_idxs(0)
    assert np.array_equal(sorted(row0), [0, 1])
    
    # Get col 1 (x=100) -> (100,0), (100,100) -> indices 1, 3
    col1 = ds._get_col_idxs(1)
    assert np.array_equal(sorted(col1), [1, 3])
