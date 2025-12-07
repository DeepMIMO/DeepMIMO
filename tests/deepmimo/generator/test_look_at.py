"""Verification script for look_at function modifications
Tests all the improvements made to bs_look_at and ue_look_at functions
"""

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.generator.dataset import Dataset
from deepmimo.generator.channel import ChannelParameters


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


def test_basic_functionality():
    """Test basic functionality of both functions"""
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


def test_coordinate_handling():
    """Test 2D/3D coordinate handling"""
    dataset = create_test_dataset()

    # Test 2D coordinates
    target_2d = [100, 50]  # No z-coordinate
    
    dataset.bs_look_at(target_2d)
    rotation = dataset.ch_params.bs_antenna[c.PARAMSET_ANT_ROTATION]

    # Expected: azimuth = arctan2(50, 100), elevation = arctan2(-10, sqrt(100^2 + 50^2))
    expected_azimuth = np.degrees(np.arctan2(50, 100))
    expected_elevation = np.degrees(np.arctan2(-10, np.sqrt(100**2 + 50**2)))

    assert np.isclose(rotation[0], expected_azimuth, atol=1e-2)
    assert np.isclose(rotation[1], expected_elevation, atol=1e-2)


def test_z_rot_preservation():
    """Test z_rot preservation"""
    dataset = create_test_dataset()

    # Set initial z_rot values
    initial_bs_z_rot = 45.0
    initial_ue_z_rots = np.array([10, 20, 30, 40, 50])

    dataset.ch_params.bs_antenna[c.PARAMSET_ANT_ROTATION] = np.array([0, 0, initial_bs_z_rot])
    dataset.ch_params.ue_antenna[c.PARAMSET_ANT_ROTATION] = np.column_stack(
        [
            np.zeros(5),
            np.zeros(5),
            initial_ue_z_rots,
        ],
    )

    # Apply look_at functions
    dataset.bs_look_at([100, 0, 0])
    dataset.ue_look_at([0, 0, 10])

    # Check preservation
    final_bs_rotation = dataset.ch_params.bs_antenna[c.PARAMSET_ANT_ROTATION]
    final_ue_rotations = dataset.ch_params.ue_antenna[c.PARAMSET_ANT_ROTATION]

    assert final_bs_rotation[2] == initial_bs_z_rot
    assert np.allclose(final_ue_rotations[:, 2], initial_ue_z_rots)


def test_edge_cases():
    """Test edge cases and different input formats"""
    dataset = create_test_dataset()

    # 1. Single position for all UEs
    dataset.ue_look_at([50, 50, 5])
    rotations = dataset.ch_params.ue_antenna[c.PARAMSET_ANT_ROTATION]
    assert rotations.shape == (5, 3)

    # 2. Individual positions for each UE
    individual_targets = np.array(
        [
            [10, 10, 5],
            [20, 20, 5],
            [30, 30, 5],
            [40, 40, 5],
            [50, 50, 5],
        ],
    )
    dataset.ue_look_at(individual_targets)
    rotations = dataset.ch_params.ue_antenna[c.PARAMSET_ANT_ROTATION]
    assert rotations.shape == (5, 3)
    # Check that azimuths are different (they should be since targets are in same line but UEs are not)
    # Wait, targets are at [10,10], [20,20] etc. UEs are at [100,0], [0,100], [-100,0], etc.
    # So angles should be different.
    assert np.unique(rotations[:, 0]).size > 1

    # 3. Error case: wrong number of positions
    with pytest.raises(ValueError):
        dataset.ue_look_at(np.array([[10, 10, 5], [20, 20, 5], [30, 30, 5]]))


def create_realistic_dataset_for_compute_channels():
    """Create a realistic dataset that can actually run compute_channels"""
    dataset = Dataset()
    from deepmimo.scene import Scene
    dataset['scene'] = Scene()

    # Set realistic positions
    dataset.tx_pos = np.array([0, 0, 25])  # BS at 25m height
    dataset.rx_pos = np.array(
        [
            [50, 0, 1.5],  # UE1: 50m east
            [0, 50, 1.5],  # UE2: 50m north
        ],
    )
    dataset.n_ue = 2

    # Create proper channel parameters
    ch_params = ChannelParameters()

    # Set antenna parameters
    ch_params.bs_antenna = {
        c.PARAMSET_ANT_ROTATION: np.array([0, 0, 0]),
        c.PARAMSET_ANT_SPACING: 0.5,
        c.PARAMSET_ANT_SHAPE: [1, 1],
    }

    ch_params.ue_antenna = {
        c.PARAMSET_ANT_ROTATION: np.zeros((2, 3)),
        c.PARAMSET_ANT_SPACING: 0.5,
        c.PARAMSET_ANT_SHAPE: [1, 1],
    }

    # Set basic channel matrices with some realistic data
    dataset.aoa_az = np.random.uniform(-180, 180, (2, 5))  # 2 UEs, 5 paths
    dataset.aoa_el = np.random.uniform(-90, 90, (2, 5))
    dataset.aod_az = np.random.uniform(-180, 180, (2, 5))
    dataset.aod_el = np.random.uniform(-90, 90, (2, 5))
    dataset.delay = np.random.exponential(1e-8, (2, 5))  # Random delays
    dataset.power = np.random.exponential(1, (2, 5))  # Random powers
    dataset.phase = np.random.uniform(0, 2 * np.pi, (2, 5))  # Random phases
    dataset.los = np.array([1, 0])  # UE1 has LOS, UE2 doesn't
    dataset.num_paths = np.array([5, 5])  # 5 paths per UE

    dataset.ch_params = ch_params

    # Add cache clearing function
    dataset._clear_cache_rotated_angles = lambda: None

    return dataset


def test_compute_channels():
    """Test that compute_channels works after look_at modifications"""
    # Create a realistic test dataset that can actually run compute_channels
    dataset = create_realistic_dataset_for_compute_channels()

    # Test 1: Apply bs_look_at and compute channels
    target_ue = dataset.rx_pos[0]
    dataset.bs_look_at(target_ue)

    # Try to compute channels - this should work now!
    channel1 = dataset.compute_channels()
    assert channel1 is not None

    # Test 2: Apply ue_look_at and compute channels
    dataset.ue_look_at(dataset.tx_pos)

    channel2 = dataset.compute_channels()
    assert channel2 is not None

