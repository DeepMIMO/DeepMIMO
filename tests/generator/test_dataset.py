"""Dataset tests for DeepMIMO generator."""

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.generator.dataset import Dataset, DynamicDataset, MacroDataset


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    # Initialize with minimal required data to avoid KeyErrors
    data = {
        "n_ue": 5,
        "rx_pos": np.array(
            [[100, 0, 1.5], [0, 100, 1.5], [-100, 0, 1.5], [0, -100, 1.5], [100, 100, 1.5]]
        ),
        "tx_pos": np.array([0, 0, 10]),
        "user_ids": np.arange(5),
    }
    ds = Dataset(data)

    # Initialize required arrays with mock data
    ds.los = np.ones(5, dtype=bool)
    rng = np.random.default_rng()
    ds.power = rng.random((5, 1)) * -80
    ds.path_loss = rng.random(5) * 100

    # Mock channel params
    class MockChannelParams:
        def __init__(self) -> None:
            self.bs_antenna = {c.PARAMSET_ANT_ROTATION: np.zeros(3)}
            self.ue_antenna = {c.PARAMSET_ANT_ROTATION: np.zeros((5, 3))}
            self.freq_domain = False
            self.ofdm_params = None
            self.validate = lambda _value: None
            self.deepcopy = lambda: self

        def __getitem__(self, key):
            return getattr(self, key, None)

    ds.ch_params = MockChannelParams()
    ds.clear_cache_rotated_angles()

    return ds


def test_dataset_initialization() -> None:
    """Test dataset initialization defaults."""
    ds = Dataset()
    # Accessing properties on empty dataset should raise KeyError because
    # they depend on missing data
    with pytest.raises(KeyError):
        _ = ds.n_ue
    with pytest.raises(KeyError):
        _ = ds.rx_pos


def test_bs_look_at(sample_dataset) -> None:
    """Test BS look_at functionality."""
    ds = sample_dataset
    target = np.array([100, 0, 1.5])

    ds.bs_look_at(target)

    rot = ds.ch_params.bs_antenna[c.PARAMSET_ANT_ROTATION]
    # Should point East (0 azimuth) and slightly down
    assert np.isclose(rot[0], 0.0, atol=1e-2)
    assert rot[1] < 0  # Negative elevation (looking down)
    assert rot[2] == 0.0  # No tilt by default


def test_ue_look_at(sample_dataset) -> None:
    """Test UE look_at functionality."""
    ds = sample_dataset
    ds.ue_look_at(ds.tx_pos)

    rots = ds.ch_params.ue_antenna[c.PARAMSET_ANT_ROTATION]
    # Check first UE (East of BS) - should look West (180 or -180)
    # azimuth = arctan2(dy, dx) = arctan2(0, -100) = 180
    ue1_rot = rots[0]
    assert np.isclose(abs(ue1_rot[0]), 180.0, atol=1e-2)


def test_trim_dataset(sample_dataset) -> None:
    """Test trimming dataset by indices."""
    ds = sample_dataset
    idxs = [0, 2]  # Select 1st and 3rd UE

    trimmed = ds.trim(idxs=idxs)

    assert trimmed.n_ue == 2
    # Check that rx_pos is correctly sliced
    assert trimmed.rx_pos.shape == (2, 3)
    assert np.array_equal(trimmed.rx_pos[0], ds.rx_pos[0])
    assert np.array_equal(trimmed.rx_pos[1], ds.rx_pos[2])

    # Check that original dataset is unchanged
    assert ds.n_ue == 5


def test_trim_dataset_edge_cases(sample_dataset) -> None:
    """Test trimming with edge cases."""
    ds = sample_dataset

    # Empty indices - must be integer type
    trimmed_empty = ds.trim(idxs=np.array([], dtype=int))
    assert trimmed_empty.n_ue == 0
    assert trimmed_empty.rx_pos.shape == (0, 3)

    # Indices out of bounds should raise IndexError
    with pytest.raises(IndexError):
        ds.trim(idxs=[10])


def test_macro_dataset() -> None:
    """Test MacroDataset container."""
    ds1 = Dataset({"n_ue": 5, "rx_pos": np.zeros((5, 3))})
    ds2 = Dataset({"n_ue": 3, "rx_pos": np.zeros((3, 3))})

    macro = MacroDataset([ds1, ds2])

    assert len(macro) == 2
    assert macro[0] is ds1
    assert macro[1] is ds2

    # Propagated attributes
    # rx_pos should return list of arrays
    rx_positions = macro.rx_pos
    assert len(rx_positions) == 2
    assert rx_positions[0].shape == (5, 3)
    assert rx_positions[1].shape == (3, 3)

    # Test setting item
    macro["new_attr"] = 10
    assert ds1.new_attr == 10
    assert ds2.new_attr == 10


def test_dynamic_dataset() -> None:
    """Test DynamicDataset."""
    ds1 = Dataset({"n_ue": 2, "rx_pos": np.zeros((2, 3)), "tx_pos": np.zeros((1, 3)), "name": "s1"})
    ds2 = Dataset({"n_ue": 2, "rx_pos": np.ones((2, 3)), "tx_pos": np.ones((1, 3)), "name": "s2"})

    # Add scene with objects for _compute_speeds
    class MockObject:
        def __init__(self) -> None:
            self.position = np.zeros(3)
            self.vel = np.zeros(3)

    class MockScene:
        def __init__(self) -> None:
            self.objects = [MockObject()]  # Needs at least one object

    class MockObjectList:
        def __init__(self) -> None:
            self.objs = [MockObject()]

        @property
        def position(self):
            return [o.position for o in self.objs]

        @property
        def vel(self):
            return [o.vel for o in self.objs]

        @vel.setter
        def vel(self, v) -> None:
            for i, o in enumerate(self.objs):
                o.vel = v[i]

        def __getitem__(self, idx):
            return self.objs[idx]

    ds1.scene = MockScene()
    ds2.scene = MockScene()
    ds1.scene.objects = MockObjectList()
    ds2.scene.objects = MockObjectList()

    dyn = DynamicDataset([ds1, ds2], name="test_dyn")

    # Test timestamps setting
    timestamps = [0.0, 1.0]
    dyn.set_timestamps(timestamps)

    assert np.array_equal(dyn.timestamps, np.array([0.0, 1.0]))

    # Check if speeds were computed
    # rx_pos diff is 1.0 over 1.0s -> 1.0 m/s
    assert np.allclose(ds2.rx_vel, 1.0)
    # ds1 (first frame) assumes same velocity as next interval -> 1.0
    assert np.allclose(ds1.rx_vel, 1.0)


def test_compute_num_interactions() -> None:
    """Test interactions computation logic."""
    ds = Dataset({"n_ue": 2})  # Need n_ue for Dataset
    ds.inter = np.array(
        [[0, 1, 9, 10, 99, 100, 999, 1000], [np.nan, 0, 0, 0, 0, 0, 0, 0]], dtype=float
    )

    n_inter = ds.compute_num_interactions()
    expected_u0 = [0, 1, 1, 2, 2, 3, 3, 4]
    np.testing.assert_array_equal(n_inter[0], expected_u0)
    assert np.isnan(n_inter[1, 0])


def test_get_idxs(sample_dataset) -> None:
    """Test index selection dispatcher."""
    ds = sample_dataset
    # sample_dataset has 5 users
    # Mock num_paths to be [1, 0, 1, 0, 1] for active check
    ds.num_paths = np.array([1, 0, 1, 0, 1])

    active = ds.get_idxs("active")
    assert np.array_equal(active, [0, 2, 4])

    # Limits
    # [100, 0, 1.5] -> x=100
    # [0, 100, 1.5] -> y=100
    # [-100, 0, 1.5]
    # [0, -100, 1.5]
    # [100, 100, 1.5]

    # Select x > 50
    idxs = ds.get_idxs("limits", x_min=50)
    # Users 0 (100,0) and 4 (100,100) match
    assert 0 in idxs
    assert 4 in idxs
    assert len(idxs) == 2

    # Invalid mode
    with pytest.raises(ValueError, match="Unknown mode"):
        ds.get_idxs("invalid_mode")
