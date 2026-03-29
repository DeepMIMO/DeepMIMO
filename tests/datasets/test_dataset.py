"""Dataset tests for DeepMIMO generator."""

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.datasets.compat_v3 import _apply_v3_compat, _merge_rx_grids_v3, _rx_rank_map
from deepmimo.datasets.dataset import (
    Dataset,
    DynamicDataset,
    MacroDataset,
    MergedGridDataset,
)
from deepmimo.datasets.load import _validate_txrx_sets


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


def test_dynamic_dataset_subset_preserves_dynamic_type() -> None:
    """Selecting multiple snapshots should preserve the DynamicDataset wrapper."""
    snapshot1 = MacroDataset([Dataset({"rx_pos": np.zeros((1, 3)), "tx_pos": np.zeros((1, 3))})])
    snapshot2 = MacroDataset([Dataset({"rx_pos": np.ones((1, 3)), "tx_pos": np.ones((1, 3))})])
    snapshot3 = MacroDataset(
        [Dataset({"rx_pos": np.full((1, 3), 2.0), "tx_pos": np.zeros((1, 3))})]
    )
    snapshot1.name = "s1"
    snapshot2.name = "s2"
    snapshot3.name = "s3"
    dyn = DynamicDataset([snapshot1, snapshot2, snapshot3], name="test_dyn")
    dyn.timestamps = np.array([0.0, 1.0, 2.0])

    subset = dyn[0:2]

    assert isinstance(subset, DynamicDataset)
    assert subset.n_scenes == 2
    np.testing.assert_array_equal(subset.timestamps, np.array([0.0, 1.0]))


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


def _make_grid_dataset(nx: int, ny: int, tx_set_id: int, tx_idx: int, rx_set_id: int) -> Dataset:
    """Create a synthetic grid dataset with x-fastest ordering."""
    rx_pos = np.array([[x, y, 0.0] for y in range(ny) for x in range(nx)], dtype=float)
    n_ue = nx * ny
    return Dataset(
        {
            "rx_pos": rx_pos,
            "tx_pos": np.array([0.0, 0.0, 10.0], dtype=float),
            "power": np.zeros((n_ue, 2), dtype=float),
            "phase": np.zeros((n_ue, 2), dtype=float),
            "delay": np.zeros((n_ue, 2), dtype=float),
            "aoa_az": np.zeros((n_ue, 2), dtype=float),
            "aoa_el": np.zeros((n_ue, 2), dtype=float),
            "aod_az": np.zeros((n_ue, 2), dtype=float),
            "aod_el": np.zeros((n_ue, 2), dtype=float),
            "inter": np.zeros((n_ue, 2), dtype=float),
            "inter_pos": np.zeros((n_ue, 2, 1, 3), dtype=float),
            "txrx": {"tx_set_id": tx_set_id, "tx_idx": tx_idx, "rx_set_id": rx_set_id},
        }
    )


def _make_macro_dataset(*datasets: Dataset) -> MacroDataset:
    """Create a MacroDataset for merge tests."""
    return MacroDataset(list(datasets))


def test_macro_dataset_multi_index_returns_ordered_subset() -> None:
    """Selecting multiple indices should return a MacroDataset in the requested order."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)
    g3 = _make_grid_dataset(nx=2, ny=3, tx_set_id=0, tx_idx=0, rx_set_id=2)
    macro = _make_macro_dataset(g1, g2, g3)

    subset = macro[0, 2, 1]

    assert isinstance(subset, MacroDataset)
    assert subset[0] is g1
    assert subset[1] is g3
    assert subset[2] is g2


def test_macro_dataset_merge_native_preserves_selected_grid_order() -> None:
    """Native merges should use the requested dataset order and native row/col semantics."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)
    macro = _make_macro_dataset(g1, g2)

    merged = macro[1, 0].merge()

    assert isinstance(merged, MergedGridDataset)
    row_idxs = merged.get_idxs("row", row_idxs=np.array([0, 2]))
    np.testing.assert_array_equal(row_idxs, np.array([0, 1, 2, 3, 8, 9, 10]))


def test_macro_dataset_merge_handles_asymmetric_keys() -> None:
    """Merging should not fail when RX-grid datasets have non-uniform keys."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)
    g1["grid1_only"] = np.ones((g1.n_ue, 1), dtype=float)
    g2["grid2_only"] = np.zeros((g2.n_ue, 1), dtype=float)
    macro = _make_macro_dataset(g1, g2)

    merged = macro.merge()

    assert "grid1_only" in merged
    assert "grid2_only" in merged
    np.testing.assert_array_equal(merged["grid1_only"][: g1.n_ue], g1["grid1_only"])
    np.testing.assert_array_equal(merged["grid2_only"][g1.n_ue :], g2["grid2_only"])
    assert np.all(np.isnan(merged["grid1_only"][g1.n_ue :]))
    assert np.all(np.isnan(merged["grid2_only"][: g1.n_ue]))


def test_macro_dataset_merge_ignores_stale_cached_n_ue() -> None:
    """Repeated merges should not inherit a stale n_ue key from child dataset caches."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)
    macro = _make_macro_dataset(g1, g2)

    first_merge = macro.merge()
    assert first_merge.n_ue == g1.n_ue + g2.n_ue

    second_merge = macro.merge()
    assert second_merge.n_ue == g1.n_ue + g2.n_ue
    assert second_merge.rx_pos.shape[0] == g1.n_ue + g2.n_ue

    subset_merge = macro[0, 1].merge()
    assert subset_merge.n_ue == g1.n_ue + g2.n_ue


def test_macro_dataset_merge_rejects_multiple_transmitters() -> None:
    """Merging multiple transmitters should fail until Dataset supports that view explicitly."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=3, ny=2, tx_set_id=1, tx_idx=0, rx_set_id=0)
    macro = _make_macro_dataset(g1, g2)

    with pytest.raises(NotImplementedError, match="multiple transmitters"):
        macro.merge()


def test_apply_v3_compat_merges_rx_grids_per_tx_with_global_row_col_semantics() -> None:
    """compat_v3 should merge RX grids per TX and apply v3 global row/col indexing."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)
    g3 = _make_grid_dataset(nx=2, ny=3, tx_set_id=0, tx_idx=0, rx_set_id=2)
    compat = _apply_v3_compat(_make_macro_dataset(g1, g2, g3), rx_rank_map={0: 0, 1: 1, 2: 2})

    assert isinstance(compat, MergedGridDataset)
    row_idxs = compat.get_idxs("row", row_idxs=np.array([0, 2, 6]))
    col_idxs = compat.get_idxs("col", col_idxs=np.array([0, 3, 5]))

    np.testing.assert_array_equal(row_idxs, np.array([0, 1, 2, 6, 10, 14, 16, 18]))
    np.testing.assert_array_equal(col_idxs, np.array([0, 3, 6, 7, 8, 9, 14, 15]))


def test_apply_v3_compat_single_nonprimary_grid_swaps_row_col() -> None:
    """compat_v3 should wrap explicit non-primary RX loads with swapped row/col semantics."""
    dataset = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)

    compat = _apply_v3_compat(dataset, rx_rank_map={0: 0, 1: 1})

    assert isinstance(compat, MergedGridDataset)
    np.testing.assert_array_equal(compat.get_idxs("row", row_idxs=np.array([0])), np.array([0, 4]))
    np.testing.assert_array_equal(
        compat.get_idxs("col", col_idxs=np.array([0])),
        np.array([0, 1, 2, 3]),
    )


def test_apply_v3_compat_returns_one_merged_dataset_per_transmitter() -> None:
    """compat_v3 should group child datasets by TX before merging receiver grids."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)
    g3 = _make_grid_dataset(nx=2, ny=3, tx_set_id=1, tx_idx=0, rx_set_id=0)
    g4 = _make_grid_dataset(nx=2, ny=2, tx_set_id=1, tx_idx=0, rx_set_id=1)

    compat = _apply_v3_compat(_make_macro_dataset(g1, g2, g3, g4), rx_rank_map={0: 0, 1: 1})

    assert isinstance(compat, MacroDataset)
    assert len(compat) == 2
    assert all(isinstance(child, MergedGridDataset) for child in compat.datasets)
    assert compat[0].n_ue == g1.n_ue + g2.n_ue
    assert compat[1].n_ue == g3.n_ue + g4.n_ue


def test_merge_rx_grids_v3_requires_rx_rank_metadata() -> None:
    """Direct v3 compatibility helpers should fail loudly without RX rank metadata."""
    dataset = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)

    with pytest.raises(ValueError, match="requires RX rank metadata"):
        _merge_rx_grids_v3([dataset], rx_rank_map={})


def test_rx_rank_map_uses_numeric_set_ids() -> None:
    """RX rank mapping should sort by numeric RX set id, not key string."""
    txrx_dict = {
        "txrx_set_10": {"id": 10, "is_tx": False, "is_rx": True},
        "txrx_set_2": {"id": 2, "is_tx": False, "is_rx": True},
        "txrx_set_0": {"id": 0, "is_tx": False, "is_rx": True},
        "txrx_set_3": {"id": 3, "is_tx": True, "is_rx": False},
    }

    assert _rx_rank_map(txrx_dict) == {0: 0, 2: 1, 10: 2}


def test_validate_txrx_sets_orders_allowed_ids_deterministically() -> None:
    """TX set validation should follow the loader's deterministic key ordering."""
    txrx_dict = {
        "txrx_set_10": {"id": 10, "is_tx": True, "is_rx": False, "num_points": 1},
        "txrx_set_3": {"id": 3, "is_tx": True, "is_rx": False, "num_points": 1},
        "txrx_set_1": {"id": 1, "is_tx": False, "is_rx": True, "num_points": 1},
    }

    validated = _validate_txrx_sets("all", txrx_dict, "tx")
    assert list(validated.keys()) == [10, 3]

    with pytest.raises(ValueError, match=r"allowed sets \[10, 3\]"):
        _validate_txrx_sets([0], txrx_dict, "tx")
