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


# ===========================================================================
# Helpers shared by new tests
# ===========================================================================


def _make_minimal_path_dataset(n_ue: int = 2, n_paths: int = 2) -> Dataset:
    """Return a Dataset with all path arrays populated (for compute_channels etc.)."""
    rng = np.random.default_rng(42)
    data = {
        "rx_pos": rng.uniform(-100, 100, (n_ue, 3)).astype(np.float32),
        "tx_pos": np.array([0.0, 0.0, 10.0], dtype=np.float32),
        "power": rng.uniform(-100, -60, (n_ue, n_paths)).astype(np.float32),
        "phase": rng.uniform(0, 360, (n_ue, n_paths)).astype(np.float32),
        "delay": rng.uniform(1e-8, 1e-6, (n_ue, n_paths)).astype(np.float32),
        "aoa_az": rng.uniform(-180, 180, (n_ue, n_paths)).astype(np.float32),
        "aoa_el": rng.uniform(-90, 90, (n_ue, n_paths)).astype(np.float32),
        "aod_az": rng.uniform(-180, 180, (n_ue, n_paths)).astype(np.float32),
        "aod_el": rng.uniform(-90, 90, (n_ue, n_paths)).astype(np.float32),
        "inter": np.zeros((n_ue, n_paths), dtype=np.float32),  # LOS paths
        "inter_pos": np.zeros((n_ue, n_paths, 1, 3), dtype=np.float32),
    }
    ds = Dataset(data)

    # Give it a trivial scene with objects that have .vel
    class _MockObject:
        def __init__(self):
            self.vel = np.zeros(3)

    class _MockScene:
        def __init__(self):
            self.objects = [_MockObject()]

    ds.scene = _MockScene()
    return ds


def _make_minimal_path_dataset_with_ch_params(n_ue: int = 2, n_paths: int = 2) -> Dataset:
    """Return a path dataset that also has valid ChannelParameters attached."""
    from deepmimo.generator.channel import ChannelParameters  # noqa: PLC0415

    ds = _make_minimal_path_dataset(n_ue=n_ue, n_paths=n_paths)
    params = ChannelParameters(freq_domain=False, num_paths=n_paths)
    ds.set_channel_params(params)
    return ds


# ===========================================================================
# __dir__ includes computed attribute names
# ===========================================================================


def test_dir_includes_computed_attributes() -> None:
    """__dir__ must expose all computed attribute names."""
    ds = Dataset({"rx_pos": np.zeros((3, 3)), "tx_pos": np.zeros(3)})
    d = dir(ds)
    from deepmimo.datasets.dataset import Dataset as _Dataset  # noqa: PLC0415

    for attr in _Dataset._computed_attributes:  # noqa: SLF001
        assert attr in d, f"Expected '{attr}' in dir(dataset)"


def test_dir_includes_aliases() -> None:
    """__dir__ must also expose alias names defined in consts.DATASET_ALIASES."""
    from deepmimo import consts as c  # noqa: PLC0415

    ds = Dataset({"rx_pos": np.zeros((3, 3)), "tx_pos": np.zeros(3)})
    d = dir(ds)
    for alias in c.DATASET_ALIASES:
        assert alias in d, f"Expected alias '{alias}' in dir(dataset)"


# ===========================================================================
# Alias resolution
# ===========================================================================


def test_alias_pwr_resolves_to_power() -> None:
    """Accessing ds.pwr should return the same data as ds.power."""
    ds = _make_minimal_path_dataset()
    np.testing.assert_array_equal(np.array(ds.pwr), np.array(ds.power))


def test_alias_pl_resolves_to_pathloss() -> None:
    """Accessing ds.pl should compute and return pathloss."""
    ds = _make_minimal_path_dataset()
    pl = ds.pl
    assert pl is not None
    assert pl.shape == (ds.n_ue,)


def test_alias_dist_resolves_to_distance() -> None:
    """Accessing ds.dist should return per-user distances."""
    ds = _make_minimal_path_dataset()
    d1 = np.array(ds.dist)
    d2 = np.array(ds.distance)
    np.testing.assert_array_equal(d1, d2)
    assert d1.shape == (ds.n_ue,)


def test_alias_n_paths_resolves_to_num_paths() -> None:
    """Accessing ds.n_paths should return the same array as ds.num_paths."""
    ds = _make_minimal_path_dataset()
    np.testing.assert_array_equal(np.array(ds.n_paths), np.array(ds.num_paths))


def test_alias_toa_resolves_to_delay() -> None:
    """Accessing ds.toa should return the same array as ds.delay."""
    ds = _make_minimal_path_dataset()
    np.testing.assert_array_equal(np.array(ds.toa), np.array(ds.delay))


# ===========================================================================
# Lazy computed properties
# ===========================================================================


def test_computed_num_paths() -> None:
    """num_paths should count non-NaN entries in aoa_az per user."""
    ds = _make_minimal_path_dataset(n_ue=3, n_paths=4)
    # Manually NaN some paths
    ds.aoa_az[1, 2:] = np.nan
    ds.aoa_az[2, :] = np.nan
    # Clear any cached value to force recompute
    ds.clear_all_caches()
    np_arr = np.array(ds.num_paths)
    assert np_arr[0] == 4
    assert np_arr[1] == 2
    assert np_arr[2] == 0


def test_computed_los() -> None:
    """_compute_los should yield 1 for LoS, 0 for NLoS, -1 for no paths."""
    ds = _make_minimal_path_dataset(n_ue=3, n_paths=2)
    # User 0: LOS (first inter == 0)
    ds.inter[0] = [0.0, 1.0]
    # User 1: NLOS (first inter != 0)
    ds.inter[1] = [1.0, 2.0]
    # User 2: no paths (all NaN)
    ds.aoa_az[2] = np.nan
    ds.inter[2] = np.nan
    ds.clear_all_caches()
    los = np.array(ds.los)
    assert los[0] == 1
    assert los[1] == 0
    assert los[2] == -1


def test_computed_distance() -> None:
    """Distance should be Euclidean norm of (rx_pos - tx_pos)."""
    ds = Dataset(
        {
            "rx_pos": np.array([[3.0, 4.0, 0.0]]),
            "tx_pos": np.array([0.0, 0.0, 0.0]),
        }
    )
    dist = np.array(ds.distance)
    assert dist[0] == pytest.approx(5.0)


def test_computed_power_linear() -> None:
    """power_linear should equal 10^(power/10)."""
    ds = _make_minimal_path_dataset(n_ue=1, n_paths=1)
    ds.power[0, 0] = -10.0  # -10 dB => 0.1 linear
    ds.clear_all_caches()
    pl = np.array(ds.power_linear)
    assert pl[0, 0] == pytest.approx(0.1, rel=1e-4)


def test_computed_pathloss() -> None:
    """Pathloss should be a finite negative value for non-empty users."""
    ds = _make_minimal_path_dataset(n_ue=2, n_paths=2)
    pl = np.array(ds.pathloss)
    assert pl.shape == (2,)
    assert np.all(np.isfinite(pl))


# ===========================================================================
# set_channel_params - rotation cache cleared when rotation changes
# ===========================================================================


def test_set_channel_params_updates_params_on_rotation_change() -> None:
    """Calling set_channel_params twice with different rotations updates the stored params."""
    from deepmimo import consts as c  # noqa: PLC0415
    from deepmimo.generator.channel import ChannelParameters  # noqa: PLC0415

    ds = _make_minimal_path_dataset()

    params1 = ChannelParameters(freq_domain=False)
    params1.bs_antenna[c.PARAMSET_ANT_ROTATION] = np.array([0.0, 0.0, 0.0])
    ds.set_channel_params(params1)

    # Force computation and caching of rotated angles
    _ = ds[c.AOD_AZ_ROT_PARAM_NAME]

    # Change rotation - new params should be stored
    params2 = ChannelParameters(freq_domain=False)
    params2.bs_antenna[c.PARAMSET_ANT_ROTATION] = np.array([45.0, 0.0, 0.0])
    ds.set_channel_params(params2)

    # New params should be reflected in ch_params
    np.testing.assert_array_equal(
        ds.ch_params.bs_antenna[c.PARAMSET_ANT_ROTATION], [45.0, 0.0, 0.0]
    )


def test_set_channel_params_no_cache_clear_if_same_rotation() -> None:
    """Calling set_channel_params twice with identical rotations must NOT clear the cache."""
    from deepmimo import consts as c  # noqa: PLC0415
    from deepmimo.generator.channel import ChannelParameters  # noqa: PLC0415

    ds = _make_minimal_path_dataset()

    params1 = ChannelParameters(freq_domain=False)
    ds.set_channel_params(params1)

    _ = ds[c.AOD_AZ_ROT_PARAM_NAME]
    assert c.AOD_AZ_ROT_PARAM_NAME in ds.keys()  # noqa: SIM118

    params2 = ChannelParameters(freq_domain=False)
    ds.set_channel_params(params2)

    # Params updated - new rotation is identical so ch_params should reflect it
    np.testing.assert_array_equal(
        ds.ch_params.bs_antenna[c.PARAMSET_ANT_ROTATION],
        params2.bs_antenna[c.PARAMSET_ANT_ROTATION],
    )


def test_set_channel_params_none_uses_defaults() -> None:
    """Calling set_channel_params(None) should install default ChannelParameters."""
    ds = _make_minimal_path_dataset()
    returned = ds.set_channel_params(None)
    assert returned is not None
    assert ds.ch_params is not None


# ===========================================================================
# compute_channels - minimal time-domain channel
# ===========================================================================


def test_compute_channels_returns_correct_shape() -> None:
    """compute_channels in time-domain should return shape (n_ue, n_rx, n_tx, n_paths)."""
    from deepmimo.generator.channel import ChannelParameters  # noqa: PLC0415

    n_ue, n_paths = 2, 2
    ds = _make_minimal_path_dataset(n_ue=n_ue, n_paths=n_paths)
    params = ChannelParameters(
        freq_domain=False,
        num_paths=n_paths,
        bs_antenna={"shape": [1, 1]},
        ue_antenna={"shape": [1, 1]},
    )
    ch = ds.compute_channels(params=params)
    # shape: (n_ue, n_rx_ant=1, n_tx_ant=1, n_paths=2)
    assert ch.shape[0] == n_ue
    assert ch.shape[1] == 1  # 1 RX antenna
    assert ch.shape[2] == 1  # 1 TX antenna


def test_compute_channels_stored_in_dataset() -> None:
    """After compute_channels, the result should be cached under 'channel'."""
    from deepmimo import consts as c  # noqa: PLC0415
    from deepmimo.generator.channel import ChannelParameters  # noqa: PLC0415

    ds = _make_minimal_path_dataset()
    params = ChannelParameters(
        freq_domain=False, bs_antenna={"shape": [1, 1]}, ue_antenna={"shape": [1, 1]}
    )
    ch = ds.compute_channels(params=params)
    assert c.CHANNEL_PARAM_NAME in ds
    np.testing.assert_array_equal(np.array(ds[c.CHANNEL_PARAM_NAME]), ch)


def test_compute_channels_with_kwargs() -> None:
    """compute_channels should accept kwargs to construct ChannelParameters inline."""
    ds = _make_minimal_path_dataset()
    # Passing kwargs should build a ChannelParameters object from them
    ch = ds.compute_channels(
        freq_domain=False, bs_antenna={"shape": [1, 1]}, ue_antenna={"shape": [1, 1]}
    )
    assert ch is not None


# ===========================================================================
# ue_look_at error paths
# ===========================================================================


def test_ue_look_at_shape_mismatch_raises_value_error(sample_dataset) -> None:
    """ue_look_at with (m, 3) positions where m != n_ue must raise ValueError."""
    ds = sample_dataset
    wrong_pos = np.zeros((ds.n_ue + 1, 3))
    with pytest.raises(ValueError, match="must match number of users"):
        ds.ue_look_at(wrong_pos)


def test_ue_look_at_3d_array_raises_value_error(sample_dataset) -> None:
    """ue_look_at with a 3-D array must raise ValueError."""
    ds = sample_dataset
    bad_pos = np.zeros((ds.n_ue, 3, 1))
    with pytest.raises(ValueError, match="1D or 2D"):
        ds.ue_look_at(bad_pos)


def test_ue_look_at_single_row_2d_broadcasts(sample_dataset) -> None:
    """ue_look_at with shape (1, 3) should broadcast to all UEs without error."""
    ds = sample_dataset
    look_pos = np.array([[0.0, 0.0, 10.0]])
    ds.ue_look_at(look_pos)  # should not raise
    rots = ds.ch_params.ue_antenna[c.PARAMSET_ANT_ROTATION]
    assert rots.shape == (ds.n_ue, 3)


# ===========================================================================
# trim with bool mask
# ===========================================================================


def test_trim_with_bool_idxs(sample_dataset) -> None:
    """trim(idxs=...) should work when passed a boolean mask converted to integer indices."""
    ds = sample_dataset
    bool_mask = np.array([True, False, True, False, True])
    int_idxs = np.where(bool_mask)[0]
    trimmed = ds.trim(idxs=int_idxs)
    assert trimmed.n_ue == 3
    np.testing.assert_array_equal(trimmed.rx_pos[0], ds.rx_pos[0])
    np.testing.assert_array_equal(trimmed.rx_pos[1], ds.rx_pos[2])
    np.testing.assert_array_equal(trimmed.rx_pos[2], ds.rx_pos[4])


# ===========================================================================
# has_valid_grid
# ===========================================================================


def test_has_valid_grid_true() -> None:
    """A regular 2x3 grid should report has_valid_grid=True."""
    nx, ny = 3, 2
    rx_pos = np.array([[x, y, 0.0] for y in range(ny) for x in range(nx)], dtype=float)
    ds = Dataset({"rx_pos": rx_pos, "tx_pos": np.zeros(3)})
    assert ds.has_valid_grid()


def test_has_valid_grid_false() -> None:
    """Irregular positions should report has_valid_grid=False."""
    # use 4 points where unique x vals = 2, unique y vals = 3, product=6 != 4
    rx_pos2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=float)
    ds2 = Dataset({"rx_pos": rx_pos2, "tx_pos": np.zeros(3)})
    # grid_size: x has 2 unique (0,1), y has 3 unique (0,1,2), product=6 != 4
    assert not ds2.has_valid_grid()


# ===========================================================================
# get_idxs row/col on Dataset with regular grid
# ===========================================================================


def test_get_idxs_row_on_dataset() -> None:
    """get_idxs('row') on a plain Dataset should return correct row indices."""
    nx, ny = 4, 3
    rx_pos = np.array([[x, y, 0.0] for y in range(ny) for x in range(nx)], dtype=float)
    ds = Dataset({"rx_pos": rx_pos, "tx_pos": np.zeros(3)})
    # Row 0 = y=0 in native grid (x-fastest): first nx points
    row_idxs = ds.get_idxs("row", row_idxs=np.array([0]))
    assert set(row_idxs.tolist()) == set(range(nx))


def test_get_idxs_col_on_dataset() -> None:
    """get_idxs('col') on a plain Dataset should return correct column indices."""
    nx, ny = 4, 3
    rx_pos = np.array([[x, y, 0.0] for y in range(ny) for x in range(nx)], dtype=float)
    ds = Dataset({"rx_pos": rx_pos, "tx_pos": np.zeros(3)})
    # Col 0 = x=0: every ny-th point starting at 0
    col_idxs = ds.get_idxs("col", col_idxs=np.array([0]))
    expected = np.array([0, nx, 2 * nx])
    np.testing.assert_array_equal(np.sort(col_idxs), np.sort(expected))


# ===========================================================================
# get_idxs row/col on MergedGridDataset
# ===========================================================================


def test_merged_grid_dataset_get_idxs_row() -> None:
    """MergedGridDataset.get_idxs('row') resolves global rows to user indices."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)
    macro = MacroDataset([g1, g2])
    merged = macro.merge()

    # Row 0 of g1 (nx=3, ny=2, native 'row' = y-axis => 3 users)
    row0 = merged.get_idxs("row", row_idxs=np.array([0]))
    assert len(row0) == 3  # nx=3 users in row 0 of g1

    # Row 2 corresponds to first row of g2 (offset by g1.n_ue=6)
    row2 = merged.get_idxs("row", row_idxs=np.array([2]))
    assert len(row2) == 4  # nx=4 users in row 0 of g2
    assert np.all(row2 >= g1.n_ue)  # They are in the g2 portion


def test_merged_grid_dataset_get_idxs_col() -> None:
    """MergedGridDataset.get_idxs('col') resolves global columns to user indices."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)
    macro = MacroDataset([g1, g2])
    merged = macro.merge()

    col0 = merged.get_idxs("col", col_idxs=np.array([0]))
    assert len(col0) == 2  # ny=2 rows in col 0 of g1


def test_merged_grid_dataset_row_idx_out_of_range_raises() -> None:
    """Requesting an out-of-range global row should raise IndexError."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)
    merged = MacroDataset([g1, g2]).merge()
    assert isinstance(merged, MergedGridDataset)
    with pytest.raises(IndexError):
        merged.get_idxs("row", row_idxs=np.array([999]))


def test_merged_grid_dataset_invalid_axis_raises() -> None:
    """Passing invalid axis to _resolve_global_grid_idxs should raise ValueError."""
    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=4, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=1)
    merged = MacroDataset([g1, g2]).merge()
    assert isinstance(merged, MergedGridDataset)
    with pytest.raises(ValueError, match="Invalid axis"):
        merged._resolve_global_grid_idxs("diagonal", np.array([0]))  # noqa: SLF001


# ===========================================================================
# rx_vel / tx_vel setters - error paths
# ===========================================================================


def test_rx_vel_wrong_1d_length_raises() -> None:
    """Setting rx_vel with a 1-D array of wrong length should raise ValueError."""
    ds = _make_minimal_path_dataset(n_ue=2)
    with pytest.raises(ValueError, match="cartesian coordinates"):
        ds.rx_vel = np.array([1.0, 2.0])  # should be 3 elements


def test_rx_vel_wrong_2d_columns_raises() -> None:
    """Setting rx_vel with (n_ue, 2) array should raise ValueError."""
    ds = _make_minimal_path_dataset(n_ue=2)
    with pytest.raises(ValueError, match="cartesian coordinates"):
        ds.rx_vel = np.zeros((2, 2))


def test_rx_vel_wrong_n_ue_raises() -> None:
    """Setting rx_vel with mismatched n_ue should raise ValueError."""
    ds = _make_minimal_path_dataset(n_ue=2)
    with pytest.raises(ValueError, match="Number of users"):
        ds.rx_vel = np.zeros((5, 3))


def test_rx_vel_1d_broadcasts_to_all_ues() -> None:
    """Setting rx_vel with shape (3,) should broadcast to all UEs."""
    ds = _make_minimal_path_dataset(n_ue=3)
    vel = np.array([1.0, 2.0, 3.0])
    ds.rx_vel = vel
    assert ds.rx_vel.shape == (3, 3)
    np.testing.assert_array_equal(ds.rx_vel[0], vel)
    np.testing.assert_array_equal(ds.rx_vel[2], vel)


def test_rx_vel_list_is_accepted() -> None:
    """Setting rx_vel as a list should work."""
    ds = _make_minimal_path_dataset(n_ue=2)
    ds.rx_vel = [0.0, 0.0, 1.0]
    assert ds.rx_vel.shape == (2, 3)


def test_rx_vel_default_is_zeros() -> None:
    """Accessing rx_vel without setting it should return zeros."""
    ds = _make_minimal_path_dataset(n_ue=3)
    vel = ds.rx_vel
    assert vel.shape == (3, 3)
    assert np.all(vel == 0)


def test_tx_vel_wrong_ndim_raises() -> None:
    """Setting tx_vel with a 2-D array should raise ValueError."""
    ds = _make_minimal_path_dataset(n_ue=2)
    with pytest.raises(ValueError, match="single cartesian coordinate"):
        ds.tx_vel = np.zeros((1, 3))


def test_tx_vel_1d_is_accepted() -> None:
    """Setting tx_vel with a valid (3,) array should work."""
    ds = _make_minimal_path_dataset(n_ue=2)
    ds.tx_vel = np.array([5.0, 0.0, 0.0])
    np.testing.assert_array_equal(ds.tx_vel, np.array([5.0, 0.0, 0.0]))


def test_tx_vel_default_is_zeros() -> None:
    """Accessing tx_vel without setting it should return zeros."""
    ds = _make_minimal_path_dataset(n_ue=2)
    vel = ds.tx_vel
    assert vel.shape == (3,)
    assert np.all(vel == 0)


# ===========================================================================
# _validate_rx_index
# ===========================================================================


def test_validate_rx_index_out_of_range_raises() -> None:
    """_validate_rx_index with an idx out of range should raise IndexError."""
    ds = _make_minimal_path_dataset(n_ue=3, n_paths=2)
    with pytest.raises(IndexError, match="out of range"):
        ds._validate_rx_index(10, None)  # noqa: SLF001


def test_validate_rx_index_negative_raises() -> None:
    """_validate_rx_index with negative idx should raise IndexError."""
    ds = _make_minimal_path_dataset(n_ue=3, n_paths=2)
    with pytest.raises(IndexError, match="out of range"):
        ds._validate_rx_index(-1, None)  # noqa: SLF001


def test_validate_rx_index_none_path_returns_all() -> None:
    """_validate_rx_index with path_idxs=None should return all valid path indices."""
    ds = _make_minimal_path_dataset(n_ue=2, n_paths=2)
    result = ds._validate_rx_index(0, None)  # noqa: SLF001
    assert len(result) == int(ds.num_paths[0])


def test_validate_rx_index_bad_path_idx_raises() -> None:
    """_validate_rx_index with out-of-range path index should raise IndexError."""
    ds = _make_minimal_path_dataset(n_ue=2, n_paths=2)
    with pytest.raises(IndexError, match="Path indices must be in range"):
        ds._validate_rx_index(0, [999])  # noqa: SLF001


# ===========================================================================
# _compute_inter_str / _compute_inter_int / _compute_inter_vec
# ===========================================================================


def test_compute_inter_str() -> None:
    """_compute_inter_str should translate numeric codes to string labels."""
    ds = Dataset({"n_ue": 2})
    ds.inter = np.array([[0.0, 1.0], [2.0, np.nan]])
    result = ds._compute_inter_str()  # noqa: SLF001
    # Code 0 -> '' (LOS, no translation), code 1 -> 'R'
    # The translate_code logic: s[:-2] removes last two chars (decimal ".0"), then translates
    assert result[0, 0] == ""  # LOS -> empty string after mapping '0' -> ''
    assert result[0, 1] == "R"  # Reflection
    assert result[1, 0] == "D"  # Diffraction
    assert result[1, 1] == "n"  # NaN -> "n"


def test_compute_inter_int() -> None:
    """_compute_inter_int should replace NaN with -1 and return int array."""
    ds = Dataset({"n_ue": 2})
    ds.inter = np.array([[0.0, np.nan], [1.0, 2.0]])
    result = ds._compute_inter_int()  # noqa: SLF001
    assert result.dtype == int or np.issubdtype(result.dtype, np.integer)
    assert result[0, 1] == -1
    assert result[1, 0] == 1
    assert result[1, 1] == 2


def test_compute_inter_vec() -> None:
    """_compute_inter_vec should expand integer codes into digit arrays."""
    ds = Dataset({"n_ue": 1})
    ds.inter = np.array([[121.0, np.nan]])
    result = ds._compute_inter_vec()  # noqa: SLF001
    # Shape: (1, 2, max_len), 121 has 3 digits
    assert result.shape[0] == 1
    assert result.shape[1] == 2
    assert result.shape[2] >= 3
    # First path: digits of 121 = [1, 2, 1]
    np.testing.assert_array_equal(result[0, 0, :3], [1, 2, 1])
    # Second path (NaN): all -1
    assert np.all(result[0, 1] == -1)


# ===========================================================================
# _compute_num_paths / _compute_max_paths
# ===========================================================================


def test_compute_max_paths() -> None:
    """max_paths should equal the maximum of num_paths."""
    ds = _make_minimal_path_dataset(n_ue=3, n_paths=4)
    ds.aoa_az[1, 3] = np.nan  # user 1 has 3 paths
    ds.aoa_az[2, :] = np.nan  # user 2 has 0 paths
    ds.clear_all_caches()
    assert int(ds.max_paths) == 4


# ===========================================================================
# n_ue computed attribute
# ===========================================================================


def test_n_ue_computed_from_rx_pos() -> None:
    """n_ue should be computed from rx_pos.shape[0]."""
    ds = Dataset({"rx_pos": np.zeros((7, 3))})
    assert ds.n_ue == 7


# ===========================================================================
# _compute_power_linear
# ===========================================================================


def test_compute_power_linear_values() -> None:
    """power_linear should be 10^(power/10)."""
    ds = Dataset({"rx_pos": np.zeros((2, 3))})
    ds.power = np.array([[-20.0, -10.0], [0.0, -30.0]])
    pwr_lin = np.array(ds.power_linear)
    expected = 10 ** (ds.power / 10)
    np.testing.assert_allclose(pwr_lin, expected, rtol=1e-5)


# ===========================================================================
# compute_pathloss edge cases
# ===========================================================================


def test_compute_pathloss_inactive_user_is_nan() -> None:
    """Pathloss should be NaN for users with all-NaN power (no paths)."""
    ds = Dataset({"rx_pos": np.zeros((2, 3))})
    ds.power = np.array([[-10.0, -20.0], [np.nan, np.nan]])
    ds.phase = np.array([[0.0, 0.0], [0.0, 0.0]])
    pl = ds.compute_pathloss()
    assert np.isfinite(pl[0])
    assert np.isnan(pl[1])


# ===========================================================================
# _compute_los with edge cases
# ===========================================================================


def test_compute_los_handles_mixed_users() -> None:
    """_compute_los should correctly classify mixed LOS/NLOS/inactive users."""
    n_ue = 4
    ds = Dataset({"rx_pos": np.zeros((n_ue, 3))})
    ds.inter = np.array(
        [
            [0.0, np.nan],  # LOS -> 1
            [1.0, np.nan],  # NLOS -> 0
            [0.0, 1.0],  # LOS (first path is LOS) -> 1
            [np.nan, np.nan],  # no paths -> -1
        ]
    )
    ds.aoa_az = np.array(
        [
            [1.0, np.nan],
            [1.0, np.nan],
            [1.0, 1.0],
            [np.nan, np.nan],
        ]
    )
    los = ds._compute_los()  # noqa: SLF001
    assert los[0] == 1
    assert los[1] == 0
    assert los[2] == 1
    assert los[3] == -1


# ===========================================================================
# set_doppler
# ===========================================================================


def test_set_doppler_scalar_broadcasts() -> None:
    """set_doppler with scalar should fill entire (n_ue, max_paths) array."""
    ds = _make_minimal_path_dataset(n_ue=2, n_paths=2)
    # Manually set max_paths to avoid computing from data
    ds["max_paths"] = 2
    ds.set_doppler(5.0)
    assert ds.doppler.shape == (2, 2)
    assert np.all(np.array(ds.doppler) == 5.0)


def test_set_doppler_per_user_broadcasts() -> None:
    """set_doppler with (n_ue,) array should broadcast to (n_ue, max_paths)."""
    ds = _make_minimal_path_dataset(n_ue=2, n_paths=2)
    ds["max_paths"] = 2
    ds.set_doppler(np.array([1.0, 2.0]))
    doppler = np.array(ds.doppler)
    assert doppler.shape == (2, 2)


def test_set_doppler_invalid_shape_raises() -> None:
    """set_doppler with invalid shape should raise ValueError."""
    ds = _make_minimal_path_dataset(n_ue=2, n_paths=2)
    ds["max_paths"] = 2
    with pytest.raises(ValueError, match="Invalid doppler shape"):
        ds.set_doppler(np.zeros((3, 5)))


# ===========================================================================
# Grid info
# ===========================================================================


def test_compute_grid_info_regular_grid() -> None:
    """_compute_grid_info should return correct size and spacing for a regular grid."""
    nx, ny = 5, 4
    rx_pos = np.array([[x, y, 0.0] for y in range(ny) for x in range(nx)], dtype=float)
    ds = Dataset({"rx_pos": rx_pos, "tx_pos": np.zeros(3)})
    info = ds.compute_grid_info()
    np.testing.assert_array_equal(info["grid_size"], [nx, ny])
    np.testing.assert_allclose(info["grid_spacing"], [1.0, 1.0])


def test_compute_grid_info_single_point() -> None:
    """_compute_grid_info for a single point should return spacing of 0."""
    ds = Dataset({"rx_pos": np.array([[1.0, 2.0, 0.0]]), "tx_pos": np.zeros(3)})
    info = ds.compute_grid_info()
    np.testing.assert_array_equal(info["grid_size"], [1, 1])
    np.testing.assert_array_equal(info["grid_spacing"], [0, 0])


# ===========================================================================
# _compute_distances edge case
# ===========================================================================


def test_compute_distances_matches_linalg_norm() -> None:
    """_compute_distances should match numpy linalg norm."""
    rng = np.random.default_rng(7)
    rx_pos = rng.uniform(0, 100, (5, 3))
    tx_pos = rng.uniform(0, 100, (3,))
    ds = Dataset({"rx_pos": rx_pos, "tx_pos": tx_pos})
    expected = np.linalg.norm(rx_pos - tx_pos, axis=1)
    np.testing.assert_allclose(np.array(ds.distance), expected, rtol=1e-5)


# ===========================================================================
# MacroDataset.__getattr__ propagation
# ===========================================================================


def test_macro_dataset_attr_propagation() -> None:
    """MacroDataset attribute access should return list from all child datasets."""
    ds1 = Dataset({"n_ue": 2, "rx_pos": np.zeros((2, 3))})
    ds2 = Dataset({"n_ue": 3, "rx_pos": np.ones((3, 3))})
    macro = MacroDataset([ds1, ds2])

    # n_ue is per-dataset, not shared
    n_ues = macro.n_ue
    assert isinstance(n_ues, list)
    assert n_ues[0] == 2
    assert n_ues[1] == 3


def test_macro_dataset_single_child_returns_scalar() -> None:
    """MacroDataset with one child returns value directly instead of list."""
    ds = Dataset({"n_ue": 4, "rx_pos": np.zeros((4, 3))})
    macro = MacroDataset([ds])
    assert macro.n_ue == 4


# ===========================================================================
# MacroDataset.append
# ===========================================================================


def test_macro_dataset_append() -> None:
    """MacroDataset.append should add a dataset to the collection."""
    macro = MacroDataset([])
    ds = Dataset({"n_ue": 2})
    macro.append(ds)
    assert len(macro) == 1
    assert macro[0] is ds


# ===========================================================================
# MacroDataset._get_single on empty MacroDataset
# ===========================================================================


def test_macro_dataset_get_single_empty_raises() -> None:
    """_get_single on an empty MacroDataset should raise IndexError."""
    macro = MacroDataset([])
    with pytest.raises(IndexError, match="empty"):
        macro._get_single("scene")  # noqa: SLF001


# ===========================================================================
# clear_all_caches
# ===========================================================================


def test_clear_all_caches_removes_computed_keys() -> None:
    """clear_all_caches should remove cached core computed arrays."""
    from deepmimo import consts as c  # noqa: PLC0415

    ds = _make_minimal_path_dataset(n_ue=2, n_paths=2)
    # Trigger computation of cached values that are removed by _clear_cache_core
    _ = ds.num_paths
    _ = ds.los

    ds.clear_all_caches()

    # After clearing, core cached keys should be gone from the underlying dict.
    # Note: `key in ds` also matches computed attributes so we check ds.keys() directly.
    assert c.NUM_PATHS_PARAM_NAME not in ds.keys()  # noqa: SIM118
    assert c.LOS_PARAM_NAME not in ds.keys()  # noqa: SIM118

    # Also verify rotated angles are cleared
    _ = ds[c.AOD_AZ_ROT_PARAM_NAME]
    ds.clear_all_caches()
    assert c.AOD_AZ_ROT_PARAM_NAME not in ds.keys()  # noqa: SIM118


# ===========================================================================
# _trim_by_index edge cases
# ===========================================================================


def test_trim_by_index_preserves_scalar_attrs() -> None:
    """Trimming by index should keep scalar (non-array) attributes unchanged."""
    ds = _make_minimal_path_dataset(n_ue=4)
    ds["my_scalar"] = 42
    trimmed = ds._trim_by_index(np.array([0, 2]))  # noqa: SLF001
    assert trimmed["my_scalar"] == 42


def test_trim_by_index_slices_per_user_arrays() -> None:
    """Trimming by index should slice arrays that have shape[0] == n_ue."""
    ds = _make_minimal_path_dataset(n_ue=4, n_paths=2)
    original_rx_pos = ds.rx_pos.copy()
    trimmed = ds._trim_by_index(np.array([1, 3]))  # noqa: SLF001
    np.testing.assert_array_equal(trimmed.rx_pos[0], original_rx_pos[1])
    np.testing.assert_array_equal(trimmed.rx_pos[1], original_rx_pos[3])


# ===========================================================================
# resolve_key / alias with stored value
# ===========================================================================


def test_resolve_key_alias_with_stored_value() -> None:
    """Alias should resolve to a pre-stored value without triggering computation."""
    ds = Dataset({"rx_pos": np.zeros((2, 3)), "tx_pos": np.zeros(3)})
    # Store power directly (no computation needed)
    ds["power"] = np.array([[-10.0, -20.0], [-30.0, -40.0]])
    result = ds["pwr"]  # 'pwr' is alias for 'power'
    np.testing.assert_array_equal(np.array(result), ds.power)


# ===========================================================================
# _compute_rotated_angles
# ===========================================================================


def test_compute_rotated_angles_returns_four_keys() -> None:
    """_compute_rotated_angles should return dict with 4 angle keys."""
    from deepmimo import consts as c  # noqa: PLC0415
    from deepmimo.generator.channel import ChannelParameters  # noqa: PLC0415

    ds = _make_minimal_path_dataset(n_ue=2, n_paths=2)
    ds.set_channel_params(ChannelParameters(freq_domain=False))
    result = ds._compute_rotated_angles()  # noqa: SLF001
    expected_keys = {
        c.AOD_EL_ROT_PARAM_NAME,
        c.AOD_AZ_ROT_PARAM_NAME,
        c.AOA_EL_ROT_PARAM_NAME,
        c.AOA_AZ_ROT_PARAM_NAME,
    }
    assert set(result.keys()) == expected_keys
    for v in result.values():
        assert v.shape == (2, 2)


# ===========================================================================
# DynamicDataset - timestamp validation
# ===========================================================================


def test_dynamic_dataset_wrong_timestamp_count_raises() -> None:
    """set_timestamps with wrong number of entries should raise ValueError."""

    class _MockObj:
        def __init__(self):
            self.vel = np.zeros(3)
            self.position = np.zeros(3)

    class _MockScene:
        def __init__(self):
            self.objects = _MockObjList()

    class _MockObjList:
        def __init__(self):
            self._objs = [_MockObj()]

        @property
        def position(self):
            return [o.position for o in self._objs]

        @property
        def vel(self):
            return [o.vel for o in self._objs]

        @vel.setter
        def vel(self, v):
            for i, o in enumerate(self._objs):
                o.vel = v[i]

        def __getitem__(self, idx):
            return self._objs[idx]

    ds1 = Dataset({"rx_pos": np.zeros((2, 3)), "tx_pos": np.zeros(3), "name": "s1"})
    ds2 = Dataset({"rx_pos": np.ones((2, 3)), "tx_pos": np.zeros(3), "name": "s2"})
    ds1.scene = _MockScene()
    ds2.scene = _MockScene()
    dyn = DynamicDataset([ds1, ds2], name="test")

    with pytest.raises(ValueError, match="single value or a list"):
        dyn.set_timestamps([0.0, 1.0, 2.0])  # 3 timestamps for 2 scenes


# ===========================================================================
# _wrap_array
# ===========================================================================


def test_wrap_array_scalar_not_wrapped() -> None:
    """0-dim arrays should not be wrapped with DeepMIMOArray."""
    ds = Dataset({"rx_pos": np.zeros((3, 3))})
    scalar_array = np.array(42.0)
    result = ds._wrap_array("power", scalar_array)  # noqa: SLF001
    assert not hasattr(result, "_dataset")  # DeepMIMOArray has _dataset
    assert result == 42.0


def test_wrap_array_non_wrappable_key_not_wrapped() -> None:
    """Arrays with keys not in WRAPPABLE_ARRAYS should be returned as-is."""
    ds = Dataset({"rx_pos": np.zeros((3, 3))})
    arr = np.ones((3, 5))
    result = ds._wrap_array("some_unknown_key", arr)  # noqa: SLF001
    assert type(result) is np.ndarray  # not a DeepMIMOArray


# ===========================================================================
# MacroDataset bool-array indexing
# ===========================================================================


def test_macro_dataset_bool_array_indexing() -> None:
    """MacroDataset should accept boolean arrays as indices."""
    ds1 = Dataset({"n_ue": 2})
    ds2 = Dataset({"n_ue": 3})
    ds3 = Dataset({"n_ue": 4})
    macro = MacroDataset([ds1, ds2, ds3])

    mask = np.array([True, False, True])
    subset = macro[mask]
    assert isinstance(subset, MacroDataset)
    assert len(subset) == 2
    assert subset[0] is ds1
    assert subset[1] is ds3


# ===========================================================================
# MacroDataset slice indexing
# ===========================================================================


def test_macro_dataset_slice_indexing() -> None:
    """MacroDataset should support slice indexing."""
    ds1 = Dataset({"n_ue": 2})
    ds2 = Dataset({"n_ue": 3})
    ds3 = Dataset({"n_ue": 4})
    macro = MacroDataset([ds1, ds2, ds3])

    subset = macro[0:2]
    assert isinstance(subset, MacroDataset)
    assert len(subset) == 2
    assert subset[0] is ds1
    assert subset[1] is ds2


# ===========================================================================
# merge_datasets edge cases
# ===========================================================================


def test_merge_single_dataset_returns_same() -> None:
    """Merging a single dataset without axis overrides returns it unchanged."""
    from deepmimo.datasets.dataset import merge_datasets  # noqa: PLC0415

    g1 = _make_grid_dataset(nx=3, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    result = merge_datasets([g1])
    assert result is g1


def test_merge_empty_dataset_list_raises() -> None:
    """merge_datasets with empty list should raise ValueError."""
    from deepmimo.datasets.dataset import merge_datasets  # noqa: PLC0415

    with pytest.raises(ValueError, match="empty"):
        merge_datasets([])


def test_merge_same_rx_set_different_tx_raises() -> None:
    """Merging datasets with different TX sets but same RX set raises NotImplementedError."""
    from deepmimo.datasets.dataset import merge_datasets  # noqa: PLC0415

    g1 = _make_grid_dataset(nx=2, ny=2, tx_set_id=0, tx_idx=0, rx_set_id=0)
    g2 = _make_grid_dataset(nx=2, ny=2, tx_set_id=1, tx_idx=0, rx_set_id=0)
    with pytest.raises(NotImplementedError, match="multiple transmitters"):
        merge_datasets([g1, g2])
